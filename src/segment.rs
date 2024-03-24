#![allow(unstable_name_collisions)]

use crate::page::{
    AllocatedBlock, DelayedMode, FreeBlock, Page, PageKind, LARGE_PAGE_SHIFT, LARGE_PAGE_SIZE,
    MEDIUM_PAGE_SHIFT, SMALL_PAGE_SIZE,
};
use crate::{
    heap::Heap,
    linked_list::{List, Node},
    page::SMALL_PAGE_SHIFT,
    SMALL_OBJ_SIZE_MAX,
};
use crate::{rem, system, thread_id, Ptr, LARGE_OBJ_SIZE_MAX, MEDIUM_OBJ_SIZE_MAX, WORD_SIZE};
use core::alloc::Layout;
use sptr::Strict;
use std::cell::Cell;
use std::cmp::max;
use std::intrinsics::unlikely;
use std::num::NonZeroUsize;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

pub const SEGMENT_SHIFT: usize = 21; // 2MiB

const _: () = {
    assert!(SEGMENT_SHIFT >= LARGE_PAGE_SHIFT);
};

pub const SEGMENT_SIZE: usize = 1 << SEGMENT_SHIFT;
pub const SEGMENT_ALIGN: usize = SEGMENT_SIZE;
pub const SEGMENT_MASK: usize = SEGMENT_ALIGN - 1;

pub const SMALL_PAGES_PER_SEGMENT: usize = SEGMENT_SIZE / SMALL_PAGE_SIZE;
//pub const MEDIUM_PAGES_PER_SEGMENT: usize = SEGMENT_SIZE / MEDIUM_PAGE_SIZE;
#[allow(dead_code)] // Actually used
const LARGE_PAGES_PER_SEGMENT: usize = SEGMENT_SIZE / LARGE_PAGE_SIZE;

const _: () = {
    // We need large and huge pages to only have one page per segment to prevent
    // them from calling `insert_in_free_queue`.
    assert!(LARGE_PAGES_PER_SEGMENT == 1);
};

pub const OPTION_EAGER_COMMIT: bool = false;
pub const OPTION_EAGER_DELAY_COUNT: usize = 1;
pub const OPTION_PURGE_DELAY: isize = 10;
pub const OPTION_PURGE_DOES_DECOMMIT: bool = true;

pub struct WholeSegmentOrStatic;

pub struct WholeSegment;

/// A raw pointer which has provenance to the whole segment allocation.
pub type Whole<T> = Ptr<T, WholeSegment>;

/// A raw pointer which has either provenance to the whole segment allocation
/// or it points to 'static data. The static data is used for empty `Page`s for new heaps.
pub type WholeOrStatic<T> = Ptr<T, WholeSegmentOrStatic>;

impl<T> Whole<T> {
    pub fn or_static(&self) -> WholeOrStatic<T> {
        unsafe { WholeOrStatic::new_unchecked(self.as_ptr()) }
    }
}

// Segments thread local data
#[derive(Debug)]
pub struct SegmentThreadData {
    small_free: List<Whole<Segment>>, // queue of segments with free small pages
    medium_free: List<Whole<Segment>>, // queue of segments with free medium pages
    pub pages_purge: List<Whole<Page>>, // queue of freed pages that are delay purged
    current_count: usize,             // current number of segments;
    count: usize,                     // total number of segments allocated,
    reclaim_count: usize,             // number of reclaimed (abandoned) segments
}

impl SegmentThreadData {
    unsafe fn free_queue_of_kind(
        data: &mut SegmentThreadData,
        kind: PageKind,
    ) -> Option<&mut List<Whole<Segment>>> {
        match kind {
            PageKind::Small => Some(&mut data.small_free),
            PageKind::Medium => Some(&mut data.medium_free),
            _ => None,
        }
    }

    fn add_segment(&mut self) {
        self.current_count += 1;
        self.count = self.count.saturating_add(1);
    }

    fn remove_segment(&mut self) {
        self.current_count -= 1;
    }
}

impl SegmentThreadData {
    pub const fn initial() -> Self {
        SegmentThreadData {
            small_free: List::empty(),
            medium_free: List::empty(),
            pages_purge: List::empty(), // queue of freed pages that are delay purged
            current_count: 0,           // current number of segments;
            count: 0,
            reclaim_count: 0, // number of reclaimed (abandoned) segments
        }
    }
}

#[derive(Debug)]
#[repr(C)]
pub struct Segment {
    // Cluster fields accessed in the alloc / free hot paths
    page_shift: usize, // `1 << page_shift` == the page sizes == `page->block_size * page->reserved` (unless the first page, then `-segment_info_size`).
    pub thread_id: AtomicUsize, // unique id of the thread owning this segment
    pub page_kind: PageKind, // kind of pages: small, medium, large, or huge

    // constant fields
    allow_decommit: bool,
    allow_purge: bool,
    segment_size: usize, // for huge pages this may be different from `SEGMENT_SIZE`
    system_alloc: system::SystemAllocation,

    was_reclaimed: Cell<bool>, // true if it was reclaimed (used to limit on-free reclamation)
    abandoned: Cell<usize>, // abandoned pages (i.e. the original owning thread stopped) (`abandoned <= used`)
    abandoned_visits: Cell<usize>, // count how often this segment is visited in the abandoned list (to force reclaim if it is too long)
    pub used: Cell<usize>,         // count of pages in use (`used <= capacity`)
    capacity: usize,               // count of available pages (`#free + used`)
    segment_info_size: usize, // space we are using from the first page for segment meta-data and possible guard pages.

    #[cfg(debug_assertions)]
    cookie: usize, // verify addresses in secure mode: `_mi_ptr_cookie(segment) == segment->cookie`

    node: Node<Whole<Segment>>,
}

#[cfg(debug_assertions)]
#[allow(overflowing_literals)]
fn cookie(ptr: Whole<Segment>) -> usize {
    ptr.as_ptr().addr() ^ 0xa0f7b251664141e9
}

impl Segment {
    #[inline]
    fn node(segment: Whole<Segment>) -> *mut Node<Whole<Segment>> {
        unsafe { &mut (*segment.as_ptr()).node }
    }

    #[inline]
    pub unsafe fn page(segment: Whole<Segment>, index: usize) -> Whole<Page> {
        Whole::new_unchecked(segment.as_ptr().add(1).cast::<Page>().add(index))
    }

    #[inline]
    unsafe fn page_index_from_pointer(
        segment: Whole<Segment>,
        ptr: Whole<AllocatedBlock>,
    ) -> usize {
        let diff = ptr.as_ptr().byte_offset_from(segment.as_ptr());
        debug_assert!(
            diff >= 0 && (diff as usize) <= SEGMENT_SIZE /* for huge alignment it can be equal */
        );
        let idx = (diff as usize) >> segment.page_shift;
        debug_assert!(idx < segment.capacity);
        debug_assert!(segment.page_kind <= PageKind::Medium || idx == 0);
        idx
    }

    #[inline]
    pub unsafe fn page_from_pointer(
        segment: Whole<Segment>,
        ptr: Whole<AllocatedBlock>,
    ) -> Whole<Page> {
        Segment::page(segment, Segment::page_index_from_pointer(segment, ptr))
    }

    #[inline]
    unsafe fn has_free(&self) -> bool {
        self.used.get() < self.capacity
    }

    #[inline]
    pub unsafe fn is_local(&self) -> bool {
        thread_id() == self.thread_id.load(Ordering::Relaxed)
    }

    #[inline]
    pub unsafe fn from_pointer_checked(ptr: Whole<AllocatedBlock>) -> Whole<Segment> {
        debug_assert!(ptr.as_ptr().addr() & !(WORD_SIZE - 1) != 0);

        let segment = Segment::from_pointer(ptr);

        #[cfg(debug_assertions)]
        debug_assert_eq!((*segment.as_ptr()).cookie, cookie(segment));

        segment
    }

    // Segment that contains the pointer
    // Large aligned blocks may be aligned at N*SEGMENT_SIZE (inside a huge segment > SEGMENT_SIZE),
    // and we need align "down" to the segment info which is `SEGMENT_SIZE` bytes before it;
    // therefore we align one byte before `p`.
    #[inline]
    pub unsafe fn from_pointer(ptr: Whole<AllocatedBlock>) -> Whole<Segment> {
        ptr.map_addr(|addr| (addr - 1) & !SEGMENT_MASK).cast()
    }

    #[inline]
    unsafe fn raw_page_size(&self) -> usize {
        if self.page_kind == PageKind::Huge {
            self.segment_size
        } else {
            1 << self.page_shift
        }
    }

    // Raw start of the page available memory; can be used on uninitialized pages (only `segment_idx` must be set)
    // The raw start is not taking aligned block allocation into consideration.
    unsafe fn raw_page_start(segment: Whole<Segment>, page: Whole<Page>) -> (Whole<u8>, usize) {
        let mut page_size = segment.raw_page_size();
        let mut p = segment
            .as_ptr()
            .cast::<u8>()
            .add(page.segment_idx as usize * page_size);

        if page.segment_idx == 0 {
            debug_assert!(page_size >= segment.segment_info_size);
            // the first page starts after the segment info (and possible guard page)
            p = p.add(segment.segment_info_size);
            page_size -= segment.segment_info_size;
        }

        (Whole::new_unchecked(p), page_size)
    }

    // Start of the page available memory; can be used on uninitialized pages (only `segment_idx` must be set)
    pub unsafe fn page_start(
        segment: Whole<Segment>,
        page: Whole<Page>,
        block_size: usize,
        pre_size: &mut usize,
    ) -> (Whole<u8>, usize) {
        let (mut p, mut page_size) = Segment::raw_page_start(segment, page);
        *pre_size = 0;
        if let Some(non_zero_block_size) = NonZeroUsize::new(block_size) {
            if page.segment_idx == 0 && segment.page_kind <= PageKind::Medium {
                debug_assert!(non_zero_block_size.get() <= LARGE_OBJ_SIZE_MAX);

                // for small and medium objects, ensure the page start is aligned with the block size (PR#66 by kickunderscore)
                let adjust = block_size - rem(p.addr(), non_zero_block_size);
                if page_size - adjust >= block_size && adjust < block_size {
                    p = Whole::new_unchecked(p.as_ptr().add(adjust));
                    page_size -= adjust;
                    *pre_size = adjust;
                }
            }
        }

        (p, page_size)
    }

    /* -----------------------------------------------------------
       Page allocation
    ----------------------------------------------------------- */

    unsafe fn page_ensure_committed(
        segment: Whole<Segment>,
        page: Whole<Page>,
        _data: &mut SegmentThreadData,
    ) -> bool {
        if page.is_committed.get() {
            return true;
        }

        debug_assert!(segment.allow_decommit);
        // mi_assert_expensive(!mi_pages_purge_contains(page, tld));

        let (start, psize) = Segment::raw_page_start(segment, page);

        //     let is_zero = false;

        if !system::commit(Ptr::new_unchecked(start.as_ptr()), psize) {
            return false;
        } // failed to commit!
        page.is_committed.set(true);

        page.used.set(0);
        // (*page).is_zero_init = false;

        true
    }

    #[inline]
    unsafe fn page_claim(
        segment: Whole<Segment>,
        page: Whole<Page>,
        data: &mut SegmentThreadData,
    ) -> bool {
        debug_assert!(Page::segment(page) == segment);
        debug_assert!(!page.segment_in_use.get());

        Page::purge_remove(page, data);

        // check commit
        if !Segment::page_ensure_committed(segment, page, data) {
            return false;
        }

        // set in-use before doing unreset to prevent delayed reset
        page.segment_in_use.set(true);
        segment.used.set(segment.used.get() + 1);
        debug_assert!(
            page.segment_in_use.get() && page.is_committed.get() && page.used.get() == 0 /* && !mi_pages_purge_contains(page, tld) */
        );
        debug_assert!(segment.used.get() <= segment.capacity);
        if segment.used.get() == segment.capacity && segment.page_kind <= PageKind::Medium {
            // if no more free pages, remove from the queue
            debug_assert!(!segment.has_free());
            Segment::remove_from_free_queue(segment, data);
        }
        true
    }

    #[inline]
    unsafe fn find_free(
        segment: Whole<Segment>,
        data: &mut SegmentThreadData,
    ) -> Option<Whole<Page>> {
        //   mi_assert_internal(mi_segment_has_free(segment));
        //  mi_assert_expensive(mi_segment_is_valid(segment, tld));
        for i in 0..segment.capacity {
            // TODO: use a bitmap instead of search?
            let page = Segment::page(segment, i);
            if !page.segment_in_use.get() && Segment::page_claim(segment, page, data) {
                return Some(page);
            }
        }
        None
    }

    // Possibly clear pages and check if free space is available
    unsafe fn check_free(segment: Whole<Segment>, block_size: usize) -> (bool, bool) {
        debug_assert!(block_size <= LARGE_OBJ_SIZE_MAX);
        let mut has_page = false;
        let mut pages_used = 0;
        let mut pages_used_empty = 0;
        for i in 0..segment.capacity {
            let page = Segment::page(segment, i);
            if page.segment_in_use.get() {
                pages_used += 1;
                // ensure used count is up to date and collect potential concurrent frees
                Page::free_collect(page, false);
                if page.all_free() {
                    // if everything free already, page can be reused for some block size
                    // note: don't clear the page yet as we can only OS reset it once it is reclaimed
                    pages_used_empty += 1;
                    has_page = true;
                } else if page.block_size() == block_size && page.any_available() {
                    // a page has available free blocks of the right size
                    has_page = true;
                }
            } else {
                // whole empty page
                has_page = true;
            }
        }
        debug_assert!(pages_used == segment.used.get() && pages_used >= pages_used_empty);
        (has_page, ((pages_used - pages_used_empty) == 0))
    }

    // Reclaim a segment; returns NULL if the segment was freed
    // set `right_page_reclaimed` to `true` if it reclaimed a page of the right `block_size` that was not full.
    unsafe fn reclaim(
        segment: Whole<Segment>,
        heap: Ptr<Heap>,
        requested_block_size: usize,
        right_page_reclaimed: &mut bool,
        data: &mut SegmentThreadData,
    ) -> Option<Whole<Segment>> {
        *right_page_reclaimed = false;

        // can be 0 still with abandoned_next, or already a thread id for segments outside an arena that are reclaimed on a free.
        debug_assert!(
            segment.thread_id.load(Ordering::Relaxed) == 0
                || segment.thread_id.load(Ordering::Relaxed) == thread_id()
        );
        segment.thread_id.store(thread_id(), Ordering::Release);
        segment.abandoned_visits.set(0);
        segment.was_reclaimed.set(true);
        data.reclaim_count += 1;
        data.add_segment();
        debug_assert!(!segment.node.used());
        //mi_assert_expensive(mi_segment_is_valid(segment, tld));
        //_mi_stat_decrease(&tld.stats.segments_abandoned, 1);

        for i in 0..segment.capacity {
            let page = Segment::page(segment, i);
            if page.segment_in_use.get() {
                debug_assert!(page.is_committed.get());
                debug_assert!(!page.node.used());
                debug_assert!(page.thread_free_flag() == DelayedMode::NeverDelayedFree);
                debug_assert!(page.heap().is_none());
                segment.abandoned.set(segment.abandoned.get() - 1);
                //_mi_stat_decrease(&tld.stats.pages_abandoned, 1);
                // set the heap again and allow heap thread delayed free again.
                page.set_heap(Some(heap));
                Page::use_delayed_free(page, DelayedMode::UseDelayedFree, true); // override never (after heap is set)
                Page::free_collect(page, false); // ensure used count is up to date
                if page.all_free() {
                    // if everything free already, clear the page directly
                    Segment::page_clear(segment, page, data); // reset is ok now
                } else {
                    // otherwise reclaim it into the heap
                    Page::reclaim(page, heap);
                    if requested_block_size == page.block_size() && page.any_available() {
                        *right_page_reclaimed = true;
                    }
                }
            }
            /* expired
            else if (page.is_committed) {  // not in-use, and not reset yet
              // note: do not reset as this includes pages that were not touched before
              // mi_pages_purge_add(segment, page, tld);
            }
            */
        }
        debug_assert!(segment.abandoned.get() == 0);
        if segment.used.get() == 0 {
            debug_assert!(!(*right_page_reclaimed));
            Segment::free(segment, data);
            None
        } else {
            if segment.page_kind <= PageKind::Medium && segment.has_free() {
                Segment::insert_in_free_queue(segment, data);
            }
            Some(segment)
        }
    }

    unsafe fn try_reclaim(
        heap: Ptr<Heap>,
        block_size: usize,
        page_kind: PageKind,
        reclaimed: &mut bool,
        data: &mut SegmentThreadData,
    ) -> Option<Whole<Segment>> {
        *reclaimed = false;

        // FIXME: Limit segment tries.

        let mut result = None;

        walk_abandoned(|segment| {
            segment
                .abandoned_visits
                .set(segment.abandoned_visits.get() + 1);
            // todo: an arena exclusive heap will potentially visit many abandoned unsuitable segments
            // and push them into the visited list and use many tries. Perhaps we can skip non-suitable ones in a better way?
            let (has_page, all_pages_free) = Segment::check_free(segment, block_size); // try to free up pages (due to concurrent frees)
            if all_pages_free {
                // free the segment (by forced reclaim) to make it available to other threads.
                // note1: we prefer to free a segment as that might lead to reclaiming another
                // segment that is still partially used.
                // note2: we could in principle optimize this by skipping reclaim and directly
                // freeing but that would violate some invariants temporarily)
                Segment::reclaim(segment, heap, 0, &mut false, data);
                false
            } else if (has_page && segment.page_kind == page_kind) && result.is_none() {
                // found a free page of the right kind, or page of the right block_size with free space
                // we return the result of reclaim (which is usually `segment`) as it might free
                // the segment due to concurrent frees (in which case `NULL` is returned).
                result = Some(Segment::reclaim(segment, heap, block_size, reclaimed, data));
                false
            } else if segment.abandoned_visits.get() >= 3 {
                // always reclaim on 3rd visit to limit the list length.
                Segment::reclaim(segment, heap, 0, &mut false, data);
                false
            } else {
                // otherwise, mark it back as abandoned
                // todo: reset delayed pages in the segment?
                true
            }
        });
        result.and_then(|r| r)
    }

    fn calculate_sizes(
        capacity: usize,
        required: usize,
        metadata_size: &mut usize,
        page_alignment: usize,
    ) -> Option<Layout> {
        let segment = Layout::new::<Segment>();
        let pages = Layout::array::<Page>(capacity).unwrap();
        let metadata = segment
            .extend(pages)
            .unwrap()
            .0
            .align_to(WORD_SIZE)
            .unwrap()
            .pad_to_align();

        assert!(metadata.align() <= SEGMENT_ALIGN);

        // Pad to the requested page alignment.
        let layout = metadata.align_to(page_alignment).ok()?.pad_to_align();

        *metadata_size = layout.size();

        let size = if required == 0 {
            debug_assert!(layout.size() <= SEGMENT_SIZE);
            SEGMENT_SIZE
        } else {
            layout.size().checked_add(required)?
        };

        let alignment = max(SEGMENT_SIZE, page_alignment);

        Layout::from_size_align(size, alignment).ok()
    }

    // Allocate a segment from the OS aligned to `SEGMENT_SIZE` .
    unsafe fn alloc(
        required: usize,
        page_kind: PageKind,
        page_shift: usize,
        page_alignment: usize,
        data: &mut SegmentThreadData,
    ) -> Option<Whole<Segment>> {
        // required is only > 0 for huge page allocations
        debug_assert!(
            (required > 0 && page_kind > PageKind::Large)
                || (required == 0 && page_kind <= PageKind::Large)
        );

        // calculate needed sizes first
        let capacity;
        if page_kind == PageKind::Huge {
            debug_assert!(page_shift == SEGMENT_SHIFT + 1 && required > 0);
            capacity = 1;
        } else {
            debug_assert!(required == 0 && page_alignment == 0);
            let page_size = 1 << page_shift;
            capacity = SEGMENT_SIZE / page_size;
            debug_assert!(SEGMENT_SIZE % page_size == 0);
            debug_assert!((1..=SMALL_PAGES_PER_SEGMENT).contains(&capacity));
        }
        let mut metadata_size = 0;
        let layout =
            Segment::calculate_sizes(capacity, required, &mut metadata_size, page_alignment)?;
        debug_assert!(layout.size() - metadata_size >= required);

        let commit = OPTION_EAGER_COMMIT ||
        page_kind > PageKind::Medium // don't delay for large objects
        || system::has_overcommit() // never delay on overcommit systems

        // This seems bad
        //  || THREAD_COUNT.load(Ordering::Relaxed) <= 1 // do not delay for the first N threads

       || data.count >= OPTION_EAGER_DELAY_COUNT;

        let (system_alloc, allocation, initially_committed) = system::alloc(layout, commit)?;

        debug_assert!(allocation.as_ptr().is_aligned_to(layout.align()));
        debug_assert!(allocation.as_ptr().is_aligned_to(SEGMENT_SIZE));

        let segment: Whole<Segment> = Whole::new_unchecked(allocation.as_ptr().cast());

        if !initially_committed && (!system::commit(allocation, metadata_size)) {
            // commit failed; we cannot touch the memory: free the segment directly and return `null_mut()`
            system::dealloc(system_alloc, allocation, layout);
            return None;
        }

        data.add_segment();

        ptr::write(
            segment.as_ptr(),
            Segment {
                allow_decommit: !initially_committed,
                allow_purge: segment.allow_decommit && (OPTION_PURGE_DELAY >= 0),
                system_alloc,
                segment_size: layout.size(),
                was_reclaimed: Cell::new(false),
                abandoned: Cell::new(0),
                abandoned_visits: Cell::new(0),
                used: Cell::new(0),
                capacity,
                segment_info_size: metadata_size,
                #[cfg(debug_assertions)]
                cookie: cookie(segment),
                node: Node::UNUSED,
                thread_id: AtomicUsize::new(thread_id()),
                page_shift,
                page_kind,
            },
        );

        // initialize pages info
        for i in 0..capacity {
            debug_assert!(i <= 255);
            let page = Segment::page(segment, i);
            (*page.as_ptr()).segment_idx = i as u8;
            page.is_committed.set(initially_committed);
            // page.is_zero_init = (*segment).memid.initially_zero;

            // Note: Page is partially initialized.
        }

        // insert in free lists for small and medium pages
        if page_kind <= PageKind::Medium {
            Segment::insert_in_free_queue(segment, data)
        }

        Some(segment)
    }

    unsafe fn reclaim_or_alloc(
        heap: Ptr<Heap>,
        block_size: usize,
        page_kind: PageKind,
        page_shift: usize,
        data: &mut SegmentThreadData,
    ) -> Option<Whole<Segment>> {
        debug_assert!(page_kind <= PageKind::Large);
        debug_assert!(block_size <= LARGE_OBJ_SIZE_MAX);

        // 1. try to reclaim an abandoned segment
        let mut reclaimed = false;
        let segment = Segment::try_reclaim(heap, block_size, page_kind, &mut reclaimed, data);
        // debug_assert!(segment.is_null() || _mi_arena_memid_is_suitable(segment->memid, heap->arena_id));
        if reclaimed {
            // reclaimed the right page right into the heap
            // FIXME
            /*     debug_assert!(
                !segment.is_null()
                    && (*segment).page_kind == page_kind
                    && page_kind <= PageKind::Large
            );*/
            return None; // pretend out-of-memory as the page will be in the page queue of the heap with available blocks
        } else if let Some(segment) = segment {
            // reclaimed a segment with empty pages (of `page_kind`) in it
            return Some(segment);
        }
        // 2. otherwise allocate a fresh segment

        Segment::alloc(0, page_kind, page_shift, 0, data)
    }

    unsafe fn page_try_alloc_in_queue(
        _heap: Ptr<Heap>,
        free_queue: impl Fn(&mut SegmentThreadData) -> &mut List<Whole<Segment>>,
        data: &mut SegmentThreadData,
    ) -> Option<Whole<Page>> {
        // find an available segment the segment free queue
        for segment in free_queue(data).iter(Segment::node) {
            /* if (_mi_arena_memid_is_suitable(segment->memid, heap->arena_id) && mi_segment_has_free(segment)) {
              return mi_segment_page_alloc_in(segment, tld);
            }  */
            if segment.has_free() {
                return Segment::find_free(segment, data);
            }
        }
        None
    }

    unsafe fn alloc_page_of_kind(
        heap: Ptr<Heap>,
        block_size: usize,
        kind: PageKind,
        page_shift: usize,
        data: &mut SegmentThreadData,
        free_queue: impl Fn(&mut SegmentThreadData) -> &mut List<Whole<Segment>> + Clone,
    ) -> Option<Whole<Page>> {
        let page = Segment::page_try_alloc_in_queue(heap, free_queue.clone(), data);
        if page.is_none() {
            // possibly allocate or reclaim a fresh segment
            if let Some(segment) =
                Segment::reclaim_or_alloc(heap, block_size, kind, page_shift, data)
            {
                debug_assert!(segment.page_kind == kind);
                debug_assert!(segment.used.get() < segment.capacity);
                //   debug_assert!(_mi_arena_memid_is_suitable((*segment).memid, heap->arena_id));
                Segment::page_try_alloc_in_queue(heap, free_queue, data)
            // this should now succeed
            } else {
                None
            }
        } else {
            page
        }
    }

    unsafe fn alloc_large_page(
        heap: Ptr<Heap>,
        block_size: usize,
        data: &mut SegmentThreadData,
    ) -> Option<Whole<Page>> {
        if let Some(segment) =
            Segment::reclaim_or_alloc(heap, block_size, PageKind::Large, LARGE_PAGE_SHIFT, data)
        {
            let page = Segment::find_free(segment, data);
            debug_assert!(page.is_some());
            page
        } else {
            None
        }
    }

    // reset memory of a huge block from another thread
    pub unsafe fn huge_page_reset(
        segment: Whole<Segment>,
        page: Whole<Page>,
        _block: Whole<FreeBlock>,
    ) {
        debug_assert!(segment.page_kind == PageKind::Huge);
        debug_assert!(segment == Page::segment(page));
        debug_assert!(page.used.get() == 1); // this is called just before the free
        debug_assert!(page.free_blocks.is_empty());

        // FIXME
        /*
        if (*(segment).allow_decommit && (*page).is_committed) {
          let usize = mi_usable_size(block);
          if (usize > sizeof(mi_block_t)) {
            usize = usize - sizeof(mi_block_t);
            uint8_t* p = (uint8_t*)block + sizeof(mi_block_t);
            _mi_os_reset(p, usize, &_mi_stats_main);
          }
        }
        */
    }

    pub unsafe fn alloc_huge_page(
        size: usize,
        page_alignment: usize,
        data: &mut SegmentThreadData,
    ) -> Option<Whole<Page>> {
        let segment = Segment::alloc(
            size,
            PageKind::Huge,
            SEGMENT_SHIFT + 1,
            page_alignment,
            data,
        )?;

        let page = Segment::find_free(segment, data).unwrap_unchecked();

        let (start, psize) = Segment::page_start(segment, page, 0, &mut 0);
        debug_assert!(psize >= size);
        debug_assert!(start.as_ptr().is_aligned_to(page_alignment));

        /*
        // reset the part of the page that will not be used; this can be quite large (close toSEGMENT_SIZE)
        if (page_alignment > 0 && (*segment).allow_decommit && (*page).is_committed.get()) {
            let aligned_p = start.map_addr(|addr| align_up(addr, page_alignment));
            debug_assert!(psize - (aligned_p.offset_from(start) >= size));
            let decommit_start = start + mem::size_of::<Block>(); // for the free list
            let decommit_size = aligned_p - decommit_start;
            // FIXME
              _mi_os_reset(decommit_start, decommit_size, os_tld->stats);  // do not decommit as it may be in a region
        }*/

        Some(page)
    }

    pub unsafe fn page_alloc(
        heap: Ptr<Heap>,
        block_size: usize,
        page_alignment: usize,
        data: &mut SegmentThreadData,
    ) -> Option<Whole<Page>> {
        let page = if unlikely(page_alignment > WORD_SIZE) {
            debug_assert!(page_alignment.is_power_of_two());
            Segment::alloc_huge_page(block_size, page_alignment, data)
        } else if block_size <= SMALL_OBJ_SIZE_MAX {
            Segment::alloc_page_of_kind(
                heap,
                block_size,
                PageKind::Small,
                SMALL_PAGE_SHIFT,
                data,
                |data| &mut data.small_free,
            )
        } else if block_size <= MEDIUM_OBJ_SIZE_MAX {
            Segment::alloc_page_of_kind(
                heap,
                block_size,
                PageKind::Medium,
                MEDIUM_PAGE_SHIFT,
                data,
                |data| &mut data.medium_free,
            )
        } else if block_size <= LARGE_OBJ_SIZE_MAX
        /* || mi_is_good_fit(block_size, LARGE_PAGE_SIZE - sizeof(mi_segment_t)) */
        {
            Segment::alloc_large_page(heap, block_size, data)
        } else {
            Segment::alloc_huge_page(block_size, page_alignment, data)
        };

        if let Some(page) = page {
            //  mi_assert_expensive( mi_segment_is_valid(_mi_page_segment(page),tld));
            debug_assert!(Page::segment(page).raw_page_size() >= block_size);
            // mi_segment_try_purge(tld);
            //  debug_assert!( mi_page_not_in_queue(page, tld));
        }
        page
    }

    unsafe fn pages_try_purge(data: &mut SegmentThreadData) {
        let now = system::clock_now();

        // from oldest up to the first that has not expired yet
        for page in data.pages_purge.iter_rev(Page::node) {
            if page.purge_is_expired(now) {
                data.pages_purge.remove(page, Page::node); // remove from the list to maintain invariant for mi_page_purge
                Segment::page_purge(Page::segment(page), page);
            } else {
                break;
            }
        }

        // FIXME: Optimize clearing out all purged pages. We could remove all at the end here.
    }

    pub unsafe fn page_abandon(page: Whole<Page>, data: &mut SegmentThreadData) {
        debug_assert!((*page).thread_free_flag() == DelayedMode::NeverDelayedFree);
        debug_assert!((*page).heap().is_none());

        let segment = Page::segment(page);
        //   mi_assert_expensive(!mi_pages_purge_contains(page, tld));
        //   mi_assert_expensive(mi_segment_is_valid(segment, tld));
        segment.abandoned.set(segment.abandoned.get() + 1);
        //    _mi_stat_increase(&tld->stats->pages_abandoned, 1);
        debug_assert!(segment.abandoned.get() <= segment.used.get());
        if segment.used.get() == segment.abandoned.get() {
            // all pages are abandoned, abandon the entire segment
            Segment::abandon(segment, data);
        }
    }

    /* -----------------------------------------------------------
      Page reset
    ----------------------------------------------------------- */
    unsafe fn page_purge(segment: Whole<Segment>, page: Whole<Page>) {
        if !segment.allow_purge {
            return;
        }
        let (start, psize) = Segment::raw_page_start(segment, page);
        let needs_recommit = system::purge(start.as_ptr(), psize, true);
        if needs_recommit {
            page.is_committed.set(false);
        }
        page.used.set(0);
    }

    unsafe fn schedule_purge(
        segment: Whole<Segment>,
        page: Whole<Page>,
        data: &mut SegmentThreadData,
    ) {
        if !segment.allow_purge {
            return;
        }

        if OPTION_PURGE_DELAY == 0 {
            // purge immediately?
            Segment::page_purge(segment, page);
        } else if OPTION_PURGE_DELAY > 0 {
            // no purging if the delay is negative
            // otherwise push on the delayed page reset queue
            page.purge_set_expire();
            data.pages_purge.push_front(page, Page::node);
        }
    }

    // clear page data; can be called on abandoned segments

    unsafe fn page_clear(segment: Whole<Segment>, page: Whole<Page>, data: &mut SegmentThreadData) {
        // zero the page data, but not the segment fields and capacity, and block_size (for page size calculations)

        *page.as_ptr() = Page {
            capacity: page.capacity.clone(),
            reserved: page.reserved.clone(),
            xblock_size: page.xblock_size.clone(),
            segment_idx: page.segment_idx,
            is_committed: page.is_committed.clone(),
            ..Page::empty()
        };

        segment.used.set(segment.used.get() - 1);

        // schedule purge
        Segment::schedule_purge(segment, page, data);

        page.capacity.set(0); // after purge these can be zero'd now
        page.reserved.set(0);
    }

    unsafe fn os_free(segment: Whole<Segment>, segment_size: usize, data: &mut SegmentThreadData) {
        // FIXME: Why are we changing `Segment` in this function?

        segment.thread_id.store(0, Ordering::Relaxed);

        // FIXME: Is this segment always local?
        data.remove_segment();

        if segment.was_reclaimed.get() {
            data.reclaim_count -= 1;
            segment.was_reclaimed.set(false);
        }

        let mut fully_committed = true;
        let mut committed_size = 0;
        let page_size = segment.raw_page_size();
        for i in 0..segment.capacity {
            let page = Segment::page(segment, i);
            if page.is_committed.get() {
                committed_size += page_size;
            }
            if !page.is_committed.get() {
                fully_committed = false;
            }
        }
        debug_assert!(
            (fully_committed && committed_size == segment_size)
                || (!fully_committed && committed_size < segment_size)
        );

        //  _mi_abandoned_await_readers(); // prevent ABA issue if concurrent readers try to access our memory (that might be purged)

        //  _mi_arena_free(segment, segment_size, committed_size, segment->memid, tld->stats);

        system::dealloc(
            segment.system_alloc,
            Ptr::new_unchecked(segment.as_ptr().cast()),
            Layout::from_size_align_unchecked(segment.segment_size, SEGMENT_ALIGN),
        );
    }

    unsafe fn remove_all_purges(
        segment: Whole<Segment>,
        force_purge: bool,
        data: &mut SegmentThreadData,
    ) {
        for i in 0..segment.capacity {
            let page = Segment::page(segment, i);
            if !page.segment_in_use.get() {
                Page::purge_remove(page, data);
                if force_purge && page.is_committed.get() {
                    Segment::page_purge(segment, page);
                }
            } else {
                // mi_assert_internal(mi_page_not_in_queue(page, tld));
            }
        }
    }

    unsafe fn free(segment: Whole<Segment>, data: &mut SegmentThreadData) {
        // don't purge as we are freeing now
        //  mi_segment_remove_all_purges(segment, false /* don't force as we are about to free */, tld);
        Segment::remove_from_free_queue(segment, data);

        // return it to the OS
        Segment::os_free(segment, segment.segment_size, data);
    }

    // remove from free queue if it is in one
    unsafe fn remove_from_free_queue(segment: Whole<Segment>, data: &mut SegmentThreadData) {
        let queue = SegmentThreadData::free_queue_of_kind(data, segment.page_kind);
        if let Some(queue) = queue {
            if (*queue).may_contain(segment, Segment::node) {
                (*queue).remove(segment, Segment::node);
            }
        }
    }

    // mark a specific segment as abandoned
    // clears the thread_id.
    unsafe fn mark_abandoned(segment: Whole<Segment>) {
        segment.thread_id.store(0, Ordering::Release);
        debug_assert!(segment.used == segment.abandoned);

        add_abandoned(segment);
    }

    /* -----------------------------------------------------------
       Abandon segment/page
    ----------------------------------------------------------- */

    unsafe fn abandon(segment: Whole<Segment>, data: &mut SegmentThreadData) {
        debug_assert!(segment.used.get() == segment.abandoned.get());
        debug_assert!(segment.used.get() > 0);

        // Potentially force purge. Only abandoned segments in arena memory can be
        // reclaimed without a free so if a segment is not from an arena we force purge here to be conservative.
        Segment::pages_try_purge(data);

        let force_purge = true; //  mi_option_is_enabled(mi_option_abandoned_page_purge);
        Segment::remove_all_purges(segment, force_purge, data);

        // remove the segment from the free page queue if needed
        Segment::remove_from_free_queue(segment, data);

        // all pages in the segment are abandoned; add it to the abandoned list

        //  mi_segments_track_size(-((long)segment.segment_size), data);

        segment.abandoned_visits.set(0);
        if segment.was_reclaimed.get() {
            data.reclaim_count -= 1;
            segment.was_reclaimed.set(false);
        }
        Segment::mark_abandoned(segment);
    }

    unsafe fn insert_in_free_queue(segment: Whole<Segment>, data: &mut SegmentThreadData) {
        let queue = SegmentThreadData::free_queue_of_kind(data, segment.page_kind);
        // FIXME, is this true? There doesn't seem to be a clear invariant here.
        debug_assert!(queue.is_some());
        queue.unwrap_unchecked().push_back(segment, Segment::node);
    }

    pub unsafe fn page_free(page: Whole<Page>, _force: bool, data: &mut SegmentThreadData) {
        let segment = Page::segment(page);
        Segment::pages_try_purge(data);

        // mark it as free now
        Segment::page_clear(segment, page, data);

        if segment.used.get() == 0 {
            // no more used pages; remove from the free list and free the segment
            Segment::free(segment, data);
        } else if segment.used.get() == segment.abandoned.get() {
            // only abandoned pages; remove from free list and abandon
            Segment::abandon(segment, data);
        } else if segment.used.get() + 1 == segment.capacity {
            Segment::insert_in_free_queue(segment, data);
        }
    }
}

/// A global list of abandoned segments.
static ABANDONED_SEGMENTS: Mutex<List<Whole<Segment>>> = Mutex::new(List::empty());

unsafe fn add_abandoned(segment: Whole<Segment>) {
    let guard = match ABANDONED_SEGMENTS.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    guard.push_back(segment, Segment::node);
}

unsafe fn walk_abandoned(mut f: impl FnMut(Whole<Segment>) -> bool) {
    let guard = match ABANDONED_SEGMENTS.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    for segment in guard.iter(Segment::node) {
        guard.remove(segment, Segment::node);
        if f(segment) {
            guard.push_front(segment, Segment::node);
        }
    }
}

/*
/// A global list of abandoned segments.
static ABANDONED_SEGMENTS: AtomicPtr<Segment> = AtomicPtr::new(null_mut());

unsafe fn add_abandoned(segment: Whole<Segment>) {
    let mut current = ABANDONED_SEGMENTS.load(Ordering::Relaxed);
    loop {
        segment.node.next.set(Ptr::new(current));
        if compare_exchange_weak_acq_rel(&ABANDONED_SEGMENTS, &mut current, segment.as_ptr()) {
            break;
        }
    }
}

unsafe fn take_abandoned(segment: Whole<Segment>) {
    let mut list = ABANDONED_SEGMENTS.load(Ordering::Relaxed);
    loop {
        if compare_exchange_weak_acq_rel(&ABANDONED_SEGMENTS, &mut current, null_mut()) {
            break;
        }
    }

}
*/
