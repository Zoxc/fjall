#![allow(unstable_name_collisions)]

use crate::page::{
    DelayedMode, Page, PageKind, PageQueue, LARGE_PAGE_SHIFT, LARGE_PAGE_SIZE, MEDIUM_PAGE_SIZE,
    SMALL_PAGE_SIZE,
};
use crate::{
    align_up, system, thread_id, HUGE_BLOCK_SIZE, MAX_ALIGN_SIZE, PAGE_HUGE_ALIGN, PTR_SIZE,
};
use crate::{
    heap::Heap,
    linked_list::{List, Node},
    page::SMALL_PAGE_SHIFT,
    SMALL_OBJ_SIZE_MAX,
};
use core::{
    alloc::{GlobalAlloc, Layout},
    ptr::null_mut,
};
use sptr::Strict;
use std::alloc::System;
use std::mem;
use std::sync::atomic::{AtomicUsize, Ordering};

pub const SEGMENT_SHIFT: usize = LARGE_PAGE_SHIFT; // 4MiB

// Derived constants
pub const SEGMENT_SIZE: usize = 1 << SEGMENT_SHIFT;
pub const SEGMENT_ALIGN: usize = SEGMENT_SIZE;
pub const SEGMENT_MASK: usize = SEGMENT_ALIGN - 1;

// Alignments over ALIGNMENT_MAX are allocated in dedicated huge page segments
pub const ALIGNMENT_MAX: usize = SEGMENT_SIZE >> 1;

pub const SMALL_PAGES_PER_SEGMENT: usize = SEGMENT_SIZE / SMALL_PAGE_SIZE;
pub const MEDIUM_PAGES_PER_SEGMENT: usize = SEGMENT_SIZE / MEDIUM_PAGE_SIZE;
pub const LARGE_PAGES_PER_SEGMENT: usize = SEGMENT_SIZE / LARGE_PAGE_SIZE;

// Segments thread local data

#[derive(Debug)]
pub struct SegmentThreadData {
    small_free: List<Segment>,  // queue of segments with free small pages
    medium_free: List<Segment>, // queue of segments with free medium pages
    pages_purge: PageQueue,     // queue of freed pages that are delay purged
    count: usize,               // current number of segments;
    peak_count: usize,          // peak number of segments
    current_size: usize,        // current size of all segments
    peak_size: usize,           // peak size of all segments
    reclaim_count: usize,       // number of reclaimed (abandoned) segments
}

impl SegmentThreadData {
    unsafe fn free_queue_of_kind(
        data: *mut SegmentThreadData,
        kind: PageKind,
    ) -> *mut List<Segment> {
        match kind {
            PageKind::Small => &mut (*data).small_free as *mut _,
            PageKind::Medium => &mut (*data).medium_free as *mut _,
            _ => null_mut(),
        }
    }
}

impl SegmentThreadData {
    pub const INITIAL: SegmentThreadData = SegmentThreadData {
        small_free: List::EMPTY,
        medium_free: List::EMPTY,
        pages_purge: PageQueue::new(1), // queue of freed pages that are delay purged
        count: 0,                       // current number of segments;
        peak_count: 0,                  // peak number of segments
        current_size: 0,                // current size of all segments
        peak_size: 0,                   // peak size of all segments
        reclaim_count: 0,               // number of reclaimed (abandoned) segments
    };
}

#[derive(Debug)]
pub struct Segment {
    // constant fields
    // mi_memid_t           memid;            // id for the os-level memory manager
    // bool                 allow_decommit;
    allow_purge: bool,
    segment_size: usize, // for huge pages this may be different from `SEGMENT_SIZE`

    // segment fields
    // struct mi_segment_s* next;             // must be the first segment field after abandoned_next -- see `segment.c:segment_init`
    // struct mi_segment_s* prev;
    was_reclaimed: bool, // true if it was reclaimed (used to limit on-free reclamation)
    abandoned: usize, // abandoned pages (i.e. the original owning thread stopped) (`abandoned <= used`)
    abandoned_visits: usize, // count how often this segment is visited in the abandoned list (to force reclaim if it is too long)
    used: usize,             // count of pages in use (`used <= capacity`)
    capacity: usize,         // count of available pages (`#free + used`)
    segment_info_size: usize, // space we are using from the first page for segment meta-data and possible guard pages.

    #[cfg(debug_assertions)]
    cookie: usize, // verify addresses in secure mode: `_mi_ptr_cookie(segment) == segment->cookie`

    node: Node<Segment>,

    pub thread_id: AtomicUsize, // unique id of the thread owning this segment

    // layout like this to optimize access in `mi_free`
    page_shift: usize, // `1 << page_shift` == the page sizes == `page->block_size * page->reserved` (unless the first page, then `-segment_info_size`).
    // _Atomic(mi_threadid_t) thread_id;      // unique id of the thread owning this segment
    pub page_kind: PageKind, // kind of pages: small, medium, large, or huge

                             /*

                             // constant fields
                             mi_memid_t           memid;            // id for the os-level memory manager
                             bool                 allow_decommit;
                             bool                 allow_purge;
                             size_t               segment_size;     // for huge pages this may be different from `SEGMENT_SIZE`

                             // segment fields
                             struct mi_segment_s* next;             // must be the first segment field after abandoned_next -- see `segment.c:segment_init`
                             struct mi_segment_s* prev;
                             bool                 was_reclaimed;    // true if it was reclaimed (used to limit on-free reclamation)

                             size_t               abandoned;        // abandoned pages (i.e. the original owning thread stopped) (`abandoned <= used`)
                             size_t               abandoned_visits; // count how often this segment is visited in the abandoned list (to force reclaim if it is too long)

                             size_t               used;             // count of pages in use (`used <= capacity`)
                             size_t               capacity;         // count of available pages (`#free + used`)
                             size_t               segment_info_size;// space we are using from the first page for segment meta-data and possible guard pages.
                             uintptr_t            cookie;           // verify addresses in secure mode: `_mi_ptr_cookie(segment) == segment->cookie`

                             // layout like this to optimize access in `mi_free`
                             size_t                 page_shift;     // `1 << page_shift` == the page sizes == `page->block_size * page->reserved` (unless the first page, then `-segment_info_size`).
                             _Atomic(mi_threadid_t) thread_id;      // unique id of the thread owning this segment
                             mi_page_kind_t       page_kind;        // kind of pages: small, medium, large, or huge
                             mi_page_t            pages[1];         // up to `SMALL_PAGES_PER_SEGMENT` pages
                                */
}

#[allow(overflowing_literals)]
fn cookie(ptr: *mut Segment) -> usize {
    ptr.addr() ^ 0xa0f7b251664141e9
}

impl Segment {
    #[inline]
    fn node(segment: *mut Segment) -> *mut Node<Segment> {
        unsafe { &mut (*segment).node }
    }

    #[inline]
    pub fn page(segment: *mut Segment, index: usize) -> *mut Page {
        unsafe { segment.add(1).cast::<Page>().add(index) }
    }

    #[inline]
    unsafe fn page_index_from_pointer(segment: *mut Segment, ptr: *mut u8) -> usize {
        let diff = ptr.byte_offset_from(segment);
        debug_assert!(
            diff >= 0 && (diff as usize) <= SEGMENT_SIZE /* for huge alignment it can be equal */
        );
        let idx = (diff as usize) >> (*segment).page_shift;
        debug_assert!(idx < (*segment).capacity);
        debug_assert!((*segment).page_kind <= PageKind::Medium || idx == 0);
        idx
    }

    #[inline]
    pub unsafe fn page_from_pointer(segment: *mut Segment, ptr: *mut u8) -> *mut Page {
        Segment::page(segment, Segment::page_index_from_pointer(segment, ptr))
    }

    #[inline]
    unsafe fn has_free(segment: *mut Segment) -> bool {
        (*segment).used < (*segment).capacity
    }

    #[inline]
    pub unsafe fn is_local(segment: *mut Segment) -> bool {
        thread_id() == (*segment).thread_id.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn from_pointer_checked(ptr: *mut u8) -> *mut Segment {
        debug_assert!(!ptr.is_null());
        debug_assert!(ptr.addr() & !(PTR_SIZE - 1) != 0);

        let segment = Segment::from_pointer(ptr);
        debug_assert!(!segment.is_null());

        #[cfg(debug_assertions)]
        unsafe {
            debug_assert_eq!((*segment).cookie, cookie(segment));
        }

        segment
    }

    // Segment that contains the pointer
    // Large aligned blocks may be aligned at N*SEGMENT_SIZE (inside a huge segment > SEGMENT_SIZE),
    // and we need align "down" to the segment info which is `SEGMENT_SIZE` bytes before it;
    // therefore we align one byte before `p`.
    #[inline]
    pub fn from_pointer(ptr: *mut u8) -> *mut Segment {
        ptr.map_addr(|addr| (addr - 1) & !SEGMENT_MASK).cast()
    }

    #[inline]
    unsafe fn raw_page_size(segment: *mut Segment) -> usize {
        if (*segment).page_kind == PageKind::Huge {
            (*segment).segment_size
        } else {
            1 << (*segment).page_shift
        }
    }

    // Raw start of the page available memory; can be used on uninitialized pages (only `segment_idx` must be set)
    // The raw start is not taking aligned block allocation into consideration.
    unsafe fn raw_page_start(segment: *mut Segment, page: *mut Page) -> (*mut u8, usize) {
        let mut psize = Segment::raw_page_size(segment);
        let mut p = segment
            .cast::<u8>()
            .add((*page).segment_idx as usize * psize);

        if (*page).segment_idx == 0 {
            // the first page starts after the segment info (and possible guard page)
            p = p.add((*segment).segment_info_size);
            psize -= (*segment).segment_info_size;
        }

        (p, psize)
    }

    // Start of the page available memory; can be used on uninitialized pages (only `segment_idx` must be set)
    pub unsafe fn page_start(
        segment: *mut Segment,
        page: *mut Page,
        block_size: usize,
        pre_size: &mut usize,
    ) -> (*mut u8, usize) {
        let (mut p, mut psize) = Segment::raw_page_start(segment, page);
        *pre_size = 0;
        if (*page).segment_idx == 0 && block_size > 0 && (*segment).page_kind <= PageKind::Medium {
            // for small and medium objects, ensure the page start is aligned with the block size (PR#66 by kickunderscore)
            let adjust = block_size - ((p as usize) % block_size);
            if psize - adjust >= block_size && adjust < block_size {
                p = p.add(adjust);
                psize -= adjust;
                *pre_size = adjust;
            }
        }

        (p, psize)
    }

    /* -----------------------------------------------------------
       Page allocation
    ----------------------------------------------------------- */

    unsafe fn page_ensure_committed(
        _segment: *mut Segment,
        page: *mut Page,
        _data: *mut SegmentThreadData,
    ) -> bool {
        if (*page).is_committed {
            return true;
        }
        (*page).is_committed = true;
        true

        /*
        if (*page).is_committed {
            return true;
        }
        //  debug_assert!((*segment).allow_decommit);
        // mi_assert_expensive(!mi_pages_purge_contains(page, tld));

        let (start, psize) = Segment::raw_page_start(segment, page);

        /*
                let is_zero = false;
                const size_t gsize = (SECURE >= 2 ? _mi_os_page_size() : 0);
                let ok = _mi_os_commit(start, psize + gsize, &is_zero, tld->stats);
                if (!ok) {return false; }// failed to commit!
        */
        (*page).is_committed = true;
        (*page).used = 0;
        (*page).is_zero_init = false;

        return true;*/
    }

    #[inline]
    unsafe fn page_claim(
        segment: *mut Segment,
        page: *mut Page,
        data: *mut SegmentThreadData,
    ) -> bool {
        debug_assert!(Page::segment(page) == segment);
        debug_assert!(!(*page).segment_in_use);

        //  mi_page_purge_remove(page, tld);

        // check commit
        if !Segment::page_ensure_committed(segment, page, data) {
            return false;
        }

        // set in-use before doing unreset to prevent delayed reset
        (*page).segment_in_use = true;
        (*segment).used += 1;
        debug_assert!(
            (*page).segment_in_use && (*page).is_committed && (*page).used == 0 /* && !mi_pages_purge_contains(page, tld) */
        );
        debug_assert!((*segment).used <= (*segment).capacity);
        if (*segment).used == (*segment).capacity && (*segment).page_kind <= PageKind::Medium {
            // if no more free pages, remove from the queue
            debug_assert!(!Segment::has_free(segment));
            Segment::remove_from_free_queue(segment, data);
        }
        true
    }

    #[inline]
    unsafe fn find_free(segment: *mut Segment, data: *mut SegmentThreadData) -> *mut Page {
        //   mi_assert_internal(mi_segment_has_free(segment));
        //  mi_assert_expensive(mi_segment_is_valid(segment, tld));
        for i in 0..(*segment).capacity {
            // TODO: use a bitmap instead of search?
            let page = Segment::page(segment, i);
            if !(*page).segment_in_use && Segment::page_claim(segment, page, data) {
                return page;
            }
        }
        null_mut()
    }

    unsafe fn try_reclaim(
        _heap: *mut Heap,
        _block_size: usize,
        _page_kind: PageKind,
        reclaimed: &mut bool,
        _data: *mut SegmentThreadData,
    ) -> *mut Segment {
        *reclaimed = false;
        null_mut()
        /*
        mi_segment_t* segment;
        mi_arena_field_cursor_t current; _mi_arena_field_cursor_init(heap,&current);
        long max_tries = mi_segment_get_reclaim_tries();
        while ((max_tries-- > 0) && ((segment = _mi_arena_segment_clear_abandoned_next(&current)) != NULL))
        {
          segment->abandoned_visits++;
          // todo: an arena exclusive heap will potentially visit many abandoned unsuitable segments
          // and push them into the visited list and use many tries. Perhaps we can skip non-suitable ones in a better way?
          bool is_suitable = _mi_heap_memid_is_suitable(heap, segment->memid);
          bool all_pages_free;
          bool has_page = mi_segment_check_free(segment,block_size,&all_pages_free); // try to free up pages (due to concurrent frees)
          if (all_pages_free) {
            // free the segment (by forced reclaim) to make it available to other threads.
            // note1: we prefer to free a segment as that might lead to reclaiming another
            // segment that is still partially used.
            // note2: we could in principle optimize this by skipping reclaim and directly
            // freeing but that would violate some invariants temporarily)
            mi_segment_reclaim(segment, heap, 0, NULL, tld);
          }
          else if (has_page && segment->page_kind == page_kind && is_suitable) {
            // found a free page of the right kind, or page of the right block_size with free space
            // we return the result of reclaim (which is usually `segment`) as it might free
            // the segment due to concurrent frees (in which case `NULL` is returned).
            return mi_segment_reclaim(segment, heap, block_size, reclaimed, tld);
          }
          else if (segment->abandoned_visits >= 3 && is_suitable) {
            // always reclaim on 3rd visit to limit the list length.
            mi_segment_reclaim(segment, heap, 0, NULL, tld);
          }
          else {
            // otherwise, mark it back as abandoned
            // todo: reset delayed pages in the segment?
            _mi_arena_segment_mark_abandoned(segment);
          }
        }
        return NULL;
        */
    }

    fn calculate_sizes(
        capacity: usize,
        required: usize,
        pre_size: &mut usize,
        info_size: &mut usize,
    ) -> usize {
        let minsize   = mem::size_of::<Segment>() + ((capacity - 1) * mem::size_of::<Page>()) + 16 /* padding */;
        let guardsize = 0;
        let isize = align_up(minsize, 16 * MAX_ALIGN_SIZE);

        *info_size = isize;
        *pre_size = isize + guardsize;
        if required == 0 {
            SEGMENT_SIZE
        } else {
            align_up(required + isize + 2 * guardsize, PAGE_HUGE_ALIGN)
        }
    }

    unsafe fn os_alloc(
        eager_delayed: bool,
        page_alignment: usize,
        pre_size: usize,
        _info_size: usize,
        _commit: bool,
        mut segment_size: usize,

        _data: *mut SegmentThreadData,
    ) -> *mut Segment {
        //mi_memid_t memid;
        let _allow_large = !eager_delayed; // only allow large OS pages once we are no longer lazy
        let align_offset;
        let mut alignment = SEGMENT_SIZE;
        if page_alignment > 0 {
            alignment = page_alignment;
            align_offset = align_up(pre_size, SEGMENT_SIZE);
            segment_size += (align_offset - pre_size); // adjust the segment size
        }

        let segment: *mut Segment =
            system::alloc(Layout::from_size_align(segment_size, alignment).unwrap()).cast();
        /*
        mi_segment_t* segment = _mi_arena_alloc_aligned(segment_size, alignment, align_offset, commit, allow_large, req_arena_id, &memid, tld_os);
        */
        if segment.is_null() {
            return null_mut(); // failed to allocate
        }
        debug_assert!(segment.is_aligned_to(alignment));

        /*
        if (!memid.initially_committed) {
        // ensure the initial info is committed
        mi_assert_internal(!memid.is_pinned);
        let ok = _mi_os_commit(segment, pre_size, null_mut(), tld_os->stats);
        if (!ok) {
        // commit failed; we cannot touch the memory: free the segment directly and return `null_mut()`
        _mi_arena_free(segment, segment_size, 0, memid, tld_os->stats);
        return NULL;
        }
        }
         */

        /*
                   (*segment).memid = memid;
                  (*segment).allow_decommit = !memid.is_pinned;
                (*segment).allow_purge =
                    (*segment).allow_decommit && (mi_option_get(mi_option_purge_delay) >= 0);
        */

        (*segment).segment_size = segment_size;

        #[cfg(debug_assertions)]
        {
            (*segment).cookie = cookie(segment);
        }

        //   mi_segments_track_size((long)(segment_size), tld);
        //   _mi_segment_map_allocated_at(segment);
        segment
    }

    // Allocate a segment from the OS aligned to `SEGMENT_SIZE` .
    unsafe fn alloc(
        required: usize,
        page_kind: PageKind,
        page_shift: usize,
        page_alignment: usize,
        data: *mut SegmentThreadData,
    ) -> *mut Segment {
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
        let mut info_size = 0;
        let mut pre_size = 0;
        let init_segment_size =
            Segment::calculate_sizes(capacity, required, &mut pre_size, &mut info_size);
        debug_assert!(init_segment_size >= required);

        // Initialize parameters
        /*
        let eager_delayed = (page_kind <= PAGE_MEDIUM &&          // don't delay for large objects
        // !_mi_os_has_overcommit() &&          // never delay on overcommit systems
        _mi_current_thread_count() > 1 &&       // do not delay for the first N threads
        tld->peak_count < (size_t)mi_option_get(mi_option_eager_commit_delay));
        let eager  = !eager_delayed && mi_option_is_enabled(mi_option_eager_commit);
        let init_commit = eager; // || (page_kind >= PAGE_LARGE);
        */
        let init_commit = true;
        let eager_delayed = false;

        // Allocate the segment from the OS (segment_size can change due to alignment)
        let segment = Segment::os_alloc(
            eager_delayed,
            page_alignment,
            pre_size,
            info_size,
            init_commit,
            init_segment_size,
            data,
        );
        if segment.is_null() {
            return null_mut();
        }
        debug_assert!(!segment.is_null() && (segment as usize) % SEGMENT_SIZE == 0);
        //debug_assert!((*segment).memid.is_pinned ? (*segment).memid.initially_committed : true);

        // zero the segment info (but not the `mem` fields)
        /*
        let ofs = offsetof(mi_segment_t, next);
        _mi_memzero((uint8_t*)segment + ofs, info_size - ofs);
        */

        // initialize pages info
        for i in 0..capacity {
            debug_assert!(i <= 255);
            let page = Segment::page(segment, i);
            (*page).segment_idx = i as u8;
            //  page.is_committed = (*segment).memid.initially_committed;
            // page.is_zero_init = (*segment).memid.initially_zero;
        }

        // initialize
        (*segment).page_kind = page_kind;
        (*segment).capacity = capacity;
        (*segment).page_shift = page_shift;
        (*segment).segment_info_size = pre_size;

        // FIXME:
        (*segment).thread_id.store(thread_id(), Ordering::Relaxed); // thread_id();

        // insert in free lists for small and medium pages
        if page_kind <= PageKind::Medium {
            Segment::insert_in_free_queue(segment, data)
        }

        segment
    }

    unsafe fn reclaim_or_alloc(
        heap: *mut Heap,
        block_size: usize,
        page_kind: PageKind,
        page_shift: usize,
        data: *mut SegmentThreadData,
    ) -> *mut Segment {
        debug_assert!(page_kind <= PageKind::Large);
        debug_assert!(block_size < HUGE_BLOCK_SIZE as usize);

        // 1. try to reclaim an abandoned segment
        let mut reclaimed = false;
        let segment = Segment::try_reclaim(heap, block_size, page_kind, &mut reclaimed, data);
        // debug_assert!(segment.is_null() || _mi_arena_memid_is_suitable(segment->memid, heap->arena_id));
        if reclaimed {
            // reclaimed the right page right into the heap
            debug_assert!(
                !segment.is_null()
                    && (*segment).page_kind == page_kind
                    && page_kind <= PageKind::Large
            );
            return null_mut(); // pretend out-of-memory as the page will be in the page queue of the heap with available blocks
        } else if !segment.is_null() {
            // reclaimed a segment with empty pages (of `page_kind`) in it
            return segment;
        }
        // 2. otherwise allocate a fresh segment
        Segment::alloc(0, page_kind, page_shift, 0, data)
    }

    unsafe fn page_try_alloc_in_queue(
        _heap: *mut Heap,
        kind: PageKind,
        data: *mut SegmentThreadData,
    ) -> *mut Page {
        // find an available segment the segment free queue
        let free_queue = SegmentThreadData::free_queue_of_kind(data, kind);
        for segment in (*free_queue).iter(Segment::node) {
            /* if (_mi_arena_memid_is_suitable(segment->memid, heap->arena_id) && mi_segment_has_free(segment)) {
              return mi_segment_page_alloc_in(segment, tld);
            }  */
            if Segment::has_free(segment) {
                return Segment::find_free(segment, data);
            }
        }
        null_mut()
    }

    unsafe fn page_alloc_kind(
        heap: *mut Heap,
        block_size: usize,
        kind: PageKind,
        page_shift: usize,
        data: *mut SegmentThreadData,
    ) -> *mut Page {
        let mut page = Segment::page_try_alloc_in_queue(heap, kind, data);
        if page.is_null() {
            // possibly allocate or reclaim a fresh segment
            let segment = Segment::reclaim_or_alloc(heap, block_size, kind, page_shift, data);
            if segment.is_null() {
                return null_mut(); // return null_mut() if out-of-memory (or reclaimed)
            }
            debug_assert!((*segment).page_kind == kind);
            debug_assert!((*segment).used < (*segment).capacity);
            //   debug_assert!(_mi_arena_memid_is_suitable((*segment).memid, heap->arena_id));
            page = Segment::page_try_alloc_in_queue(heap, kind, data);
            // this should now succeed
        }
        debug_assert!(!page.is_null());

        page
    }

    pub unsafe fn page_alloc(
        heap: *mut Heap,
        block_size: usize,
        page_alignment: usize,
        data: *mut SegmentThreadData,
    ) -> *mut Page {
        let page;

        debug_assert!(page_alignment <= ALIGNMENT_MAX);
        /*
        if unlikely(page_alignment > ALIGNMENT_MAX) {
          debug_assert!(page_alignment.is_power_of_two());
          debug_assert!(page_alignment >= SEGMENT_SIZE);
          //debug_assert!((SEGMENT_SIZE % page_alignment) == 0);
          if (page_alignment < SEGMENT_SIZE) { page_alignment = SEGMENT_SIZE; }
          page = mi_segment_huge_page_alloc(block_size, page_alignment, heap->arena_id, tld, os_tld);
        } else { */

        if block_size <= SMALL_OBJ_SIZE_MAX {
            page =
                Segment::page_alloc_kind(heap, block_size, PageKind::Small, SMALL_PAGE_SHIFT, data);
        } else {
            panic!()
        }
        /*
                 else if block_size <= MEDIUM_OBJ_SIZE_MAX {
                    page = Segment::page_alloc_kind(
                        heap,
                        block_size,
                        PageKind::Medium,
                        MEDIUM_PAGE_SHIFT,
                        data,
                    );
                } else if block_size <= LARGE_OBJ_SIZE_MAX
                /* || mi_is_good_fit(block_size, LARGE_PAGE_SIZE - sizeof(mi_segment_t)) */
                {
                    page = mi_segment_large_page_alloc(heap, block_size, tld, os_tld);
                } else {
                    page = mi_segment_huge_page_alloc(
                        block_size,
                        page_alignment,
                        (*heap).arena_id,
                        tld,
                        os_tld,
                    );
                }
        */
        // mi_assert_expensive(page.is_null() || mi_segment_is_valid(_mi_page_segment(page),tld));
        // debug_assert!(page.is_null() || (mi_segment_page_size(Page::segment(page))) >= block_size);
        // mi_segment_try_purge(tld);
        // debug_assert!(page.is_null() || mi_page_not_in_queue(page, tld));
        page
    }

    unsafe fn pages_try_purge(_data: *mut SegmentThreadData) {
        /*

        mi_msecs_t now = _mi_clock_now();
        mi_page_queue_t* pq = &tld->pages_purge;
        // from oldest up to the first that has not expired yet
        mi_page_t* page = pq->last;
        while (page != null_mut() && mi_page_purge_is_expired(page,now)) {
          mi_page_t* const prev = (*page).prev; // save previous field
          mi_page_purge_remove(page, tld);    // remove from the list to maintain invariant for mi_page_purge
          mi_page_purge(_mi_page_segment(page), page, tld);
          page = prev;
        }
        // discard the reset pages from the queue
        pq->last = page;
        if (page != null_mut()){
          page->next = null_mut();
        }
        else {
          pq->first = null_mut();
        }

        */
    }

    pub unsafe fn page_abandon(page: *mut Page, data: *mut SegmentThreadData) {
        debug_assert!(!page.is_null());
        debug_assert!(Page::thread_free_flag(page) == DelayedMode::NeverDelayedFree);
        debug_assert!(Page::heap(page).is_null());

        let segment = Page::segment(page);
        //   mi_assert_expensive(!mi_pages_purge_contains(page, tld));
        //   mi_assert_expensive(mi_segment_is_valid(segment, tld));
        (*segment).abandoned += 1;
        //    _mi_stat_increase(&tld->stats->pages_abandoned, 1);
        debug_assert!((*segment).abandoned <= (*segment).used);
        if ((*segment).used == (*segment).abandoned) {
            // all pages are abandoned, abandon the entire segment
            Segment::abandon(segment, data);
        }
    }

    /* -----------------------------------------------------------
      Page reset
    ----------------------------------------------------------- */
    unsafe fn page_purge(segment: *mut Segment, page: *mut Page) {
        if !(*segment).allow_purge {
            return;
        }
        // FIXME
        //   let (start, psize) = Segment::raw_page_start(segment, page) mi_segment_raw_page_start(segment, page,);
        //   let needs_recommit = _mi_os_purge(start, psize,);
        let needs_recommit = false;
        if needs_recommit {
            (*page).is_committed = false;
        }
        (*page).used = 0;
    }

    unsafe fn schedule_purge(
        segment: *mut Segment,
        page: *mut Page,
        _data: *mut SegmentThreadData,
    ) {
        /* if (!(*segment).allow_purge) {return;}

        if (mi_option_get(mi_option_purge_delay) == 0) { */
        // purge immediately?
        Segment::page_purge(segment, page);
        /*
        }
        else if (mi_option_get(mi_option_purge_delay) > 0) {   // no purging if the delay is negative
        // otherwise push on the delayed page reset queue
        let queue = &mut *data.pages_purge as *mut _;
        // push on top
        mi_page_purge_set_expire(page);
        PageQueue::push_front(queue, page);
        }*/
    }

    // clear page data; can be called on abandoned segments

    unsafe fn page_clear(segment: *mut Segment, page: *mut Page, data: *mut SegmentThreadData) {
        (*page).is_zero_init = false;
        (*page).segment_in_use = false;

        // zero the page data, but not the segment fields and capacity, and block_size (for page size calculations)
        let block_size = (*page).xblock_size;
        let capacity = (*page).capacity;
        let reserved = (*page).reserved;
        let segment_idx = (*page).segment_idx;
        let segment_in_use = (*page).segment_in_use;
        let is_committed = (*page).is_committed;
        let is_zero_init = (*page).is_zero_init;

        // FIXME: Data race with deallocation in miri?
        *page = Page::EMPTY;

        (*page).capacity = capacity;
        (*page).reserved = reserved;
        (*page).xblock_size = block_size;
        (*page).segment_idx = segment_idx;
        (*page).segment_in_use = segment_in_use;
        (*page).is_committed = is_committed;
        (*page).is_zero_init = is_zero_init;

        (*segment).used -= 1;

        // schedule purge
        Segment::schedule_purge(segment, page, data);

        (*page).capacity = 0; // after purge these can be zero'd now
        (*page).reserved = 0;
    }

    /*

    unsafe fn map_index_of( segment: *mut Segment) -> (usize,usize) {
        // note: segment can be invalid or null_mut().
        debug_assert!(Segment::from_pointer(segment + 1) == segment); // is it aligned on SEGMENT_SIZE?
        if (segment as usize >= MAX_ADDRESS) {
          *bitidx = 0;
           (SEGMENT_MAP_WSIZE, 0)
        }
        else {
          const uintptr_t segindex = ((uintptr_t)segment) / SEGMENT_SIZE;
          let bitidx = segindex % PTR_BITS;
          const size_t mapindex = segindex / PTR_BITS;
          debug_assert!(mapindex < SEGMENT_MAP_WSIZE);
           (mapindex, bitidx)
        }
      }

    unsafe fn map_freed_at( segment: *mut Segment) {
        let (index, bitidx) = Segment::map_index_of(segment);
        debug_assert!(index <= SEGMENT_MAP_WSIZE);
        if (index == SEGMENT_MAP_WSIZE) return;
        uintptr_t mask = atomic_load_relaxed(&mi_segment_map[index]);
        uintptr_t newmask;
        do {
          newmask = (mask & ~((uintptr_t)1 << bitidx));
        } while (!mi_atomic_cas_weak_release(&mi_segment_map[index], &mask, newmask));
      }
      */

    unsafe fn os_free(segment: *mut Segment, segment_size: usize, data: *mut SegmentThreadData) {
        // FIXME: Why are we changing `Segment` in this function?

        (*segment).thread_id.store(0, Ordering::Relaxed);

        //  Segment::map_freed_at(segment);

        //  mi_segments_track_size(-((long)segment_size),tld);
        if (*segment).was_reclaimed {
            (*data).reclaim_count -= 1;
            (*segment).was_reclaimed = false;
        }

        let mut fully_committed = true;
        let mut committed_size = 0;
        let page_size = Segment::raw_page_size(segment);
        for i in 0..(*segment).capacity {
            let page = Segment::page(segment, i);
            if (*page).is_committed {
                committed_size += page_size;
            }
            if !(*page).is_committed {
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
            segment.cast(),
            Layout::from_size_align((*segment).segment_size, SEGMENT_ALIGN).unwrap(),
        );
    }

    unsafe fn free(segment: *mut Segment, data: *mut SegmentThreadData) {
        // don't purge as we are freeing now
        //  mi_segment_remove_all_purges(segment, false /* don't force as we are about to free */, tld);
        Segment::remove_from_free_queue(segment, data);

        // return it to the OS
        Segment::os_free(segment, (*segment).segment_size, data);
    }

    // remove from free queue if it is in one
    unsafe fn remove_from_free_queue(segment: *mut Segment, data: *mut SegmentThreadData) {
        let queue = SegmentThreadData::free_queue_of_kind(data, (*segment).page_kind);
        if !queue.is_null()
            && (!(*segment).node.next.is_null()
                || !(*segment).node.prev.is_null()
                || (*queue).first == segment)
        {
            (*queue).remove(segment, Segment::node);
        }
    }

    // mark a specific segment as abandoned
    // clears the thread_id.
    unsafe fn mark_abandoned(segment: *mut Segment) {
        (*segment).thread_id.store(0, Ordering::Release);
        debug_assert!((*segment).used == (*segment).abandoned);

        /*
        if ((*segment).memid.memkind != MEM_ARENA) {
          // not in an arena; count it as abandoned and return
          mi_atomic_increment_relaxed(&abandoned_count);
          return;
        }
        size_t arena_idx;
        size_t bitmap_idx;
        mi_arena_memid_indices((*segment).memid, &arena_idx, &bitmap_idx);
        mi_assert_internal(arena_idx < MAX_ARENAS);
        mi_arena_t* arena = mi_atomic_load_ptr_acquire(mi_arena_t, &mi_arenas[arena_idx]);
        mi_assert_internal(arena != NULL);
        const bool was_unmarked = _mi_bitmap_claim(arena->blocks_abandoned, arena->field_count, 1, bitmap_idx, NULL);
        if (was_unmarked) { mi_atomic_increment_relaxed(&abandoned_count); }
        mi_assert_internal(was_unmarked);
        mi_assert_internal(_mi_bitmap_is_claimed(arena->blocks_inuse, arena->field_count, 1, bitmap_idx));
        */
    }

    /* -----------------------------------------------------------
       Abandon segment/page
    ----------------------------------------------------------- */

    unsafe fn abandon(segment: *mut Segment, data: *mut SegmentThreadData) {
        debug_assert!((*segment).used == (*segment).abandoned);
        debug_assert!((*segment).used > 0);

        // Potentially force purge. Only abandoned segments in arena memory can be
        // reclaimed without a free so if a segment is not from an arena we force purge here to be conservative.
        Segment::pages_try_purge(data);
        //  let force_purge = true;//  mi_option_is_enabled(mi_option_abandoned_page_purge);
        // mi_segment_remove_all_purges(segment, force_purge, data);

        // remove the segment from the free page queue if needed
        Segment::remove_from_free_queue(segment, data);

        // all pages in the segment are abandoned; add it to the abandoned list

        //  mi_segments_track_size(-((long)(*segment).segment_size), data);

        (*segment).abandoned_visits = 0;
        if (*segment).was_reclaimed {
            (*data).reclaim_count -= 1;
            (*segment).was_reclaimed = false;
        }
        Segment::mark_abandoned(segment);
    }

    unsafe fn insert_in_free_queue(segment: *mut Segment, data: *mut SegmentThreadData) {
        let queue = SegmentThreadData::free_queue_of_kind(data, (*segment).page_kind);
        debug_assert!(!queue.is_null()); // FIXME, is this true?
        (*queue).push_back(segment, Segment::node);
    }

    pub unsafe fn page_free(page: *mut Page, _force: bool, data: *mut SegmentThreadData) {
        let segment = Page::segment(page);
        Segment::pages_try_purge(data);

        // mark it as free now
        Segment::page_clear(segment, page, data);

        if (*segment).used == 0 {
            // no more used pages; remove from the free list and free the segment
            Segment::free(segment, data);
        } else if (*segment).used == (*segment).abandoned {
            // only abandoned pages; remove from free list and abandon
            Segment::abandon(segment, data);
        } else if (*segment).used + 1 == (*segment).capacity {
            Segment::insert_in_free_queue(segment, data);
        }
    }
}
