use crate::page::{AllocatedBlock, DelayedMode, FreeBlock, Page, PageFlags, PageKind, PageQueue};
use crate::segment::{Segment, SegmentThreadData, Whole, WholeOrStatic};
use crate::{
    align_up, bin_index, compare_exchange_weak_acq_rel, compare_exchange_weak_release, index_array,
    is_main_thread, system, word_count, yield_now, Ptr, BINS, BIN_FULL, BIN_FULL_BLOCK_SIZE,
    BIN_HUGE, BIN_HUGE_BLOCK_SIZE, LARGE_OBJ_SIZE_MAX, LARGE_OBJ_WSIZE_MAX, MEDIUM_ALIGN_MAX,
    MEDIUM_ALIGN_MAX_SIZE, SMALL_ALLOC, SMALL_ALLOC_WORDS, SMALL_SIZE_MAX, WORD_SIZE,
};
use core::{alloc::Layout, intrinsics::unlikely, ptr::null_mut};
use std::cell::{Cell, UnsafeCell};
use std::intrinsics::likely;
use std::ptr::{addr_of, addr_of_mut};
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

// Thread local data

#[derive(Debug)]
pub struct ThreadData {
    // heartbeat: usize, // monotonic heartbeat count
    //  bool                recurse;       // true if deferred was called; used to prevent infinite recursion.
    //  mi_heap_t*          heap_backing;  // backing heap of this thread (cannot be deleted)
    //  mi_heap_t*          heaps;         // list of heaps in this thread (so we can abandon all when the thread terminates)
    pub segment: SegmentThreadData,
}

impl ThreadData {
    const INITIAL: ThreadData = ThreadData {
        //  heartbeat: 0,
        segment: SegmentThreadData::INITIAL,
    };
}

#[derive(Debug, Copy, Clone, PartialEq)]
enum HeapState {
    Uninit,
    Active,
    Abandoned,
}

#[derive(Debug)]
#[repr(C)]
pub struct Heap {
    /// optimize: array where every entry points a page with possibly free blocks in the corresponding queue for that size.
    // FIXME: Use Cell in the array
    pub pages_free: UnsafeCell<[WholeOrStatic<Page>; SMALL_ALLOC_WORDS + 1]>,

    pub thread_data: UnsafeCell<ThreadData>,

    state: Cell<HeapState>,

    /// queue of pages for each size class (or "bin")
    pub page_queues: [PageQueue; BIN_FULL + 1],

    pub thread_delayed_free: AtomicPtr<FreeBlock>,

    pub page_retired_min: Cell<usize>, // smallest retired index (retired pages are fully free, but still in the page queues)
    pub page_retired_max: Cell<usize>, // largest retired index into the `pages` array.
    pub page_count: Cell<usize>,       // total number of pages in the `page_queues` queues.
    no_reclaim: Cell<bool>,            // `true` if this heap should not reclaim abandoned pages

                                       /*
                                         mi_tld_t*             tld;
                                         mi_page_t*            pages_free_direct[MI_PAGES_DIRECT];  // optimize: array where every entry points a page with possibly free blocks in the corresponding queue for that size.
                                         mi_page_queue_t       pages[MI_BIN_FULL + 1];              // queue of pages for each size class (or "bin")
                                         _Atomic(mi_block_t*)  thread_delayed_free;
                                         mi_threadid_t         thread_id;                           // thread this heap belongs too
                                         mi_arena_id_t         arena_id;                            // arena id if the heap belongs to a specific arena (or 0)
                                         uintptr_t             cookie;                              // random cookie to verify pointers (see `_mi_ptr_cookie`)
                                         uintptr_t             keys[2];                             // two random keys used to encode the `thread_delayed_free` list
                                         mi_random_ctx_t       random;                              // random number context used for secure allocation
                                         size_t                page_count;                          // total number of pages in the `pages` queues.
                                         size_t                page_retired_min;                    // smallest retired index (retired pages are fully free, but still in the page queues)
                                         size_t                page_retired_max;                    // largest retired index into the `pages` array.
                                         mi_heap_t*            next;                                // list of heaps per thread
                                         bool                  no_reclaim;                          // `true` if this heap should not reclaim abandoned pages
                                       */
}

/*
{
            let mut array = [PageQueue::new(1); BIN_FULL + 1];
            let mut i = 0;
            while i < BIN_FULL + 1 {
                array[i] = PageQueue::new(match i {
                    BIN_FULL => LARGE_OBJ_WSIZE_MAX + 2, // Full queue
                    BINS => LARGE_OBJ_WSIZE_MAX + 1,     // 655360

                    0 => 1,
                    // FIXME: Find the right function here
                    _ => {
                        let block_size = i * PTR_SIZE;
                        assert!(bin_index(block_size) == i);
                        block_size
                    }
                });
                i += 1;
            }
            array
        }
*/

const fn queue(size: usize) -> PageQueue {
    PageQueue::new(size * WORD_SIZE)
}

const PAGE_QUEUES_INITIAL: [PageQueue; BIN_FULL + 1] = [
    queue(1),
    queue(1),
    queue(2),
    queue(3),
    queue(4),
    queue(5),
    queue(6),
    queue(7),
    queue(8), /* 8 */
    queue(10),
    queue(12),
    queue(14),
    queue(16),
    queue(20),
    queue(24),
    queue(28),
    queue(32), /* 16 */
    queue(40),
    queue(48),
    queue(56),
    queue(64),
    queue(80),
    queue(96),
    queue(112),
    queue(128), /* 24 */
    queue(160),
    queue(192),
    queue(224),
    queue(256),
    queue(320),
    queue(384),
    queue(448),
    queue(512), /* 32 */
    queue(640),
    queue(768),
    queue(896),
    queue(1024),
    queue(1280),
    queue(1536),
    queue(1792),
    queue(2048), /* 40 */
    queue(2560),
    queue(3072),
    queue(3584),
    queue(4096),
    queue(5120),
    queue(6144),
    queue(7168),
    queue(8192), /* 48 */
    queue(10240),
    queue(12288),
    queue(14336),
    queue(16384),
    queue(20480),
    queue(24576),
    queue(28672),
    queue(32768), /* 56 */
    queue(40960),
    queue(49152),
    queue(57344),
    queue(65536), // 1 MB block size
    /*
    queue(81920),
    queue(98304),
    queue(114688),
    queue(131072), // 2 MB block size  64
    */
    /*
        // 16 byte word size (65 bins)
        queue(163840),
        queue(196608),
        queue(229376),
        queue(262144), // 8 byte word size (69 bins)
     */
    /*
        // Is this for smaller word sizes?
        queue(327680),
        queue(393216),
         queue(458752),
         queue(524288), /* 72 */
    */
    queue(BIN_HUGE_BLOCK_SIZE / WORD_SIZE), /* 655360, Huge queue */
    queue(BIN_FULL_BLOCK_SIZE / WORD_SIZE), /* Full queue */
];

#[derive(PartialEq, PartialOrd)]
enum Collect {
    Normal,
    Force,
    Abandon,
}

impl Heap {
    #[allow(clippy::declare_interior_mutable_const)]
    pub const INITIAL: Heap = Heap {
        state: Cell::new(HeapState::Uninit),
        pages_free: UnsafeCell::new(
            [unsafe { WholeOrStatic::new_unchecked(Page::EMPTY_PTR) }; SMALL_ALLOC_WORDS + 1],
        ),
        page_queues: PAGE_QUEUES_INITIAL,
        page_retired_min: Cell::new(0),
        page_retired_max: Cell::new(0),
        page_count: Cell::new(0),
        thread_data: UnsafeCell::new(ThreadData::INITIAL),
        thread_delayed_free: AtomicPtr::new(null_mut()),
        no_reclaim: Cell::new(false),
    };

    #[allow(clippy::mut_from_ref)]
    pub unsafe fn thread_data(&self) -> &mut ThreadData {
        &mut *self.thread_data.get()
    }

    unsafe fn delayed_free_all(heap: Ptr<Heap>) {
        while (!Heap::delayed_free_partial(heap)) {
            yield_now();
        }
    }

    // returns true if all delayed frees were processed
    unsafe fn delayed_free_partial(heap: Ptr<Heap>) -> bool {
        // take over the list (note: no atomic exchange since it is often NULL)
        let mut block = heap.thread_delayed_free.load(Ordering::Relaxed);
        while !block.is_null()
            && !compare_exchange_weak_acq_rel(&heap.thread_delayed_free, &mut block, null_mut())
        { /* nothing */
        }
        let mut all_freed = true;

        // and free them all
        while (!block.is_null()) {
            let next = (*block).next.get();
            // use internal free instead of regular one to keep stats etc correct
            if (!Page::free_delayed_block(Whole::new_unchecked(block))) {
                // we might already start delayed freeing while another thread has not yet
                // reset the delayed_freeing flag; in that case delay it further by reinserting the current block
                // into the delayed free list
                all_freed = false;
                let mut dfree = heap.thread_delayed_free.load(Ordering::Relaxed);
                loop {
                    (*block).next.set(Whole::new(dfree));
                    if compare_exchange_weak_release(&heap.thread_delayed_free, &mut dfree, block) {
                        break;
                    }
                }
            }
            block = Whole::as_maybe_null_ptr(next);
        }
        all_freed
    }

    // Visit all pages in a heap; returns `false` if break was called.
    unsafe fn visit_pages(
        heap: Ptr<Heap>,
        mut visitor: impl FnMut(Whole<Page>, Ptr<PageQueue>) -> bool,
    ) -> bool {
        if heap.page_count.get() == 0 {
            return false;
        }

        // visit all pages
        let total = heap.page_count.get();
        let mut count = 0;

        for i in 0..=BIN_FULL {
            let queue = Heap::page_queue(heap, i);
            for page in queue.list.iter(Page::node) {
                debug_assert!((*page).heap() == Some(heap));
                if !visitor(page, queue) {
                    return false;
                }
                count += 1;
            }
        }
        debug_assert!(count == total);
        true
    }

    #[inline]
    pub unsafe fn done(heap: Ptr<Heap>) {
        if heap.state.get() == HeapState::Uninit {
            return;
        }

        debug_assert!(heap.state.get() == HeapState::Active);

        // collect if not the main / final thread (as in we're immediately going to exit)
        Heap::collect(heap, Collect::Abandon);

        (*heap.as_ptr()) = Heap::INITIAL;
        heap.state.set(HeapState::Abandoned);
    }

    #[inline]
    pub unsafe fn page_queue(heap: Ptr<Heap>, bin: usize) -> Ptr<PageQueue> {
        Ptr::new_unchecked(index_array(&heap.page_queues, bin).cast_mut())
    }

    #[inline]
    unsafe fn collect(heap: Ptr<Heap>, collect: Collect) {
        debug_assert!(heap.state.get() == HeapState::Active);

        // if (heap==NULL || !mi_heap_is_initialized(heap)) return;

        // (*heap).thread_data.heartbeat += 1;

        /*
                // note: never reclaim on collect but leave it to threads that need storage to reclaim
                if (
                    collect == Collect::Force
                    && is_main_thread() && !(*heap).no_reclaim)
                {
                    // the main thread is abandoned (end-of-program), try to reclaim all abandoned segments.
                    // if all memory is freed by now, all segments should be freed.
                    _mi_abandoned_reclaim_all(heap, &(*heap).thread_data.segment);
                }
        */

        // if abandoning, mark all pages to no longer add to delayed_free
        if collect == Collect::Abandon {
            Heap::visit_pages(heap, |page, _| {
                Page::use_delayed_free(page, DelayedMode::NeverDelayedFree, false);
                true
            });
        }

        if cfg!(debug_assertions) {
            Heap::visit_pages(heap, |page, _| {
                assert!((*page).thread_free_flag() == DelayedMode::NeverDelayedFree);
                true
            });
        }

        // free all current thread delayed blocks.
        // (if abandoning, after this there are no more thread-delayed references into the pages.)
        Heap::delayed_free_all(heap);

        // collect retired pages
        Heap::collect_retired(heap, collect >= Collect::Force);

        // collect all pages owned by this thread
        Heap::visit_pages(heap, |page, queue| {
            //mi_assert_internal(mi_heap_page_is_valid(heap, pq, page, NULL, NULL));
            Page::free_collect(page, collect >= Collect::Force);
            if (page.all_free()) {
                // no more used blocks, free the page.
                // note: this will free retired pages as well.
                Page::free(page, queue, collect >= Collect::Force);
            } else if (collect == Collect::Abandon) {
                // still used blocks but the thread is done; abandon the page
                Page::abandon(page, queue);
            }
            true // don't break
        });

        debug_assert!(
            collect != Collect::Abandon
                || heap.thread_delayed_free.load(Ordering::Acquire).is_null()
        );

        // collect segment and thread caches
        if collect >= Collect::Force {
            //    _mi_segment_thread_collect(&(*heap).thread_data.segment);
        }

        /*
                // collect arenas on program-exit (or shared library unload)
                if collect >= Collect::Force && _mi_is_main_thread() && mi_heap_is_backing(heap) {
                    _mi_thread_data_collect();  // collect thread data cache
                    _mi_arena_collect(true /* force purge */, &(*heap).tld->stats);
                }
        */
    }

    // free retired pages: we don't need to look at the entire queues
    // since we only retire pages that are at the head position in a queue.
    pub unsafe fn collect_retired(heap: Ptr<Heap>, force: bool) {
        let mut min = BIN_FULL;
        let mut max = 0;
        for bin in heap.page_retired_min.get()..=heap.page_retired_max.get() {
            let queue = Heap::page_queue(heap, bin);
            let page = queue.list.first.get();
            if let Some(page) = page {
                if page.retire_expire.get() != 0 {
                    if page.all_free() {
                        page.retire_expire.set(page.retire_expire.get() - 1);
                        if force || page.retire_expire.get() == 0 {
                            Page::free(page, queue, force);
                        } else {
                            // keep retired, update min/max
                            if bin < min {
                                min = bin;
                            }
                            if bin > max {
                                max = bin;
                            }
                        }
                    } else {
                        page.retire_expire.set(0);
                    }
                }
            }
        }
        heap.page_retired_min.set(min);
        heap.page_retired_max.set(max);
    }

    // The current small page array is for efficiency and for each
    // small size (up to 256) it points directly to the page for that
    // size without having to compute the bin. This means when the
    // current free page queue is updated for a small bin, we need to update a
    // range of entries in `_mi_page_small_free`.
    #[inline]
    pub unsafe fn queue_first_update(heap: Ptr<Heap>, queue: Ptr<PageQueue>) {
        let size = queue.block_size;
        if size > SMALL_ALLOC {
            return;
        }

        let page = queue
            .list
            .first
            .get()
            .unwrap_or(Whole::new_unchecked(Page::EMPTY_PTR));

        // find index in the right direct page array
        let mut start;
        let idx = word_count(size);
        let pages_free = (*heap.pages_free.get()).as_mut_ptr();

        if *pages_free.add(idx) == page.or_static() {
            return; // already set
        }

        // find start slot
        if idx <= 1 {
            start = 0;
        } else {
            // find previous size; due to minimal alignment upto 3 previous bins may need to be skipped
            let bin = bin_index(size);
            let mut prev = queue.as_ptr().sub(1);
            let first = heap.page_queues.as_ptr().cast_mut();
            while bin == bin_index((*prev).block_size) && prev > first {
                prev = prev.sub(1);
            }
            start = 1 + word_count((*prev).block_size);
            if start > idx {
                start = idx;
            }
        }

        // set size range to the right page
        debug_assert!(start <= idx);
        for sz in start..=idx {
            *pages_free.add(sz) = page.or_static();
        }
    }

    #[inline]
    pub unsafe fn page_queue_for_size(heap: Ptr<Heap>, size: usize) -> Ptr<PageQueue> {
        let index = bin_index(size);
        let queue = Heap::page_queue(heap, index);
        if size > LARGE_OBJ_SIZE_MAX {
            debug_assert!(BIN_HUGE == index);
            debug_assert!(queue.block_size == BIN_HUGE_BLOCK_SIZE);
        } else {
            debug_assert!(queue.block_size <= LARGE_OBJ_SIZE_MAX);
        }
        debug_assert!(size <= queue.block_size);
        queue
    }

    #[inline]
    unsafe fn find_free_page(heap: Ptr<Heap>, layout: Layout) -> Option<Whole<Page>> {
        let queue = Heap::page_queue_for_size(heap, layout.size());
        let page = queue.list.first.get();
        if let Some(page) = page {
            Page::free_collect(page, false);

            if page.immediately_available() {
                return Some(page);
            }
        }

        PageQueue::find_free(queue, heap, true)
    }

    #[inline]
    unsafe fn find_page(heap: Ptr<Heap>, layout: Layout) -> Option<Whole<Page>> {
        if unlikely(layout.size() > LARGE_OBJ_SIZE_MAX || layout.align() > WORD_SIZE) {
            // FIXME: Align huge
            if unlikely(layout.size() > isize::MAX as usize) {
                None
            } else {
                Heap::alloc_huge(heap, layout)
            }
        } else {
            Heap::find_free_page(heap, layout)
        }
    }

    #[inline(never)]
    unsafe fn free_generic(
        segment: Whole<Segment>,
        page: Whole<Page>,
        is_local: bool,
        ptr: Whole<AllocatedBlock>,
    ) {
        let block = if (*page).flags().contains(PageFlags::HAS_ALIGNED) {
            Page::unalign_pointer(page, segment, ptr)
        } else {
            ptr.cast()
        };

        Page::free_block(page, is_local, block);
    }

    #[inline]
    pub unsafe fn free(ptr: Whole<AllocatedBlock>) {
        let segment = Segment::from_pointer_checked(ptr);
        let is_local = segment.is_local();
        let page = Segment::page_from_pointer(segment, ptr);

        // thread-local free?
        if likely(is_local) {
            if likely((*page).flags() == PageFlags::empty())
            // and it is not a full page (full pages need to move from the full bin), nor has aligned blocks (aligned blocks need to be unaligned)
            {
                let block: Whole<FreeBlock> = ptr.cast();

                // mi_check_padding(page, block);
                page.local_deferred_free_blocks.push_front(block);
                page.used.set(page.used.get() - 1);
                if unlikely(page.used.get() == 0) {
                    Page::retire(page);
                }
            } else {
                // page is full or contains (inner) aligned blocks; use generic path
                Heap::free_generic(segment, page, true, ptr);
            }
        } else {
            // not thread-local; use generic path
            Heap::free_generic(segment, page, false, ptr);
        }
    }

    // A huge page is allocated directly without being in a queue.
    // Because huge pages contain just one block, and the segment contains
    // just that page, we always treat them as abandoned and any thread
    // that frees the block can free the whole page and segment directly.
    // Huge pages are also used if the requested alignment is very large (> MI_ALIGNMENT_MAX).
    unsafe fn alloc_huge(heap: Ptr<Heap>, layout: Layout) -> Option<Whole<Page>> {
        let block_size = layout.size(); //_mi_os_good_alloc_size(size);
        debug_assert!(bin_index(block_size) == BIN_HUGE || layout.align() > WORD_SIZE);

        let pq = Heap::page_queue(heap, BIN_HUGE); // not block_size as that can be low if the page_alignment > 0
        debug_assert!((*pq).is_huge());

        let page = Page::fresh_alloc(heap, pq, block_size, layout.align());
        if let Some(page) = page {
            let bsize = Page::actual_block_size(page); // note: not `mi_page_usable_block_size` as `size` includes padding already
            debug_assert!(bsize >= layout.size());
            debug_assert!(page.immediately_available());
            debug_assert!(Page::segment(page).page_kind == PageKind::Huge);
            debug_assert!(Page::segment(page).used.get() == 1);
        }
        page
    }

    #[inline]
    pub unsafe fn alloc_small(heap: Ptr<Heap>, layout: Layout) -> Option<Whole<AllocatedBlock>> {
        debug_assert!(layout.align() <= WORD_SIZE);
        let page = (*heap.pages_free.get())[word_count(layout.size())];
        Page::alloc_small(page, layout, heap)
    }

    #[inline]
    pub unsafe fn alloc(heap: Ptr<Heap>, layout: Layout) -> Option<Whole<AllocatedBlock>> {
        if likely(layout.size() <= SMALL_SIZE_MAX && layout.align() <= WORD_SIZE) {
            Heap::alloc_small(heap, layout)
        } else {
            // FIXME: Add `mi_heap_malloc_zero_aligned_at_fallback` fallback for alignment
            Heap::alloc_slow(heap, layout)
        }
    }

    pub unsafe fn init(heap: Ptr<Heap>) {
        // Register a thread exit callback.
        system::register_thread();
    }

    pub unsafe fn adjust_alignment(layout: Layout) -> bool {
        layout.size() <= MEDIUM_ALIGN_MAX_SIZE
            && layout.align() <= MEDIUM_ALIGN_MAX
            && layout.align() > WORD_SIZE
    }

    #[inline(never)]
    pub unsafe fn alloc_slow(heap: Ptr<Heap>, layout: Layout) -> Option<Whole<AllocatedBlock>> {
        if unlikely(Self::adjust_alignment(layout)) {
            Heap::alloc_and_adjust_align(heap, layout)
        } else {
            Heap::alloc_generic(heap, layout)
        }
    }

    pub unsafe fn alloc_and_adjust_align(
        heap: Ptr<Heap>,
        layout: Layout,
    ) -> Option<Whole<AllocatedBlock>> {
        debug_assert!(Self::adjust_alignment(layout));

        // FIXME: Port some alignment fast paths from `mi_heap_malloc_zero_aligned_at_fallback`
        // and `mi_heap_malloc_zero_aligned_at`.

        // SAFETY: This will not overflow due to bounds on `layout.size() `and `layout.align()`.
        let expanded_layout =
            Layout::from_size_align_unchecked(layout.size() + (layout.align() - 1), 1);

        let result = Heap::alloc_generic(heap, expanded_layout)?;

        let aligned = result.map_addr(|addr| align_up(addr, layout.align()));

        if cfg!(debug_assertions) {
            let segment = Segment::from_pointer(result);
            debug_assert!(segment.page_kind <= PageKind::Large);
            let page = Segment::page_from_pointer(segment, result);
            let (page_start, page_size) =
                Segment::page_start(segment, page, page.block_size(), &mut 0);
        }

        if result != aligned {
            let segment = Segment::from_pointer(aligned);
            let page = Segment::page_from_pointer(segment, aligned);
            page.insert_flags(PageFlags::HAS_ALIGNED);
        }

        Some(aligned)
    }

    pub unsafe fn alloc_generic(heap: Ptr<Heap>, layout: Layout) -> Option<Whole<AllocatedBlock>> {
        debug_assert!(!Self::adjust_alignment(layout));

        if unlikely(heap.state.get() != HeapState::Active) {
            match heap.state.get() {
                HeapState::Abandoned => {
                    // allocation from an abandoned heap
                    return None;
                }
                HeapState::Uninit => {
                    Heap::init(heap);
                    heap.state.set(HeapState::Active);
                }
                HeapState::Active => {}
            }
        }

        // free delayed frees from other threads (but skip contended ones)
        Heap::delayed_free_partial(heap);

        let mut page = Heap::find_page(heap, layout);

        if page.is_none() {
            Heap::collect(heap, Collect::Force);
            page = Heap::find_page(heap, layout);
        }

        if unlikely(page.is_none()) {
            // Out of memory
            return None;
        }
        let page = page.unwrap_unchecked();

        debug_assert!(page.immediately_available());
        debug_assert!(Page::actual_block_size(page) >= layout.size());

        Page::alloc_free(page.or_static())
    }
}

#[test]
#[cfg(not(debug_assertions))]
fn bin_indices() {
    let queues = PAGE_QUEUES_INITIAL;

    for i in 0..queues.len() {
        if i < queues.len() - 2 {
            assert!(queues[i].block_size <= LARGE_OBJ_SIZE_MAX);
        }

        if i > 0 {
            assert!(queues[i].block_size >= queues[i - 1].block_size);
        }
    }

    for size in 0..=LARGE_OBJ_WSIZE_MAX {
        let index = bin_index(size);
        let block_size = queues[index].block_size;

        assert!(size <= queues[index].block_size);

        // We must be above the size of the lower bin
        assert!(index <= 1 || size > queues[index - 1].block_size);
    }
}

#[test]
fn bin_special_indices() {
    let queues = PAGE_QUEUES_INITIAL;

    let huge = queues.len() - 2;
    let full = queues.len() - 1;

    assert_eq!(huge, BIN_HUGE);
    assert_eq!(full, BIN_FULL);

    assert_eq!(queues[huge].block_size, BIN_HUGE_BLOCK_SIZE);
    assert_eq!(queues[full].block_size, BIN_FULL_BLOCK_SIZE);

    assert_eq!(bin_index(BIN_HUGE_BLOCK_SIZE), BIN_HUGE);
    assert_eq!(bin_index(LARGE_OBJ_SIZE_MAX + 1), BIN_HUGE);
    assert_eq!(bin_index(LARGE_OBJ_SIZE_MAX), BIN_HUGE - 1);
}
