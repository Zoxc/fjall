use crate::page::{Block, DelayedMode, Page, PageQueue};
use crate::segment::SegmentThreadData;
use crate::thread_exit::register_thread;
use crate::{
    bin_index, compare_exchange_weak_acq_rel, compare_exchange_weak_release, index_array,
    is_main_thread, word_count, yield_now, BINS, BIN_FULL, LARGE_OBJ_SIZE_MAX, LARGE_OBJ_WSIZE_MAX,
    PTR_SIZE, SMALL_ALLOC, SMALL_ALLOC_WORDS,
};
use core::{alloc::Layout, intrinsics::unlikely, ptr::null_mut};
use std::ptr::{addr_of, addr_of_mut};
use std::sync::atomic::{AtomicPtr, Ordering};

// Thread local data

#[derive(Debug)]
pub struct ThreadData {
    heartbeat: usize, // monotonic heartbeat count
    //  bool                recurse;       // true if deferred was called; used to prevent infinite recursion.
    //  mi_heap_t*          heap_backing;  // backing heap of this thread (cannot be deleted)
    //  mi_heap_t*          heaps;         // list of heaps in this thread (so we can abandon all when the thread terminates)
    pub segment: SegmentThreadData,
}

impl ThreadData {
    const INITIAL: ThreadData = ThreadData {
        heartbeat: 0,
        segment: SegmentThreadData::INITIAL,
    };
}

#[derive(Debug, PartialEq)]
enum HeapState {
    Uninit,
    Active,
    Abandoned,
}

#[derive(Debug)]
pub struct Heap {
    pub thread_data: ThreadData,

    state: HeapState,

    /// optimize: array where every entry points a page with possibly free blocks in the corresponding queue for that size.
    pub pages_free: [*mut Page; SMALL_ALLOC_WORDS + 1],

    /// queue of pages for each size class (or "bin")
    pub page_queues: [PageQueue; BIN_FULL + 1],

    pub thread_delayed_free: AtomicPtr<Block>,

    pub page_retired_min: usize, // smallest retired index (retired pages are fully free, but still in the page queues)
    pub page_retired_max: usize, // largest retired index into the `pages` array.
    pub page_count: usize,       // total number of pages in the `pages` queues.
    no_reclaim: bool,            // `true` if this heap should not reclaim abandoned pages

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
    PageQueue::new(size * PTR_SIZE)
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
    queue(65536),
    queue(81920),
    queue(98304),
    queue(114688),
    queue(131072), /* 64 */
    queue(163840),
    queue(196608),
    queue(229376),
    queue(262144),
    /*
        queue(327680),
        queue(393216),
         queue(458752),
         queue(524288), /* 72 */
    */
    queue(LARGE_OBJ_WSIZE_MAX + 1 /* 655360, Huge queue */),
    queue(LARGE_OBJ_WSIZE_MAX + 2), /* Full queue */
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
        state: HeapState::Uninit,
        pages_free: [Page::EMPTY_PTR; SMALL_ALLOC_WORDS + 1],
        page_queues: PAGE_QUEUES_INITIAL,
        page_retired_min: 0,
        page_retired_max: 0,
        page_count: 0,
        thread_data: ThreadData::INITIAL,
        thread_delayed_free: AtomicPtr::new(null_mut()),
        no_reclaim: false,
    };

    unsafe fn delayed_free_all(heap: *mut Heap) {
        while (!Heap::delayed_free_partial(heap)) {
            yield_now();
        }
    }

    // returns true if all delayed frees were processed
    unsafe fn delayed_free_partial(heap: *mut Heap) -> bool {
        // take over the list (note: no atomic exchange since it is often NULL)
        let mut block = (*heap).thread_delayed_free.load(Ordering::Relaxed);
        while !block.is_null()
            && !compare_exchange_weak_acq_rel(&(*heap).thread_delayed_free, &mut block, null_mut())
        { /* nothing */
        }
        let mut all_freed = true;

        // and free them all
        while (!block.is_null()) {
            let next = (*block).next;
            // use internal free instead of regular one to keep stats etc correct
            if (!Page::free_delayed_block(block)) {
                // we might already start delayed freeing while another thread has not yet
                // reset the delayed_freeing flag; in that case delay it further by reinserting the current block
                // into the delayed free list
                all_freed = false;
                let mut dfree = (*heap).thread_delayed_free.load(Ordering::Relaxed);
                loop {
                    (*block).next = dfree;
                    if compare_exchange_weak_release(
                        &(*heap).thread_delayed_free,
                        &mut dfree,
                        block,
                    ) {
                        break;
                    }
                }
            }
            block = next;
        }
        all_freed
    }

    // Visit all pages in a heap; returns `false` if break was called.
    unsafe fn visit_pages(
        heap: *mut Heap,
        mut visitor: impl FnMut(*mut Page, *mut PageQueue) -> bool,
    ) -> bool {
        debug_assert!(!heap.is_null());
        if (*heap).page_count == 0 {
            return false;
        }

        // visit all pages
        let total = (*heap).page_count;
        let mut count = 0;

        for i in 0..=BIN_FULL {
            let queue = Heap::page_queue(heap, i);
            for page in (*queue).list.iter(Page::node) {
                debug_assert!(Page::heap(page) == heap);
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
    pub unsafe fn done(heap: *mut Heap) {
        debug_assert!(!heap.is_null());
        debug_assert!((*heap).state == HeapState::Active);

        // collect if not the main / final thread (as in we're immediately going to exit)
        Heap::collect(heap, Collect::Abandon);

        (*heap) = Heap::INITIAL;
        (*heap).state = HeapState::Abandoned;
    }

    #[inline]
    pub unsafe fn page_queue(heap: *mut Heap, bin: usize) -> *mut PageQueue {
        index_array(addr_of_mut!((*heap).page_queues), bin)
    }

    #[inline]
    unsafe fn collect(heap: *mut Heap, collect: Collect) {
        debug_assert!(!heap.is_null());
        debug_assert!((*heap).state == HeapState::Active);

        // if (heap==NULL || !mi_heap_is_initialized(heap)) return;

        (*heap).thread_data.heartbeat += 1;

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
                assert!(Page::thread_free_flag(page) == DelayedMode::NeverDelayedFree);
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
            if (Page::all_free(page)) {
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
                || (*heap)
                    .thread_delayed_free
                    .load(Ordering::Acquire)
                    .is_null()
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
    pub unsafe fn collect_retired(heap: *mut Heap, force: bool) {
        let mut min = BIN_FULL;
        let mut max = 0;
        for bin in (*heap).page_retired_min..=(*heap).page_retired_max {
            let queue = Heap::page_queue(heap, bin);
            let page = (*queue).list.first;
            if !page.is_null() && (*page).retire_expire != 0 {
                if Page::all_free(page) {
                    (*page).retire_expire -= 1;
                    if force || (*page).retire_expire == 0 {
                        Page::free((*queue).list.first, queue, force);
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
                    (*page).retire_expire = 0;
                }
            }
        }
        (*heap).page_retired_min = min;
        (*heap).page_retired_max = max;
    }

    // The current small page array is for efficiency and for each
    // small size (up to 256) it points directly to the page for that
    // size without having to compute the bin. This means when the
    // current free page queue is updated for a small bin, we need to update a
    // range of entries in `_mi_page_small_free`.
    #[inline]
    pub unsafe fn queue_first_update(heap: *mut Heap, queue: *mut PageQueue) {
        let size = (*queue).block_size;
        if size > SMALL_ALLOC {
            return;
        }

        let mut page = (*queue).list.first;
        if (*queue).list.first.is_null() {
            page = Page::EMPTY_PTR;
        }

        // find index in the right direct page array
        let mut start;
        let idx = word_count(size);
        let pages_free = (*heap).pages_free.as_mut_ptr();

        if *pages_free.add(idx) == page {
            return; // already set
        }

        // find start slot
        if idx <= 1 {
            start = 0;
        } else {
            // find previous size; due to minimal alignment upto 3 previous bins may need to be skipped
            let bin = bin_index(size);
            let mut prev = queue.sub(1);
            let first = (*heap).page_queues.as_mut_ptr();
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
            *pages_free.add(sz) = page;
        }
    }

    #[inline]
    pub unsafe fn page_queue_for_size(heap: *mut Heap, size: usize) -> *mut PageQueue {
        let queue = Heap::page_queue(heap, bin_index(size));
        debug_assert!(size <= (*queue).block_size);
        queue
    }

    #[inline]
    unsafe fn find_free_page(heap: *mut Heap, layout: Layout) -> *mut Page {
        let queue = Heap::page_queue_for_size(heap, layout.size());
        let page = (*queue).list.first;
        if !page.is_null() {
            Page::free_collect(page, false);

            if Page::immediately_available(page) {
                return page;
            }
        }

        PageQueue::find_free(queue, heap, true)
    }

    #[inline]
    unsafe fn find_page(heap: *mut Heap, layout: Layout) -> *mut Page {
        // FIXME: Check large alignments
        if unlikely(layout.size() > isize::MAX as usize) {
            null_mut()
        } else {
            Heap::find_free_page(heap, layout)
        }
    }

    #[inline]
    pub unsafe fn alloc_small(heap: *mut Heap, layout: Layout) -> *mut u8 {
        let page = (*heap).pages_free[word_count(layout.size())];
        Page::alloc(page, layout, heap)
    }

    #[inline(never)]
    pub unsafe fn alloc_slow(heap: *mut Heap, layout: Layout) -> *mut u8 {
        match (*heap).state {
            HeapState::Abandoned => {
                // allocation from an abandoned heap
                return null_mut();
            }
            HeapState::Uninit => {
                // Register a thread exit callback.
                register_thread();

                (*heap).state = HeapState::Active;
            }
            HeapState::Active => {}
        }

        let mut page = Heap::find_page(heap, layout);

        if page.is_null() {
            Heap::collect(heap, Collect::Force);
            page = Heap::find_page(heap, layout);
        }

        if unlikely(page.is_null()) {
            // Out of memory
            return null_mut();
        }

        debug_assert!(Page::immediately_available(page));
        debug_assert!(Page::block_size(page) >= layout.size());

        Page::alloc(page, layout, heap)
    }
}

#[test]
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
