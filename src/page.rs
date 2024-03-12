use crate::heap::Heap;
use crate::linked_list::{List, Node};
use crate::segment::Segment;
use crate::{
    bin_index, compare_exchange_weak_acq_rel, compare_exchange_weak_release, free, index_array,
    thread_id, word_count, yield_now, BINS, BIN_FULL, HUGE_BLOCK_SIZE, LARGE_OBJ_SIZE_MAX,
    MAX_EXTEND_SIZE, MIN_EXTEND, PTR_SIZE, SMALL_OBJ_SIZE_MAX,
};
use bitflags::bitflags;
use core::intrinsics::likely;
use core::{alloc::Layout, ptr::null_mut};
use sptr::Strict;
use std::cell::Cell;
use std::intrinsics::unlikely;
use std::mem;
use std::ptr::{addr_of, addr_of_mut};
use std::sync::atomic::{AtomicPtr, AtomicU8, Ordering};

// Main tuning parameters for segment and page sizes
// Sizes for 64-bit, divide by two for 32-bit
pub const SMALL_PAGE_SHIFT: usize = 16; // 64KiB
pub const MEDIUM_PAGE_SHIFT: usize = 3 + SMALL_PAGE_SHIFT; // 512KiB
pub const LARGE_PAGE_SHIFT: usize = 3 + MEDIUM_PAGE_SHIFT; // 4MiB

pub const SMALL_PAGE_SIZE: usize = 1 << SMALL_PAGE_SHIFT;
pub const MEDIUM_PAGE_SIZE: usize = 1 << MEDIUM_PAGE_SHIFT;
pub const LARGE_PAGE_SIZE: usize = 1 << LARGE_PAGE_SHIFT;

const MAX_RETIRE_SIZE: usize = LARGE_OBJ_SIZE_MAX;
const RETIRE_CYCLES: u8 = 16;

#[derive(Clone, Copy, PartialEq, PartialOrd, Debug)]
pub enum PageKind {
    Small,  // small blocks go into 64KiB pages inside a segment
    Medium, // medium blocks go into 512KiB pages inside a segment
    Large,  // larger blocks go into a single page spanning a whole segment
    Huge, // huge blocks (>512KiB) are put into a single page in a segment of the exact size (but still 2MiB aligned)
}

pub struct Block {
    pub next: *mut Block,
}

#[derive(Copy, Clone, Debug)]
pub struct PageQueue {
    pub list: List<Page>,
    pub block_size: usize,
}

impl PageQueue {
    #[inline]
    pub const fn new(block_size: usize) -> Self {
        debug_assert!(block_size > 0);
        PageQueue {
            list: List::EMPTY,
            block_size,
        }
    }

    #[inline]
    unsafe fn is_huge(&self) -> bool {
        self.block_size == LARGE_OBJ_SIZE_MAX + PTR_SIZE
    }

    #[inline]
    unsafe fn is_full(&self) -> bool {
        self.block_size == LARGE_OBJ_SIZE_MAX + (2 * PTR_SIZE)
    }

    #[inline]
    unsafe fn is_special(&self) -> bool {
        self.block_size > LARGE_OBJ_SIZE_MAX
    }

    #[inline]
    unsafe fn node(page: *mut Page) -> *mut Node<Page> {
        unsafe { &mut (*page).node }
    }

    unsafe fn add(queue: *mut PageQueue, page: *mut Page, heap: *mut Heap) {
        debug_assert!(Page::heap(page) == heap);
        //   debug_assert!(!mi_page_queue_contains(queue, page));
        debug_assert!(
            ((*page).xblock_size as usize) == (*queue).block_size
                || ((*page).xblock_size as usize > LARGE_OBJ_SIZE_MAX && (*queue).is_huge())
                || (Page::flags(page).contains(PageFlags::IN_FULL) && (*queue).is_full())
        );

        Page::set_flags(page, PageFlags::IN_FULL, (*queue).is_full());

        // mi_atomic_store_ptr_release(mi_atomic_cast(void*, &page->heap), heap);

        (*queue).list.push_front(page, Page::node);

        // update direct
        Heap::queue_first_update(heap, queue);

        (*heap).page_count += 1;
    }

    unsafe fn remove(queue: *mut PageQueue, page: *mut Page) {
        let heap = Page::heap(page);
        let update = page == (*queue).list.first;

        (*queue).list.remove(page, Page::node);

        if update {
            Heap::queue_first_update(heap, queue);
        }

        (*heap).page_count -= 1;

        Page::remove_flags(page, PageFlags::IN_FULL);
    }

    pub unsafe fn push_front(queue: *mut PageQueue, page: *mut Page) {
        (*queue).list.push_front(page, Page::node);
    }

    unsafe fn enqueue_from(to: *mut PageQueue, from: *mut PageQueue, page: *mut Page) {
        debug_assert!(!page.is_null());
        //mi_assert_expensive(mi_page_queue_contains(from, page));
        // mi_assert_expensive(!mi_page_queue_contains(to, page));

        #[allow(clippy::nonminimal_bool)]
        {
            debug_assert!(
                ((*page).xblock_size as usize == (*to).block_size
                    && (*page).xblock_size as usize == (*from).block_size)
                    || ((*page).xblock_size as usize == (*to).block_size && (*from).is_full())
                    || ((*page).xblock_size as usize == (*from).block_size && (*to).is_full())
                    || ((*page).xblock_size as usize > LARGE_OBJ_SIZE_MAX && (*to).is_huge())
                    || ((*page).xblock_size as usize > LARGE_OBJ_SIZE_MAX && (*to).is_full())
            );
        }

        let heap = Page::heap(page);

        let update = page == (*from).list.first;

        (*from).list.remove(page, Page::node);

        if update {
            //  mi_assert_internal(mi_heap_contains_queue(heap, from));
            Heap::queue_first_update(heap, from);
        }

        debug_assert!((*to).list.last.is_null() || heap == Page::heap((*to).list.last));

        let update = (*to).list.last.is_null();

        (*to).list.push_back(page, Page::node);

        if update {
            Heap::queue_first_update(heap, to);
        }

        Page::set_flags(page, PageFlags::IN_FULL, (*to).is_full());
    }

    unsafe fn from_page_and_heap(page: *mut Page, heap: *mut Heap) -> *mut PageQueue {
        let bin = if Page::flags(page).contains(PageFlags::IN_FULL) {
            BIN_FULL
        } else {
            bin_index((*page).xblock_size as usize)
        };
        debug_assert!(!heap.is_null() && bin <= BIN_FULL);
        let queue: *mut PageQueue = index_array(addr_of_mut!((*heap).page_queues), bin);
        debug_assert!(bin >= BINS || (*page).xblock_size as usize == (*queue).block_size);
        //  mi_assert_expensive(mi_page_queue_contains(pq, page));
        queue
    }

    unsafe fn from_page(page: *mut Page) -> *mut PageQueue {
        Self::from_page_and_heap(page, Page::heap(page))
    }

    /// Find a page with free blocks.
    #[inline]
    pub unsafe fn find_free(queue: *mut PageQueue, heap: *mut Heap, first_try: bool) -> *mut Page {
        let mut page = (*queue).list.first;
        while !page.is_null() {
            let next = (*page).node.next;

            // FIXME
            // 0. collect freed blocks by us and other threads
            //    _page_free_collect(page, false);

            // 1. if the page contains free blocks, we are done
            if !(*page).free.is_null() {
                break;
            }

            // 2. Try to extend
            if (*page).capacity < (*page).reserved {
                Page::extend_free(page);
                break;
            }

            // 3. If the page is completely full, move it to the `pages_full`
            // queue so we don't visit long-lived pages too often.
            // Page::to_full(page, queue);

            page = next;
        }

        if !page.is_null() {
            (*page).retire_expire = 0;
            return page;
        }

        Heap::collect_retired(heap, false); // perhaps make a page available
        let page = Page::fresh(heap, queue);
        if page.is_null() && first_try {
            // out-of-memory _or_ an abandoned page with free blocks was reclaimed, try once again
            Self::find_free(queue, heap, false)
        } else {
            page
        }
    }
}

bitflags! {
    #[derive( Clone, Copy, PartialEq, Eq, Debug)]
    pub struct PageFlags: u8 {
        const IN_FULL = 1;
        const HAS_ALIGNED = 1 << 1;
    }
}

bitflags! {
    #[derive( Clone, Copy, PartialEq, Eq, Debug)]
    pub struct PDelayedFreeFlags: u8 {
        const IN_FULL = 1;
        const HAS_ALIGNED = 1 << 1;
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum DelayedMode {
    UseDelayedFree = 0,   // push on the owning heap thread delayed list
    DelayedFreeing = 1,   // temporary: another thread is accessing the owning heap
    NoDelayedFree = 2, // optimize: push on page local thread free queue if another block is already in the heap thread delayed free list
    NeverDelayedFree = 3, // sticky, only resets on page reclaim
}

struct DelayedFree;

impl DelayedFree {
    fn new(block: *mut Block, mode: DelayedMode) -> *mut DelayedFree {
        block.map_addr(|addr| addr | (mode as usize)).cast()
    }

    fn change_mode(delayed: *mut DelayedFree, mode: DelayedMode) -> *mut DelayedFree {
        Self::new(Self::block(delayed), mode)
    }

    fn change_block(delayed: *mut DelayedFree, block: *mut Block) -> *mut DelayedFree {
        Self::new(block, Self::mode(delayed))
    }

    fn mode(delayed: *mut DelayedFree) -> DelayedMode {
        match (delayed.addr() & 0x3) {
            0 => DelayedMode::UseDelayedFree,
            1 => DelayedMode::DelayedFreeing,
            2 => DelayedMode::NoDelayedFree,
            3 => DelayedMode::NeverDelayedFree,
            _ => panic!(),
        }
    }

    fn block(delayed: *mut DelayedFree) -> *mut Block {
        delayed.map_addr(|addr| addr & !0x3).cast()
    }
}

#[derive(Debug)]
pub struct Page {
    // "owned" by the segment
    pub segment_idx: u8, // index in the segment `pages` array, `page == &segment->pages[page->segment_idx]`

    // FIXME: Bitflags
    pub segment_in_use: bool, // `true` if the segment allocated this page
    pub is_committed: bool,   // `true` if the page virtual memory is committed
    pub is_zero_init: bool,   // `true` if the page was initially zero initialized

    pub capacity: u16, // number of blocks committed, must be the first field, see `segment.c:page_clear`
    pub reserved: u16, // number of blocks reserved in memory
    pub flags: AtomicU8, // `in_full` and `has_aligned` flags (8 bits). Holds PageFlags

    //free_is_zero:bool;    // `true` if the blocks in the free list are zero initialized
    pub retire_expire: u8, // expiration count for retired blocks

    pub free: *mut Block, // list of available free blocks (`malloc` allocates from this list)
    pub used: usize,      // number of blocks in use (including blocks in `thread_free`)
    pub xblock_size: u32, // size available in each block (always `>0`)
    pub local_free: *mut Block, // list of deferred free blocks by this thread (migrates to `free`)

    xthread_free: AtomicPtr<DelayedFree>, // list of deferred free blocks freed by other threads
    xheap: AtomicPtr<Heap>,

    node: Node<Page>,
    /*

    // "owned" by the segment
    uint8_t               segment_idx;       // index in the segment `pages` array, `page == &segment->pages[page->segment_idx]`
    uint8_t               segment_in_use:1;  // `true` if the segment allocated this page
    uint8_t               is_committed:1;    // `true` if the page virtual memory is committed
    uint8_t               is_zero_init:1;    // `true` if the page was initially zero initialized

    // layout like this to optimize access in `mi_malloc` and `mi_free`
    uint16_t              capacity;          // number of blocks committed, must be the first field, see `segment.c:page_clear`
    uint16_t              reserved;          // number of blocks reserved in memory
    mi_page_flags_t       flags;             // `in_full` and `has_aligned` flags (8 bits)
    uint8_t               free_is_zero:1;    // `true` if the blocks in the free list are zero initialized
    uint8_t               retire_expire:7;   // expiration count for retired blocks

    mi_block_t*           free;              // list of available free blocks (`malloc` allocates from this list)
    uint32_t              used;              // number of blocks in use (including blocks in `thread_free`)
    uint32_t              xblock_size;       // size available in each block (always `>0`)
    mi_block_t*           local_free;        // list of deferred free blocks by this thread (migrates to `free`)

    #if (MI_ENCODE_FREELIST || MI_PADDING)
    uintptr_t             keys[2];           // two random keys to encode the free lists (see `_mi_block_next`) or padding canary
    #endif

    _Atomic(mi_thread_free_t) xthread_free;  // list of deferred free blocks freed by other threads
    _Atomic(uintptr_t)        xheap;

    struct mi_page_s*     next;              // next page owned by this thread with the same `block_size`
    struct mi_page_s*     prev;              // previous page owned by this thread with the same `block_size`
       */
}

unsafe impl Sync for Page {}
static EMPTY_PAGE: Page = Page::EMPTY;

impl Page {
    #[allow(clippy::declare_interior_mutable_const)]
    pub const EMPTY: Page = Page {
        segment_idx: 0,

        segment_in_use: false,
        is_committed: false,
        is_zero_init: false,

        free: null_mut(),
        local_free: null_mut(),
        used: 0,

        node: Node::UNUSED,

        capacity: 0,
        reserved: 0,

        flags: AtomicU8::new(PageFlags::empty().bits()),

        retire_expire: 0,

        xblock_size: 0,

        xthread_free: AtomicPtr::new(null_mut()),
        xheap: AtomicPtr::new(null_mut()),
    };

    pub const EMPTY_PTR: *mut Page = addr_of!(EMPTY_PAGE) as *mut Page;

    #[inline]
    pub fn node(page: *mut Page) -> *mut Node<Page> {
        unsafe { addr_of_mut!((*page).node) }
    }

    #[inline]
    pub fn flags(page: *mut Page) -> PageFlags {
        unsafe { PageFlags::from_bits_retain((*page).flags.load(Ordering::Relaxed)) }
    }

    #[inline]
    fn modify_flags(page: *mut Page, f: impl FnOnce(PageFlags) -> PageFlags) {
        unsafe {
            let flags = PageFlags::from_bits_retain((*page).flags.load(Ordering::Relaxed));
            (*page).flags.store(f(flags).bits(), Ordering::Relaxed)
        }
    }

    #[inline]
    pub unsafe fn insert_flags(page: *mut Page, flags: PageFlags) {
        Page::modify_flags(page, |mut f| {
            f.insert(flags);
            f
        });
    }

    #[inline]
    pub unsafe fn set_flags(page: *mut Page, flags: PageFlags, value: bool) {
        Page::modify_flags(page, |mut f| {
            f.set(flags, value);
            f
        });
    }

    #[inline]
    pub unsafe fn remove_flags(page: *mut Page, flags: PageFlags) {
        Page::modify_flags(page, |mut f| {
            f.remove(flags);
            f
        });
    }

    #[inline]
    pub unsafe fn segment(page: *mut Page) -> *mut Segment {
        let segment = Segment::from_pointer(page.cast());
        debug_assert!(
            segment.is_null() || page == Segment::page(segment, (*page).segment_idx as usize)
        );
        segment
    }

    // Quick page start for initialized pages
    #[inline]
    pub unsafe fn start(page: *mut Page, segment: *mut Segment) -> (*mut u8, usize) {
        let bsize = (*page).xblock_size as usize;
        debug_assert!(bsize > 0 && (bsize % PTR_SIZE) == 0);
        Segment::page_start(segment, page, bsize, &mut 0)
    }

    #[inline]
    pub unsafe fn heap(page: *mut Page) -> *mut Heap {
        (*page).xheap.load(Ordering::Relaxed)
    }

    #[inline]
    pub unsafe fn thread_free_flag(page: *mut Page) -> DelayedMode {
        DelayedFree::mode((*page).xthread_free.load(Ordering::Relaxed))
    }

    #[inline]
    unsafe fn set_heap(page: *mut Page, heap: *mut Heap) {
        debug_assert!(Page::thread_free_flag(page) != DelayedMode::DelayedFreeing);
        (*page).xheap.store(heap, Ordering::Release)
    }

    // Adjust a block that was allocated aligned, to the actual start of the block in the page.
    pub unsafe fn unalign_pointer(
        page: *mut Page,
        segment: *mut Segment,
        ptr: *mut u8,
    ) -> *mut Block {
        let diff = ptr.offset_from(Page::start(page, segment).0);
        let adjust = (diff as usize) % Page::block_size(page);
        ptr.map_addr(|addr| addr - adjust).cast()
    }

    // Get the block size of a page (special case for huge objects)
    pub unsafe fn block_size(page: *mut Page) -> usize {
        let bsize = (*page).xblock_size;
        debug_assert!(bsize > 0);
        if likely(bsize < HUGE_BLOCK_SIZE) {
            bsize as usize
        } else {
            let mut psize = 0;
            Segment::page_start(Page::segment(page), page, bsize as usize, &mut psize);
            psize
        }
    }

    #[inline]
    pub unsafe fn immediately_available(page: *mut Page) -> bool {
        !(*page).free.is_null()
    }

    #[inline]
    pub unsafe fn all_free(page: *mut Page) -> bool {
        (*page).used == 0
    }

    // Retire a page with no more used blocks
    // Important to not retire too quickly though as new
    // allocations might coming.
    // Note: called from `mi_free` and benchmarks often
    // trigger this due to freeing everything and then
    // allocating again so careful when changing this.
    pub unsafe fn retire(page: *mut Page) {
        debug_assert!(!page.is_null());
        //  mi_assert_expensive(_mi_page_is_valid(page));
        debug_assert!(Page::all_free(page));

        Page::remove_flags(page, PageFlags::HAS_ALIGNED);

        // don't retire too often..
        // (or we end up retiring and re-allocating most of the time)
        // NOTE: refine this more: we should not retire if this
        // is the only page left with free blocks. It is not clear
        // how to check this efficiently though...
        // for now, we don't retire if it is the only page left of this size class.
        let queue = PageQueue::from_page(page);
        if likely((*page).xblock_size as usize <= MAX_RETIRE_SIZE && !(*queue).is_special()) {
            // not too large && not full or huge queue?
            if ((*queue).list.last == page && (*queue).list.first == page) {
                // the only page in the queue?
                (*page).retire_expire = if (*page).xblock_size as usize <= SMALL_OBJ_SIZE_MAX {
                    RETIRE_CYCLES
                } else {
                    RETIRE_CYCLES / 4
                };
                let heap = Page::heap(page);
                debug_assert!(queue >= (*heap).page_queues.as_mut_ptr());
                let index = queue.offset_from((*heap).page_queues.as_mut_ptr()) as usize;
                debug_assert!(index < BINS);
                if (index < (*heap).page_retired_min) {
                    (*heap).page_retired_min = index;
                }
                if (index > (*heap).page_retired_max) {
                    (*heap).page_retired_max = index;
                }
                debug_assert!(Page::all_free(page));
                return; // dont't free after all
            }
        }

        Page::free(page, queue, false);
    }

    pub unsafe fn free(page: *mut Page, queue: *mut PageQueue, force: bool) {
        // no more aligned blocks in here
        Page::remove_flags(page, PageFlags::HAS_ALIGNED);

        // remove from the page list
        // (no need to do _mi_heap_delayed_free first as all blocks are already free)
        let segment_data = &mut (*Page::heap(page)).thread_data.segment as *mut _;
        PageQueue::remove(queue, page);

        // and free it
        Page::set_heap(page, null_mut());
        Segment::page_free(page, force, segment_data);
    }

    // Collect the local `thread_free` list using an atomic exchange.
    // Note: The exchange must be done atomically as this is used right after
    // moving to the full list in `mi_page_collect_ex` and we need to
    // ensure that there was no race where the page became unfull just before the move.
    unsafe fn thread_free_collect(page: *mut Page) {
        let mut head;
        let mut tfree = (*page).xthread_free.load(Ordering::Relaxed);
        loop {
            head = DelayedFree::block(tfree);
            let tfreex = DelayedFree::change_block(tfree, null_mut());
            if compare_exchange_weak_acq_rel(&(*page).xthread_free, &mut tfree, tfreex) {
                break;
            }
        }

        // return if the list is empty
        if head.is_null() {
            return;
        }

        // find the tail -- also to get a proper count (without data races)
        let max_count = (*page).capacity; // cannot collect more than capacity
        let mut count = 1;
        let mut tail = head;
        let mut next;
        while {
            next = (*tail).next;
            !next.is_null() && count <= max_count
        } {
            count += 1;
            tail = next;
        }

        debug_assert!(count <= max_count, "corrupted thread-free list");
        // if `count > max_count` there was a memory corruption (possibly infinite list due to double multi-threaded free)
        // FIXME: Remove branch?
        if (count > max_count) {
            return; // the thread-free items cannot be freed
        }

        // and append the current local free list
        (*tail).next = (*page).local_free;
        (*page).local_free = head;

        // update counts now
        (*page).used -= count as usize;
    }

    pub unsafe fn free_collect(page: *mut Page, force: bool) {
        // collect the thread free list
        if (force || !DelayedFree::block((*page).xthread_free.load(Ordering::Relaxed)).is_null()) {
            // quick test to avoid an atomic operation
            Page::thread_free_collect(page);
        }

        // and the local free list
        if !(*page).local_free.is_null() {
            if likely((*page).free.is_null()) {
                // usual case
                (*page).free = (*page).local_free;
                (*page).local_free = null_mut();
                //  (*page).free_is_zero = false;
            } else if (force) {
                // append -- only on shutdown (force) as this is a linear operation
                let mut tail = (*page).local_free;
                let mut next;
                while {
                    next = (*tail).next;
                    !next.is_null()
                } {
                    tail = next;
                }
                (*tail).next = (*page).free;
                (*page).free = (*page).local_free;
                (*page).local_free = null_mut();
                //  (*page).free_is_zero = false;
            }
        }

        debug_assert!(!force || (*page).local_free.is_null());
    }

    // return true if successful
    pub unsafe fn free_delayed_block(block: *mut Block) -> bool {
        // get segment and page
        let segment = Segment::from_pointer_checked(block.cast());
        debug_assert!(thread_id() == (*segment).thread_id.load(Ordering::Relaxed));
        let page = Segment::page_from_pointer(segment, block.cast());

        // Clear the no-delayed flag so delayed freeing is used again for this page.
        // This must be done before collecting the free lists on this page -- otherwise
        // some blocks may end up in the page `thread_free` list with no blocks in the
        // heap `thread_delayed_free` list which may cause the page to be never freed!
        // (it would only be freed if we happen to scan it in `mi_page_queue_find_free_ex`)
        if (!Page::try_use_delayed_free(
            page,
            DelayedMode::UseDelayedFree,
            false, /* dont overwrite never delayed */
        )) {
            return false;
        }

        // collect all other non-local frees to ensure up-to-date `used` count
        Page::free_collect(page, false);

        // and free the block (possibly freeing the page as well since used is updated)
        Page::free_block(page, true, block);
        true
    }

    pub unsafe fn use_delayed_free(page: *mut Page, delay: DelayedMode, override_never: bool) {
        while (!Page::try_use_delayed_free(page, delay, override_never)) {
            // This ensures the value is set.
            yield_now();
        }
    }

    unsafe fn try_use_delayed_free(
        page: *mut Page,
        delay: DelayedMode,
        override_never: bool,
    ) -> bool {
        let mut tfree;
        let mut yield_count = 0;
        loop {
            tfree = (*page).xthread_free.load(Ordering::Acquire); // note: must acquire as we can break/repeat this loop and not do a CAS;
            let tfreex = DelayedFree::change_mode(tfree, delay);
            let old_delay = DelayedFree::mode(tfree);
            if unlikely(old_delay == DelayedMode::DelayedFreeing) {
                if yield_count >= 4 {
                    return false; // give up after 4 tries
                }
                yield_count += 1;
                yield_now(); // delay until outstanding MI_DELAYED_FREEING are done.
                             // tfree = mi_tf_set_delayed(tfree, MI_NO_DELAYED_FREE); // will cause CAS to busy fail
            } else if delay == old_delay {
                break; // avoid atomic operation if already equal
            } else if (!override_never && old_delay == DelayedMode::NeverDelayedFree) {
                break; // leave never-delayed flag set
            }

            if (old_delay != DelayedMode::DelayedFreeing)
                && compare_exchange_weak_release(&(*page).xthread_free, &mut tfree, tfreex)
            {
                break;
            }
        }

        true // success
    }

    #[inline]
    pub unsafe fn free_non_local_block(page: *mut Page, block: *mut Block) {
        // first see if the segment was abandoned and we can reclaim it
        let segment = Page::segment(page);
        /*
        let option_abandoned_reclaim_on_free = false;
                if option_abandoned_reclaim_on_free && (*segment).thread_id.load(Ordering::Relaxed) == 0 {
                    // the segment is abandoned, try to reclaim it into our heap
                    if (_mi_segment_attempt_reclaim(mi_heap_get_default(), segment)) {
                        debug_assert!(Segment::is_local(segment));
                        free(block); // recursively free as now it will be a local free in our heap
                        return;
                    }
                }
        */
        /*
        // FIXME
                if (*segment).page_kind == PageKind::Huge {
                    // huge pages are special as they occupy the entire segment
                    // as these are large we reset the memory occupied by the page so it is available to other threads
                    // (as the owning thread needs to actually free the memory later).
                    _mi_segment_huge_page_reset(segment, page, block);
                }
        */

        // Try to put the block on either the page-local thread free list, or the heap delayed free list.
        let mut use_delayed;
        let mut tfree = (*page).xthread_free.load(Ordering::Relaxed);
        loop {
            use_delayed = (DelayedFree::mode(tfree) == DelayedMode::UseDelayedFree);
            let tfreex = if unlikely(use_delayed) {
                // unlikely: this only happens on the first concurrent free in a page that is in the full list
                DelayedFree::change_mode(tfree, DelayedMode::DelayedFreeing)
            } else {
                // usual: directly add to page thread_free list
                (*block).next = DelayedFree::block(tfree);
                DelayedFree::new(block, DelayedFree::mode(tfree))
            };

            if compare_exchange_weak_release(&(*page).xthread_free, &mut tfree, tfreex) {
                break;
            }
        }

        // Note: use_delayed = true seems to mean that the thread owning page cannot terminate / free the heap data
        // as we can read it here, so this is effectively a blocking operation.
        // Could this be fixed by having both the segment / page and thread own the heap via a ref count?

        if unlikely(use_delayed) {
            // racy read on `heap`, but ok because DelayedMode::DelayedFreeing is set (see `mi_heap_delete` and `mi_heap_collect_abandon`)
            let heap = (*page).xheap.load(Ordering::Acquire); //mi_page_heap(page);
            debug_assert!(!heap.is_null());
            // FIXME: Is this sometimes NULL?
            if (!heap.is_null()) {
                // add to the delayed free list of this heap. (do this atomically as the lock only protects heap memory validity)
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

            // and reset the DelayedMode::DelayedFreeing flag
            let mut tfree = (*page).xthread_free.load(Ordering::Relaxed);
            loop {
                // FIXME: Is this early exit correct? (not in original code)
                if DelayedFree::mode(tfree) != DelayedMode::DelayedFreeing {
                    break;
                }

                //debug_assert!(DelayedFree::mode(tfree) == DelayedMode::DelayedFreeing);

                let tfreex = DelayedFree::change_mode(tfree, DelayedMode::NoDelayedFree);
                if compare_exchange_weak_release(&(*page).xthread_free, &mut tfree, tfreex) {
                    break;
                }
            }
        }
    }

    #[inline]
    pub unsafe fn free_block(page: *mut Page, local: bool, block: *mut Block) {
        // and push it on the free list
        if likely(local) {
            (*block).next = (*page).local_free;
            (*page).local_free = block;
            (*page).used -= 1;
            if unlikely(Page::all_free(page)) {
                Page::retire(page);
            } else if unlikely(Page::flags(page).contains(PageFlags::IN_FULL)) {
                Page::unfull(page);
            }
        } else {
            Page::free_non_local_block(page, block);
        }
    }

    // Abandon a page with used blocks at the end of a thread.
    // Note: only call if it is ensured that no references exist from
    // the `page->heap->thread_delayed_free` into this page.
    // Currently only called through `Heap::collect` which ensures this.
    pub unsafe fn abandon(page: *mut Page, queue: *mut PageQueue) {
        debug_assert!(!page.is_null());
        // mi_assert_expensive(_mi_page_is_valid(page));

        let heap = Page::heap(page);

        debug_assert!(!heap.is_null());

        // remove from our page list
        PageQueue::remove(queue, page);

        // page is no longer associated with our heap
        debug_assert!(Page::thread_free_flag(page) == DelayedMode::NeverDelayedFree);
        Page::set_heap(page, null_mut());

        /*
          #if (MI_DEBUG>1) && !MI_TRACK_ENABLED
          if cfg!(debug_assertions) {
            // check there are no references left..
            for (mi_block_t* block = (mi_block_t*)(*heap).thread_delayed_free; block != NULL; block = mi_block_nextx(pheap, block, pheap->keys)) {
              debug_assert!(_mi_ptr_page(block) != page);
            }
        }
          #endif
          */
        // and abandon it
        Segment::page_abandon(page, &mut (*heap).thread_data.segment);
    }

    // Move a page from the full list back to a regular list
    #[inline]
    pub unsafe fn unfull(page: *mut Page) {
        debug_assert!(!page.is_null());
        // mi_assert_expensive(_mi_page_is_valid(page));
        debug_assert!(Page::flags(page).contains(PageFlags::IN_FULL));

        // if (!mi_page_is_in_full(page)) return;

        let heap = Page::heap(page);
        let pqfull: *mut PageQueue = addr_of_mut!((*heap).page_queues[BIN_FULL]);
        Page::remove_flags(page, PageFlags::IN_FULL); // to get the right queue
        let pq = PageQueue::from_page_and_heap(page, heap);
        Page::insert_flags(page, PageFlags::IN_FULL);
        PageQueue::enqueue_from(pq, pqfull, page);
    }

    // Start of the page available memory; can be used on uninitialized pages (only `segment_idx` must be set)
    unsafe fn page_start(page: *mut Page, segment: *mut Segment) -> (*mut u8, usize) {
        Segment::page_start(segment, page, (*page).xblock_size as usize, &mut 0)
    }

    unsafe fn init(page: *mut Page, heap: *mut Heap, block_size: usize) {
        debug_assert!(!page.is_null());
        let segment = Page::segment(page);
        debug_assert!(!segment.is_null());
        debug_assert!(block_size > 0);
        // set fields
        Page::set_heap(page, heap);
        let (_, page_size) = Segment::page_start(segment, page, block_size, &mut 0);

        (*page).xblock_size = if block_size < HUGE_BLOCK_SIZE as usize {
            block_size as u32
        } else {
            HUGE_BLOCK_SIZE
        };
        debug_assert!(page_size / block_size < (1 << 16));
        (*page).reserved = (page_size / block_size) as u16;
        debug_assert!((*page).reserved > 0);
        // (*page).free_is_zero = (*page).is_zero_init;

        debug_assert!((*page).capacity == 0);
        debug_assert!((*page).free.is_null());
        debug_assert!((*page).used == 0);
        //  debug_assert!((*page).xthread_free == 0);
        debug_assert!((*page).node.next.is_null());
        debug_assert!((*page).node.prev.is_null());
        debug_assert!((*page).retire_expire == 0);
        debug_assert!(!Page::flags(page).contains(PageFlags::HAS_ALIGNED));
        //mi_assert_expensive(mi_page_is_valid_init(page));

        // initialize an initial free list
        Page::extend_free(page);
        //mi_assert(mi_page_immediate_available(page));
    }

    // allocate a fresh page from a segment
    unsafe fn fresh_alloc(
        heap: *mut Heap,
        queue: *mut PageQueue,
        block_size: usize,
        page_alignment: usize,
    ) -> *mut Page {
        debug_assert!(!queue.is_null());
        //  debug_assert!(mi_heap_contains_queue(heap, queue));
        debug_assert!(
            page_alignment > 0
                || block_size > LARGE_OBJ_SIZE_MAX
                || block_size == (*queue).block_size
        );

        let page = Segment::page_alloc(
            heap,
            block_size,
            page_alignment,
            &mut (*heap).thread_data.segment,
        );
        if page.is_null() {
            // this may be out-of-memory, or an abandoned page was reclaimed (and in our queue)
            return null_mut();
        }

        debug_assert!(!queue.is_null() || Page::block_size(page) >= block_size);
        // a fresh page was found, initialize it
        let full_block_size = if queue.is_null() || (*queue).is_huge() {
            Page::block_size(page)
        } else {
            block_size
        }; // see also: mi_segment_huge_page_alloc
        debug_assert!(full_block_size >= block_size);
        Page::init(page, heap, full_block_size);
        //  mi_heap_stat_increase(heap, pages, 1);
        if !queue.is_null() {
            PageQueue::add(queue, page, heap);
        }
        //  mi_assert_expensive(_mi_page_is_valid(page));
        page
    }

    // Get a fresh page to use
    unsafe fn fresh(heap: *mut Heap, queue: *mut PageQueue) -> *mut Page {
        // debug_assert!(mi_heap_contains_queue(heap, pq));
        let page = Page::fresh_alloc(heap, queue, (*queue).block_size, 0);
        if (page.is_null()) {
            return null_mut();
        }
        debug_assert!((*queue).block_size == Page::block_size(page));
        debug_assert!(queue == Heap::page_queue_for_size(heap, Page::block_size(page)));
        page
    }

    /*
        unsafe fn to_full(page: *mut Page, queue: *mut PageQueue,) {
               if (mi_page_is_in_full(page)){return;}
            mi_page_queue_enqueue_from(&mi_page_heap(page)->pages[MI_BIN_FULL], pq, page);
            _mi_page_free_collect(page,false);  // try to collect right away in case another thread freed just before MI_USE_DELAYED_FREE was set

        }
    */
    #[inline]
    pub unsafe fn alloc(page: *mut Page, layout: Layout, heap: *mut Heap) -> *mut u8 {
        let block = (*page).free;
        if block.is_null() {
            Heap::alloc_slow(heap, layout)
        } else {
            (*page).free = (*block).next;
            (*page).used = (*page).used.wrapping_add(1);
            block.cast()
        }
    }

    // Index a block in a page
    unsafe fn block_at(
        page: *mut Page,
        page_start: *mut u8,
        block_size: usize,
        i: usize,
    ) -> *mut Block {
        debug_assert!(i <= (*page).reserved as usize);
        page_start.byte_add(i * block_size) as *mut Block
    }

    unsafe fn free_list_extend(page: *mut Page, bsize: usize, extend: usize) {
        let (page_area, _) = Page::page_start(page, Page::segment(page));

        let start = Page::block_at(page, page_area, bsize, (*page).capacity as usize);

        // initialize a sequential free list
        let last = Page::block_at(
            page,
            page_area,
            bsize,
            (*page).capacity as usize + extend - 1,
        );
        let mut block = start;
        while block <= last {
            let next = block.byte_add(bsize);
            (*block).next = next;
            block = next;
        }
        // prepend to free list (usually `null_mut()`)
        (*last).next = (*page).free;
        (*page).free = start;
    }

    /*
    unsafe fn free_list_extend_secure(page: *mut Page, heap: *mut Heap,   bsize: usize,  extend: usize,) {
        let (page_area, _) = Page::page_start(page, Page::segment(page));

        // initialize a randomized free list
        // set up `slice_count` slices to alternate between
       let shift = MI_MAX_SLICE_SHIFT;
        while ((extend >> shift) == 0) {
          shift--;
        }
      let slice_count = (size_t)1U << shift;
      let slice_extend = extend / slice_count;
        mi_assert_internal(slice_extend >= 1);
        mi_block_t* blocks[MI_MAX_SLICES];   // current start of the slice
       let      counts[MI_MAX_SLICES];   // available objects in the slice
        for (size_t i = 0; i < slice_count; i++) {
          blocks[i] = mi_page_block_at(page, page_area, bsize, (*page).capacity + i*slice_extend);
          counts[i] = slice_extend;
        }
        counts[slice_count-1] += (extend % slice_count);  // final slice holds the modulus too (todo: distribute evenly?)

        // and initialize the free list by randomly threading through them
        // set up first element
     let r = _mi_heap_random_next(heap);
       let current = r % slice_count;
        counts[current]--;
        mi_block_t* const free_start = blocks[current];
        // and iterate through the rest; use `random_shuffle` for performance
        uintptr_t rnd = _mi_random_shuffle(r|1); // ensure not 0
        for (size_t i = 1; i < extend; i++) {
          // call random_shuffle only every INTPTR_SIZE rounds
        let round = i%MI_INTPTR_SIZE;
          if (round == 0) rnd = _mi_random_shuffle(rnd);
          // select a random next slice index
         let next = ((rnd >> 8*round) & (slice_count-1));
          while (counts[next]==0) {                            // ensure it still has space
            next++;
            if (next==slice_count) next = 0;
          }
          // and link the current block to it
          counts[next]--;
          mi_block_t* const block = blocks[current];
          blocks[current] = (mi_block_t*)((uint8_t*)block + bsize);  // bump to the following block
          mi_block_set_next(page, block, blocks[next]);   // and set next; note: we may have `current == next`
          current = next;
        }
        // prepend to the free list (usually null_mut())
        mi_block_set_next(page, blocks[current], page->free);  // end of the list
        page->free = free_start;
      }
      */

    // Extend the capacity (up to reserved) by initializing a free list
    // We do at most `MAX_EXTEND` to avoid touching too much memory
    // Note: we also experimented with "bump" allocation on the first
    // allocations but this did not speed up any benchmark (due to an
    // extra test in malloc? or cache effects?)
    pub unsafe fn extend_free(page: *mut Page) {
        if (*page).capacity >= (*page).reserved {
            return;
        }

        let (_, page_size) = Page::page_start(page, Page::segment(page));

        // calculate the extend count
        let bsize = if (*page).xblock_size < HUGE_BLOCK_SIZE {
            (*page).xblock_size as usize
        } else {
            page_size
        };
        let mut extend = ((*page).reserved - (*page).capacity) as usize;

        let mut max_extend = if bsize >= MAX_EXTEND_SIZE {
            MIN_EXTEND
        } else {
            MAX_EXTEND_SIZE / bsize
        };
        if max_extend < MIN_EXTEND {
            max_extend = MIN_EXTEND;
        }

        if extend > max_extend {
            // ensure we don't touch memory beyond the page to reduce page commit.
            // the `lean` benchmark tests this. Going from 1 to 8 increases rss by 50%.
            extend = max_extend;
        }

        // and append the extend the free list
        // if extend < MIN_SLICES {
        Page::free_list_extend(page, bsize, extend);
        // } else {
        // TODO: Do we need the secure variant?
        //     page_free_list_extend_secure(heap, page, bsize, extend);
        //  }
        // enable the new free list
        (*page).capacity += extend as u16;
    }
}
