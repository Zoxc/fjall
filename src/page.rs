use crate::heap::Heap;
use crate::linked_list::{List, Node};
use crate::segment::{
    cookie, Segment, SegmentThreadData, Whole, WholeOrStatic, OPTION_PURGE_DELAY, SEGMENT_ALIGN,
};
use crate::{
    align_down, bin_index, compare_exchange_weak_acq_rel, compare_exchange_weak_release, div, rem,
    system, thread_id, yield_now, Ptr, BINS, BIN_FULL, BIN_FULL_BLOCK_SIZE, BIN_HUGE_BLOCK_SIZE,
    LARGE_OBJ_SIZE_MAX, LOCAL_HEAP, MAX_EXTEND_SIZE, MIN_EXTEND, SMALL_OBJ_SIZE_MAX, WORD_SIZE,
};
use bitflags::bitflags;
use core::intrinsics::likely;
use core::{alloc::Layout, ptr::null_mut};
use sptr::Strict;
use std::cell::Cell;
use std::intrinsics::unlikely;
use std::num::NonZeroUsize;
use std::ptr::{addr_of, addr_of_mut};
use std::sync::atomic::{AtomicPtr, AtomicU8, Ordering};

pub const SMALL_PAGE_SHIFT: usize = 16;
pub const MEDIUM_PAGE_SHIFT: usize = 3 + SMALL_PAGE_SHIFT;
pub const LARGE_PAGE_SHIFT: usize = 2 + MEDIUM_PAGE_SHIFT;

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

pub struct AllocatedBlock;

pub struct FreeBlock {
    pub next: Cell<Option<Whole<FreeBlock>>>,
}

pub(crate) struct BlockListIter {
    current: Option<Whole<FreeBlock>>,
}

impl Iterator for BlockListIter {
    type Item = Whole<FreeBlock>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(current) = self.current {
            self.current = current.next.get();
            Some(current)
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub struct BlockList {
    pub first: Cell<Option<Whole<FreeBlock>>>,
}

impl BlockList {
    pub const fn empty() -> Self {
        BlockList {
            first: Cell::new(None),
        }
    }

    #[inline]
    pub unsafe fn is_empty(&self) -> bool {
        self.first.get().is_none()
    }

    #[inline]
    pub unsafe fn iter(&self) -> BlockListIter {
        BlockListIter {
            current: self.first.get(),
        }
    }

    #[inline]
    pub unsafe fn validate(&self, page: Whole<Page>) {
        if !cfg!(debug_assertions) {
            return;
        }

        let (start, size) = Page::page_start(page, Page::segment(page));
        let start = start.as_ptr();
        let end = start.add(size);

        for block in self.iter() {
            internal_assert!(block.as_ptr().cast() >= start);
            internal_assert!(block.as_ptr().cast() < end);
        }
    }

    #[inline]
    pub unsafe fn count(&self) -> usize {
        self.iter().count()
    }

    #[inline]
    pub unsafe fn take(&self) -> Option<BlockList> {
        if self.is_empty() {
            None
        } else {
            let result = BlockList {
                first: self.first.clone(),
            };
            self.first.set(None);
            Some(result)
        }
    }

    /// `list` must not be empty.
    #[inline]
    pub unsafe fn extend_back(&self, list: BlockList) {
        let head = list.first.get().unwrap_unchecked();
        let mut tail = head;
        let mut next;

        loop {
            next = tail.next.get();
            if let Some(next) = next {
                tail = next;
            } else {
                break;
            }
        }

        self.push_range_front(head, tail);
    }

    #[inline]
    pub unsafe fn push_range_front(&self, first: Whole<FreeBlock>, last: Whole<FreeBlock>) {
        last.next.set(self.first.get());
        self.first.set(Some(first));
    }

    #[inline]
    pub unsafe fn push_front(&self, entry: Whole<FreeBlock>) {
        entry.next.set(self.first.get());
        self.first.set(Some(entry));
    }

    #[inline]
    pub unsafe fn pop_front(&self) -> Option<Whole<FreeBlock>> {
        self.first.get().inspect(|block| {
            self.first.set(block.next.get());
        })
    }
}

#[derive(Clone, Debug)]
pub struct PageQueue {
    pub list: List<Whole<Page>>,
    pub block_size: usize,
}

impl PageQueue {
    #[inline]
    pub const fn new(block_size: usize) -> Self {
        assert!(block_size > 0);
        PageQueue {
            list: List::empty(),
            block_size,
        }
    }

    #[inline]
    pub unsafe fn is_huge(&self) -> bool {
        self.block_size == BIN_HUGE_BLOCK_SIZE
    }

    #[inline]
    unsafe fn is_full(&self) -> bool {
        self.block_size == BIN_FULL_BLOCK_SIZE
    }

    #[inline]
    unsafe fn is_special(&self) -> bool {
        self.block_size > LARGE_OBJ_SIZE_MAX
    }

    unsafe fn add(queue: Ptr<PageQueue>, page: Whole<Page>, heap: Ptr<Heap>) {
        (*page).assert_heap(Some(heap));
        //   internal_assert!(!mi_page_queue_contains(queue, page));
        if (*queue).is_huge() {
        } else if (*queue).is_full() {
            internal_assert!((*page).flags().contains(PageFlags::IN_FULL));
        } else {
            internal_assert!((page.block_size()) == queue.block_size);
        }

        (*page).set_flags(PageFlags::IN_FULL, (*queue).is_full());

        queue.list.push_front(page, Page::node);

        // update direct
        Heap::queue_first_update(heap, queue);

        heap.page_count.set(heap.page_count.get() + 1);
    }

    unsafe fn remove(queue: Ptr<PageQueue>, page: Whole<Page>) {
        let heap = (*page).local_heap();
        let update = Some(page) == queue.list.first.get();

        queue.list.remove(page, Page::node);

        if update {
            Heap::queue_first_update(heap, queue);
        }

        heap.page_count.set(heap.page_count.get() - 1);

        (*page).remove_flags(PageFlags::IN_FULL);
    }

    unsafe fn enqueue_from(to: Ptr<PageQueue>, from: Ptr<PageQueue>, page: Whole<Page>) {
        //mi_assert_expensive(mi_page_queue_contains(from, page));
        // mi_assert_expensive(!mi_page_queue_contains(to, page));

        #[allow(clippy::nonminimal_bool)]
        {
            internal_assert!(
                (page.block_size() == to.block_size && page.block_size() == from.block_size)
                    || (page.block_size() == to.block_size && (*from).is_full())
                    || (page.block_size() == from.block_size && (*to).is_full())
                    || (page.block_size() > LARGE_OBJ_SIZE_MAX && (*to).is_huge())
                    || (page.block_size() > LARGE_OBJ_SIZE_MAX && (*to).is_full())
            );
        }

        let heap = (*page).local_heap();

        let update = Some(page) == from.list.first.get();

        from.list.remove(page, Page::node);

        if update {
            //  mi_assert_internal(mi_heap_contains_queue(heap, from));
            Heap::queue_first_update(heap, from);
        }

        if let Some(last) = to.list.last.get() {
            last.assert_heap(Some(heap));
        }

        let update = to.list.last.get().is_none();

        to.list.push_back(page, Page::node);

        if update {
            Heap::queue_first_update(heap, to);
        }

        (*page).set_flags(PageFlags::IN_FULL, (*to).is_full());
    }

    unsafe fn from_page_and_heap(page: Whole<Page>, heap: Ptr<Heap>) -> Ptr<PageQueue> {
        let bin = if (*page).flags().contains(PageFlags::IN_FULL) {
            BIN_FULL
        } else {
            bin_index(page.block_size())
        };
        internal_assert!(bin <= BIN_FULL);
        let queue = Heap::page_queue(heap, bin);
        internal_assert!(bin >= BINS || page.block_size() == queue.block_size);
        expensive_assert!(queue.list.contains(page, Page::node));
        queue
    }

    unsafe fn from_page(page: Whole<Page>) -> Ptr<PageQueue> {
        Self::from_page_and_heap(page, (*page).local_heap())
    }

    /// Find a page with free blocks.
    #[inline]
    pub unsafe fn find_free(
        queue: Ptr<PageQueue>,
        heap: Ptr<Heap>,
        first_try: bool,
    ) -> Option<Whole<Page>> {
        let mut current = queue.list.first.get();
        while let Some(page) = current {
            let next = page.node.next.get();

            // 0. collect freed blocks by us and other threads
            Page::free_collect(page, false);

            // 1. if the page contains free blocks, we are done
            if !page.free_blocks.is_empty() {
                break;
            }

            // 2. Try to extend
            if page.capacity < page.reserved {
                Page::extend_free(page);
                break;
            }

            // 3. Since the page is completely full, move it to the `pages_full`
            // queue so we don't visit long-lived pages too often.
            Page::to_full(page, queue);

            current = next;
        }

        if let Some(page) = current {
            page.retire_expire.set(0);
            return current;
        }

        Heap::collect_retired(heap, false); // perhaps make a page available
        let page = Page::fresh(heap, queue);
        if page.is_none() && first_try {
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

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum DelayedMode {
    UseDelayedFree = 0,   // push on the owning heap thread delayed list
    DelayedFreeing = 1,   // temporary: another thread is accessing the owning heap
    NoDelayedFree = 2, // optimize: push on page local thread free queue if another block is already in the heap thread delayed free list
    NeverDelayedFree = 3, // sticky, only resets on page reclaim
}

pub struct DelayedFree;

impl DelayedFree {
    fn new(block: *mut FreeBlock, mode: DelayedMode) -> *mut DelayedFree {
        block.map_addr(|addr| addr | (mode as usize)).cast()
    }

    fn change_mode(delayed: *mut DelayedFree, mode: DelayedMode) -> *mut DelayedFree {
        Self::new(Self::block(delayed), mode)
    }

    fn change_block(delayed: *mut DelayedFree, block: *mut FreeBlock) -> *mut DelayedFree {
        Self::new(block, Self::mode(delayed))
    }

    fn mode(delayed: *mut DelayedFree) -> DelayedMode {
        match delayed.addr() & 0x3 {
            0 => DelayedMode::UseDelayedFree,
            1 => DelayedMode::DelayedFreeing,
            2 => DelayedMode::NoDelayedFree,
            3 => DelayedMode::NeverDelayedFree,
            _ => panic!(),
        }
    }

    fn block(delayed: *mut DelayedFree) -> *mut FreeBlock {
        delayed.map_addr(|addr| addr & !0x3).cast()
    }
}

#[derive(Debug)]
#[repr(C)]
pub struct Page {
    // Cluster fields accessed in the alloc / free hot paths
    pub capacity: Cell<u16>, // number of blocks committed, must be the first field, see `segment.c:page_clear`
    pub reserved: Cell<u16>, // number of blocks reserved in memory
    pub flags: AtomicU8,     // `in_full` and `has_aligned` flags (8 bits). Holds PageFlags
    //free_is_zero:bool;    // `true` if the blocks in the free list are zero initialized
    pub free_blocks: BlockList, // list of available free blocks (`malloc` allocates from this list)
    pub local_deferred_free_blocks: BlockList, // list of deferred free blocks by this thread (migrates to `free`)

    /// number of blocks in use (including blocks in `thread_free`).
    /// This also stores the expiration counter for pages pending purging.
    pub used: Cell<u32>,

    // "owned" by the segment
    pub segment_idx: u8, // index in the segment `pages` array, `page == &segment->pages[page->segment_idx]`

    pub retire_expire: Cell<u8>, // expiration count for retired blocks

    // FIXME: Bitflags
    pub segment_in_use: Cell<bool>, // `true` if the segment allocated this page
    pub is_committed: Cell<bool>,   // `true` if the page virtual memory is committed
    // pub is_zero_init: Cell<bool>,   // `true` if the page was initially zero initialized
    /// Size available in each block (always `>0`).
    /// This stores `BIN_HUGE_BLOCK_SIZE` for huge pages instead of the real block size.
    /// FIXME: Use Option<NonZero> instead?
    pub xblock_size: Cell<u32>,

    pub remote_free_blocks: AtomicPtr<DelayedFree>, // list of deferred free blocks freed by other threads

    pub heap: AtomicPtr<Heap>,

    pub node: Node<Whole<Page>>,
}

unsafe impl Sync for Page {}
static EMPTY_PAGE: Page = Page::empty();

impl Page {
    pub const fn empty() -> Page {
        Page {
            segment_idx: 0,

            segment_in_use: Cell::new(false),
            is_committed: Cell::new(false),
            // is_zero_init: Cell::new(false),
            free_blocks: BlockList::empty(),
            local_deferred_free_blocks: BlockList::empty(),
            used: Cell::new(0),

            node: Node::UNUSED,

            capacity: Cell::new(0),
            reserved: Cell::new(0),

            flags: AtomicU8::new(PageFlags::empty().bits()),

            retire_expire: Cell::new(0),

            xblock_size: Cell::new(0),

            remote_free_blocks: AtomicPtr::new(null_mut()),
            heap: AtomicPtr::new(null_mut()),
        }
    }

    pub const EMPTY_PTR: *mut Page = addr_of!(EMPTY_PAGE) as *mut Page;

    #[inline]
    pub fn node(page: Whole<Page>) -> *mut Node<Whole<Page>> {
        unsafe { addr_of_mut!((*page.as_ptr()).node) }
    }

    #[inline]
    pub fn block_size(&self) -> usize {
        self.xblock_size.get() as usize
    }

    #[inline]
    pub fn flags(&self) -> PageFlags {
        PageFlags::from_bits_retain(self.flags.load(Ordering::Relaxed))
    }

    #[inline]
    fn modify_flags(&self, f: impl FnOnce(PageFlags) -> PageFlags) {
        let flags = PageFlags::from_bits_retain(self.flags.load(Ordering::Relaxed));
        self.flags.store(f(flags).bits(), Ordering::Relaxed)
    }

    // we re-use the `used` field for the expiration counter. Since this is a
    // a 32-bit field while the clock is always 64-bit we need to guard
    // against overflow, we use substraction to check for expiry which work
    // as long as the reset delay is under (2^30 - 1) milliseconds (~12 days)
    #[inline]
    pub fn purge_set_expire(&self) {
        internal_assert!(self.used.get() == 0);
        internal_assert!(OPTION_PURGE_DELAY > 0);
        self.used.set(
            system::clock_now()
                .unwrap_or(0)
                .wrapping_add(OPTION_PURGE_DELAY as u64) as u32,
        );
    }

    #[inline]
    pub fn purge_is_expired(&self, now: Option<u64>) -> bool {
        let expire = self.used.get() as i32;
        // Assume the page expired if we couldn't get the current time
        now.map(|now| now as i32 - expire >= 0).unwrap_or(true)
    }

    #[inline]
    pub unsafe fn purge_remove(page: Whole<Page>, data: &mut SegmentThreadData) {
        if data.pages_purge.may_contain(page, Page::node) {
            internal_assert!(!page.segment_in_use.get());
            page.used.set(0);
            data.pages_purge.remove(page, Page::node);
        }
    }

    #[inline]
    pub fn insert_flags(&self, flags: PageFlags) {
        self.modify_flags(|mut f| {
            f.insert(flags);
            f
        });
    }

    #[inline]
    pub fn set_flags(&self, flags: PageFlags, value: bool) {
        self.modify_flags(|mut f| {
            f.set(flags, value);
            f
        });
    }

    #[inline]
    pub fn remove_flags(&self, flags: PageFlags) {
        self.modify_flags(|mut f| {
            f.remove(flags);
            f
        });
    }

    /// This cannot use `&self` as we'd can't recover *mut Segment from it.
    #[inline]
    pub unsafe fn segment(page: Whole<Page>) -> Whole<Segment> {
        // V

        let segment: Whole<Segment> = page.map_addr(|addr| align_down(addr, SEGMENT_ALIGN)).cast();

        #[cfg(debug_assertions)]
        assert_eq!((*segment.as_ptr()).cookie, cookie(segment));

        internal_assert!(page == Segment::page(segment, page.segment_idx as usize));
        segment
    }

    // Quick page start for initialized pages
    #[inline]
    pub unsafe fn start(page: Whole<Page>, segment: Whole<Segment>) -> (Whole<u8>, usize) {
        let bsize = page.xblock_size.get() as usize;
        internal_assert!(bsize > 0 && (bsize % WORD_SIZE) == 0);
        Segment::page_start(segment, page, bsize, &mut 0)
    }

    #[inline]
    pub fn assert_heap(&self, value: Option<Ptr<Heap>>) {
        unsafe { internal_assert!(value == Ptr::new(self.heap.load(Ordering::Relaxed))) }
    }

    #[inline]
    pub unsafe fn local_heap(&self) -> Ptr<Heap> {
        let heap = unsafe { Ptr::new(self.heap.load(Ordering::Relaxed)) };
        if cfg!(debug_assertions) {
            if let Some(heap) = heap {
                LOCAL_HEAP.with(|local_heap| {
                    internal_assert!(local_heap.get() == heap.as_ptr());
                })
            }
        }
        heap.unwrap_unchecked()
    }

    #[inline]
    pub fn thread_free_flag(&self) -> DelayedMode {
        DelayedFree::mode(self.remote_free_blocks.load(Ordering::Relaxed))
    }

    #[inline]
    pub fn set_heap(&self, heap: Option<Ptr<Heap>>) {
        internal_assert!(self.thread_free_flag() != DelayedMode::DelayedFreeing);
        self.heap
            .store(Ptr::as_maybe_null_ptr(heap), Ordering::Release)
    }

    // Adjust a block that was allocated aligned, to the actual start of the block in the page.
    pub unsafe fn unalign_pointer(
        page: Whole<Page>,
        segment: Whole<Segment>,
        ptr: Whole<AllocatedBlock>,
    ) -> Whole<FreeBlock> {
        internal_assert!(segment.page_kind <= PageKind::Large);
        // This relies on the fact that allocation happens on multiplies of block size
        // from the page start.
        let page_start = Page::start(page, segment).0.as_ptr();

        // Use the `xblock_size` field directly as we know that
        // has the real block value for non-huge pages.
        let block_size = page.block_size();

        let relative_to_page_start = ptr.as_ptr().byte_offset_from(page_start);
        let adjust = rem(
            relative_to_page_start as usize,
            NonZeroUsize::new_unchecked(block_size),
        );
        let result = ptr.map_addr(|addr| addr - adjust).cast::<FreeBlock>();

        internal_assert!(
            result.as_ptr().cast()
                == page_start
                    .wrapping_add(((relative_to_page_start as usize) / block_size) * block_size)
        );

        result
    }

    // Get the block size of a page (special case for huge objects)
    pub unsafe fn actual_block_size(page: Whole<Page>) -> usize {
        let bsize = page.xblock_size.get();
        internal_assert!(bsize > 0);
        if likely(bsize <= LARGE_OBJ_SIZE_MAX as u32) {
            bsize as usize
        } else {
            Segment::page_start(Page::segment(page), page, bsize as usize, &mut 0).1
        }
    }

    #[inline]
    pub unsafe fn immediately_available(&self) -> bool {
        !self.free_blocks.is_empty()
    }

    // are there any available blocks?
    #[inline]
    pub unsafe fn any_available(&self) -> bool {
        internal_assert!(self.reserved.get() > 0);
        self.used.get() < self.reserved.get() as u32
            || !DelayedFree::block(self.remote_free_blocks.load(Ordering::Relaxed)).is_null()
    }

    #[inline]
    pub unsafe fn all_free(&self) -> bool {
        self.used.get() == 0
    }

    // Retire a page with no more used blocks
    // Important to not retire too quickly though as new
    // allocations might coming.
    // Note: called from `mi_free` and benchmarks often
    // trigger this due to freeing everything and then
    // allocating again so careful when changing this.
    pub unsafe fn retire(page: Whole<Page>) {
        //  mi_assert_expensive(_mi_page_is_valid(page));
        internal_assert!(page.all_free());

        (*page).remove_flags(PageFlags::HAS_ALIGNED);

        // don't retire too often..
        // (or we end up retiring and re-allocating most of the time)
        // NOTE: refine this more: we should not retire if this
        // is the only page left with free blocks. It is not clear
        // how to check this efficiently though...
        // for now, we don't retire if it is the only page left of this size class.
        let queue = PageQueue::from_page(page);
        if likely(page.xblock_size.get() as usize <= MAX_RETIRE_SIZE && !(*queue).is_special()) {
            // not too large && not full or huge queue?
            if queue.list.only_entry(page) {
                // the only page in the queue?
                page.retire_expire
                    .set(if page.xblock_size.get() as usize <= SMALL_OBJ_SIZE_MAX {
                        RETIRE_CYCLES
                    } else {
                        RETIRE_CYCLES / 4
                    });
                let heap = (*page).local_heap();
                internal_assert!(queue.as_ptr().cast_const() >= heap.page_queues.as_ptr());
                let index = queue.as_ptr().offset_from(heap.page_queues.as_ptr()) as usize;
                internal_assert!(index < BINS);
                if index < heap.page_retired_min.get() {
                    heap.page_retired_min.set(index);
                }
                if index > heap.page_retired_max.get() {
                    heap.page_retired_max.set(index);
                }
                internal_assert!(page.all_free());
                return; // dont't free after all
            }
        }

        Page::free(page, queue, false);
    }

    pub unsafe fn validate_init(page: Whole<Page>) {
        if !cfg!(debug_assertions) {
            return;
        }

        internal_assert!(page.block_size() > 0);
        internal_assert!(page.used.get() <= page.capacity.get() as u32);
        internal_assert!(page.capacity.get() <= page.reserved.get());

        let segment = Page::segment(page);
        let _ = Page::page_start(page, segment);
        //internal_assert!(start + page.capacity*page.block_size == page.top);

        page.free_blocks.validate(page);
        page.local_deferred_free_blocks.validate(page);

        /*
          #if MI_DEBUG>3 // generally too expensive to check this
          if (page.free_is_zero) {
            const size_t ubsize = mi_page_usable_block_size(page);
            for(mi_block_t* block = page.free; block != NULL; block = mi_block_next(page,block)) {
              mi_assert_expensive(mi_mem_is_zero(block + 1, ubsize - sizeof(mi_block_t)));
            }
          }
          #endif
        */
        /*
          #if !MI_TRACK_ENABLED && !MI_TSAN
          mi_block_t* tfree = mi_page_thread_free(page);
          internal_assert!(mi_page_list_is_valid(page, tfree));
          //size_t tfree_count = mi_page_list_count(page, tfree);
          //internal_assert!(tfree_count <= page.thread_freed + 1);
          #endif
        */

        let free_count = page.free_blocks.count() + page.local_deferred_free_blocks.count();
        internal_assert!(page.used.get() as usize + free_count == page.capacity.get() as usize);
    }

    pub unsafe fn validate(page: Whole<Page>) {
        if !cfg!(debug_assertions) {
            return;
        }

        Page::validate_init(page);

        if !page.heap.load(Ordering::Relaxed).is_null() {
            let heap = page.local_heap();
            let segment = Page::segment(page);
            internal_assert!(
                segment.thread_id.load(Ordering::Relaxed) == thread_id()
                    || segment.thread_id.load(Ordering::Relaxed) == 0
            );

            let queue = PageQueue::from_page(page);
            internal_assert!(queue.list.contains(page, Page::node));
            let block_size = Page::actual_block_size(page);
            internal_assert!(
                queue.block_size == block_size
                    || block_size > LARGE_OBJ_SIZE_MAX
                    || page.flags().contains(PageFlags::IN_FULL)
            );
            internal_assert!(Heap::contains_queue(heap, queue));
        }
    }

    pub unsafe fn free(page: Whole<Page>, queue: Ptr<PageQueue>, force: bool) {
        // no more aligned blocks in here
        (*page).remove_flags(PageFlags::HAS_ALIGNED);

        // remove from the page list
        // (no need to do _mi_heap_delayed_free first as all blocks are already free)
        let heap = page.local_heap();
        let segment_data = &mut heap.thread_data().segment;
        PageQueue::remove(queue, page);

        // and free it
        (*page).set_heap(None);
        Segment::page_free(page, force, segment_data);
    }

    // Collect the local `thread_free` list using an atomic exchange.
    // Note: The exchange must be done atomically as this is used right after
    // moving to the full list in `mi_page_collect_ex` and we need to
    // ensure that there was no race where the page became unfull just before the move.
    unsafe fn thread_free_collect(page: Whole<Page>) {
        let mut head;
        let mut tfree = page.remote_free_blocks.load(Ordering::Relaxed);
        loop {
            head = DelayedFree::block(tfree);
            let tfreex = DelayedFree::change_block(tfree, null_mut());
            if compare_exchange_weak_acq_rel(&page.remote_free_blocks, &mut tfree, tfreex) {
                break;
            }
        }

        // return if the list is empty
        let Some(head) = Whole::new(head) else { return };

        // find the tail -- also to get a proper count (without data races)
        let max_count = page.capacity.get(); // cannot collect more than capacity
        let mut count = 1;
        let mut tail = head;
        let mut next;
        loop {
            next = tail.next.get();
            if let Some(next) = next {
                if count <= max_count {
                } else {
                    break;
                }
                count += 1;
                tail = next;
            } else {
                break;
            }
        }

        internal_assert!(count <= max_count, "corrupted thread-free list");

        // FIXME: Remove branch?
        // if `count > max_count` there was a memory corruption (possibly infinite list due to double multi-threaded free)
        if count > max_count {
            return; // the thread-free items cannot be freed
        }

        // and append the current local free list
        page.local_deferred_free_blocks.push_range_front(head, tail);

        // update counts now
        page.used.set(page.used.get() - count as u32);
    }

    pub unsafe fn free_collect(page: Whole<Page>, force: bool) {
        // collect the thread free list
        if force || !DelayedFree::block(page.remote_free_blocks.load(Ordering::Relaxed)).is_null() {
            // quick test to avoid an atomic operation
            Page::thread_free_collect(page);
        }

        // and the local free list
        if !page.local_deferred_free_blocks.is_empty() {
            if likely(page.free_blocks.is_empty()) {
                // usual case
                page.free_blocks.first.set(
                    page.local_deferred_free_blocks
                        .take()
                        .unwrap_unchecked()
                        .first
                        .get(),
                );
                //  (*page).free_is_zero = false;
            } else if force {
                // append -- only on shutdown (force) as this is a linear operation
                page.free_blocks
                    .extend_back(page.local_deferred_free_blocks.take().unwrap_unchecked());
                //  (*page).free_is_zero = false;
            }
        }

        internal_assert!(!force || page.local_deferred_free_blocks.is_empty());
    }

    // return true if successful
    pub unsafe fn free_delayed_block(block: Whole<FreeBlock>) -> bool {
        // get segment and page
        let segment = Segment::from_pointer(block.cast());
        internal_assert!(thread_id() == segment.thread_id.load(Ordering::Relaxed));
        let page = Segment::page_from_pointer(segment, block.cast());

        // Clear the no-delayed flag so delayed freeing is used again for this page.
        // This must be done before collecting the free lists on this page -- otherwise
        // some blocks may end up in the page `thread_free` list with no blocks in the
        // heap `thread_delayed_free` list which may cause the page to be never freed!
        // (it would only be freed if we happen to scan it in `mi_page_queue_find_free_ex`)
        if !Page::try_use_delayed_free(
            page,
            DelayedMode::UseDelayedFree,
            false, /* dont overwrite never delayed */
        ) {
            return false;
        }

        // collect all other non-local frees to ensure up-to-date `used` count
        Page::free_collect(page, false);

        // and free the block (possibly freeing the page as well since used is updated)
        Page::free_block(page, true, block);
        true
    }

    pub unsafe fn use_delayed_free(page: Whole<Page>, delay: DelayedMode, override_never: bool) {
        while !Page::try_use_delayed_free(page, delay, override_never) {
            // This ensures the value is set.
            yield_now();
        }
    }

    unsafe fn try_use_delayed_free(
        page: Whole<Page>,
        delay: DelayedMode,
        override_never: bool,
    ) -> bool {
        let mut tfree;
        let mut yield_count = 0;
        loop {
            tfree = page.remote_free_blocks.load(Ordering::Acquire); // note: must acquire as we can break/repeat this loop and not do a CAS;
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
            } else if !override_never && old_delay == DelayedMode::NeverDelayedFree {
                break; // leave never-delayed flag set
            }

            if (old_delay != DelayedMode::DelayedFreeing)
                && compare_exchange_weak_release(&page.remote_free_blocks, &mut tfree, tfreex)
            {
                break;
            }
        }

        true // success
    }

    #[inline]
    pub unsafe fn free_non_local_block(page: Whole<Page>, block: Whole<FreeBlock>) {
        // first see if the segment was abandoned and we can reclaim it
        let segment = Page::segment(page);
        /*
        let option_abandoned_reclaim_on_free = false;
                if option_abandoned_reclaim_on_free && (*segment).thread_id.load(Ordering::Relaxed) == 0 {
                    // the segment is abandoned, try to reclaim it into our heap
                    if (_mi_segment_attempt_reclaim(mi_heap_get_default(), segment)) {
                        internal_assert!(Segment::is_local(segment));
                        free(block); // recursively free as now it will be a local free in our heap
                        return;
                    }
                }
        */

        if segment.page_kind == PageKind::Huge {
            // huge pages are special as they occupy the entire segment
            // as these are large we reset the memory occupied by the page so it is available to other threads
            // (as the owning thread needs to actually free the memory later).
            Segment::huge_page_reset(segment, page, block); // FIXME: Can we not free immediately?
        }

        // Try to put the block on either the page-local thread free list, or the heap delayed free list.
        let mut use_delayed;
        let mut tfree = page.remote_free_blocks.load(Ordering::Relaxed);
        loop {
            use_delayed = DelayedFree::mode(tfree) == DelayedMode::UseDelayedFree;
            let tfreex = if unlikely(use_delayed) {
                // unlikely: this only happens on the first concurrent free in a page that is in the full list
                DelayedFree::change_mode(tfree, DelayedMode::DelayedFreeing)
            } else {
                // usual: directly add to page thread_free list
                block.next.set(Whole::new(DelayedFree::block(tfree)));
                DelayedFree::new(block.as_ptr(), DelayedFree::mode(tfree))
            };

            if compare_exchange_weak_release(&page.remote_free_blocks, &mut tfree, tfreex) {
                break;
            }
        }

        // Note: use_delayed = true seems to mean that the thread owning page cannot terminate / free the heap data
        // as we can read it here, so this is effectively a blocking operation.
        // Could this be fixed by having both the segment / page and thread own the heap via a ref count?

        if unlikely(use_delayed) {
            // racy read on `heap`, but ok because DelayedMode::DelayedFreeing is set (see `mi_heap_delete` and `mi_heap_collect_abandon`)
            let heap = page.heap.load(Ordering::Acquire); //mi_page_heap(page);
            internal_assert!(!heap.is_null());
            // FIXME: Is this sometimes NULL?
            if !heap.is_null() {
                // add to the delayed free list of this heap. (do this atomically as the lock only protects heap memory validity)
                let mut dfree = (*heap).thread_delayed_free.load(Ordering::Relaxed);
                loop {
                    block.next.set(Whole::new(dfree));

                    if compare_exchange_weak_release(
                        &(*heap).thread_delayed_free,
                        &mut dfree,
                        block.as_ptr(),
                    ) {
                        break;
                    }
                }
            }

            // and reset the DelayedMode::DelayedFreeing flag
            let mut tfree = page.remote_free_blocks.load(Ordering::Relaxed);
            loop {
                internal_assert!(DelayedFree::mode(tfree) == DelayedMode::DelayedFreeing);

                let tfreex = DelayedFree::change_mode(tfree, DelayedMode::NoDelayedFree);
                if compare_exchange_weak_release(&page.remote_free_blocks, &mut tfree, tfreex) {
                    break;
                }
            }
        }
    }

    #[inline]
    pub unsafe fn free_block(page: Whole<Page>, local: bool, block: Whole<FreeBlock>) {
        // and push it on the free list
        if likely(local) {
            page.local_deferred_free_blocks.push_front(block);
            page.used.set(page.used.get() - 1);
            if unlikely(page.all_free()) {
                Page::retire(page);
            } else if unlikely((*page).flags().contains(PageFlags::IN_FULL)) {
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
    pub unsafe fn abandon(page: Whole<Page>, queue: Ptr<PageQueue>) {
        // mi_assert_expensive(_mi_page_is_valid(page));

        let heap = (*page).local_heap();

        // remove from our page list
        PageQueue::remove(queue, page);

        // page is no longer associated with our heap
        internal_assert!((*page).thread_free_flag() == DelayedMode::NeverDelayedFree);
        (*page).set_heap(None);

        /*
          #if (MI_DEBUG>1) && !MI_TRACK_ENABLED
          if cfg!(debug_assertions) {
            // check there are no references left..
            for (mi_block_t* block = (mi_block_t*)(*heap).thread_delayed_free; block != NULL; block = mi_block_nextx(pheap, block, pheap->keys)) {
              internal_assert!(_mi_ptr_page(block) != page);
            }
        }
          #endif
          */
        // and abandon it
        Segment::page_abandon(page, &mut (*heap).thread_data().segment);
    }

    /// called from segments when reclaiming abandoned pages
    #[inline]
    pub unsafe fn reclaim(page: Whole<Page>, heap: Ptr<Heap>) {
        // mi_assert_expensive(mi_page_is_valid_init(page));
        page.assert_heap(Some(heap));
        internal_assert!(page.thread_free_flag() != DelayedMode::NeverDelayedFree);

        // TODO: push on full queue immediately if it is full?
        let queue = Heap::page_queue_for_size(heap, Page::actual_block_size(page));
        PageQueue::add(queue, page, heap);
        //  mi_assert_expensive(_mi_page_is_valid(page));
    }

    #[inline]
    pub unsafe fn to_full(page: Whole<Page>, queue: Ptr<PageQueue>) {
        internal_assert!(queue == PageQueue::from_page(page));
        internal_assert!(!page.immediately_available());
        internal_assert!(!(*page).flags().contains(PageFlags::IN_FULL));

        PageQueue::enqueue_from(
            Heap::page_queue((*page).local_heap(), BIN_FULL),
            queue,
            page,
        );
        Page::free_collect(page, false); // try to collect right away in case another thread freed just before USE_DELAYED_FREE was set
    }

    // Move a page from the full list back to a regular list
    #[inline]
    pub unsafe fn unfull(page: Whole<Page>) {
        // mi_assert_expensive(_mi_page_is_valid(page));
        internal_assert!((*page).flags().contains(PageFlags::IN_FULL));

        // Seems redundant:
        //  if (!(*page).flags().contains(PageFlags::IN_FULL))  {return;}

        // FIXME; Is this unwrap correct?
        let heap = (*page).local_heap();

        let pqfull = Heap::page_queue(heap, BIN_FULL);

        let bin = bin_index(page.block_size());
        internal_assert!(bin < BIN_FULL);
        let queue = Heap::page_queue(heap, bin);
        internal_assert!(page.block_size() == queue.block_size);

        PageQueue::enqueue_from(queue, pqfull, page);
    }

    // Start of the page available memory; can be used on uninitialized pages (only `segment_idx` must be set)
    unsafe fn page_start(page: Whole<Page>, segment: Whole<Segment>) -> (Whole<u8>, usize) {
        Segment::page_start(segment, page, page.xblock_size.get() as usize, &mut 0)
    }

    unsafe fn init(page: Whole<Page>, heap: Ptr<Heap>, block_size: usize) {
        let segment = Page::segment(page);
        internal_assert!(block_size > 0);
        // set fields
        (*page).set_heap(Some(heap));
        let (_, page_size) = Segment::page_start(segment, page, block_size, &mut 0);

        page.xblock_size
            .set(block_size.try_into().unwrap_unchecked());

        let reserved = if block_size == BIN_HUGE_BLOCK_SIZE {
            1
        } else {
            div(page_size, NonZeroUsize::new_unchecked(block_size))
                .try_into()
                .unwrap_unchecked()
        };
        page.reserved.set(reserved);

        internal_assert!(page.reserved.get() > 0);
        // (*page).free_is_zero = (*page).is_zero_init;

        internal_assert!(page.capacity.get() == 0);
        internal_assert!(page.free_blocks.is_empty());
        internal_assert!(page.used.get() == 0);
        //  internal_assert!((*page).xthread_free == 0);
        internal_assert!(page.node.next.get().is_none());
        internal_assert!(page.node.prev.get().is_none());
        internal_assert!(page.retire_expire.get() == 0);
        internal_assert!(!(*page).flags().contains(PageFlags::HAS_ALIGNED));
        //mi_assert_expensive(mi_page_is_valid_init(page));

        // initialize an initial free list
        Page::extend_free(page);
        //mi_assert(mi_page_immediate_available(page));
    }

    // allocate a fresh page from a segment
    pub unsafe fn fresh_alloc(
        heap: Ptr<Heap>,
        queue: Ptr<PageQueue>,
        block_size: usize,
        page_alignment: usize,
    ) -> Option<Whole<Page>> {
        //  internal_assert!(mi_heap_contains_queue(heap, queue));
        internal_assert!(
            page_alignment > 0 || block_size > LARGE_OBJ_SIZE_MAX || block_size == queue.block_size
        );

        let page = Segment::page_alloc(
            heap,
            block_size,
            page_alignment,
            &mut (*heap).thread_data().segment,
        )?;

        internal_assert!((Page::segment(page).page_kind == PageKind::Huge) == (*queue).is_huge());

        // a fresh page was found, initialize it
        let full_block_size = if (*queue).is_huge() {
            BIN_HUGE_BLOCK_SIZE
        } else {
            internal_assert!(block_size <= LARGE_OBJ_SIZE_MAX);
            block_size
        };
        Page::init(page, heap, full_block_size);
        internal_assert!(Page::actual_block_size(page) >= block_size);
        //  mi_heap_stat_increase(heap, pages, 1);

        PageQueue::add(queue, page, heap);

        //  mi_assert_expensive(_mi_page_is_valid(page));
        Some(page)
    }

    // Get a fresh page to use
    unsafe fn fresh(heap: Ptr<Heap>, queue: Ptr<PageQueue>) -> Option<Whole<Page>> {
        // internal_assert!(mi_heap_contains_queue(heap, pq));
        let page = Page::fresh_alloc(heap, queue, queue.block_size, 0)?;

        internal_assert!(queue.block_size == Page::actual_block_size(page));
        internal_assert!(queue == Heap::page_queue_for_size(heap, Page::actual_block_size(page)));
        Some(page)
    }

    #[inline]
    pub unsafe fn alloc_free(page: WholeOrStatic<Page>) -> Option<Whole<AllocatedBlock>> {
        // V
        page.free_blocks.pop_front().map(|block| {
            page.used.set(page.used.get().wrapping_add(1));
            block.cast()
        })
    }

    #[inline]
    pub unsafe fn alloc_small(
        page: WholeOrStatic<Page>,
        layout: Layout,
        heap: Ptr<Heap>,
    ) -> Option<Whole<AllocatedBlock>> {
        // V
        internal_assert!(layout.align() <= WORD_SIZE);
        Page::alloc_free(page).or_else(|| Heap::alloc_generic(heap, layout))
    }

    // Index a block in a page
    unsafe fn block_at(
        page: Whole<Page>,
        page_start: Whole<u8>,
        block_size: usize,
        i: usize,
    ) -> Whole<FreeBlock> {
        internal_assert!(i <= page.reserved.get() as usize);
        Whole::new_unchecked(page_start.as_ptr().byte_add(i * block_size).cast())
    }

    unsafe fn free_list_extend(page: Whole<Page>, bsize: usize, extend: usize) {
        let (page_area, _) = Page::page_start(page, Page::segment(page));

        let start = Page::block_at(page, page_area, bsize, page.capacity.get() as usize);

        // initialize a sequential free list
        let last = Page::block_at(
            page,
            page_area,
            bsize,
            page.capacity.get() as usize + extend - 1,
        );
        let mut block = start;
        while block.as_ptr() <= last.as_ptr() {
            let next = Whole::new_unchecked(block.as_ptr().byte_add(bsize));
            block.next.set(Some(next));
            block = next;
        }
        // prepend to free list (usually `null_mut()`)
        page.free_blocks.push_range_front(start, last);
    }

    /*
    unsafe fn free_list_extend_secure(page: *mut Page, heap: Ptr<Heap>,   bsize: usize,  extend: usize,) {
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
    pub unsafe fn extend_free(page: Whole<Page>) {
        if page.capacity >= page.reserved {
            return;
        }

        let (_, page_size) = Page::page_start(page, Page::segment(page));

        // calculate the extend count
        let bsize = if page.xblock_size.get() as usize <= LARGE_OBJ_SIZE_MAX {
            page.xblock_size.get() as usize
        } else {
            page_size
        };
        let mut extend = (page.reserved.get() - page.capacity.get()) as usize;

        let mut max_extend = if bsize >= MAX_EXTEND_SIZE {
            MIN_EXTEND
        } else {
            internal_assert!(bsize <= LARGE_OBJ_SIZE_MAX);
            // FIXME: Encode bsize as NonZeroUsize
            div(MAX_EXTEND_SIZE, NonZeroUsize::new_unchecked(bsize))
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
        page.capacity.set(page.capacity.get() + extend as u16);
    }
}
