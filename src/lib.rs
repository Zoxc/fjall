#![feature(core_intrinsics, const_refs_to_static, pointer_is_aligned)]
#![allow(unstable_name_collisions, internal_features, unused)]

use crate::heap::Heap;
use crate::page::{Page, LARGE_PAGE_SIZE, MEDIUM_PAGE_SIZE, SMALL_PAGE_SIZE};
use crate::segment::SEGMENT_SIZE;
use core::cell::UnsafeCell;
use core::{
    alloc::{GlobalAlloc, Layout},
    intrinsics::likely,
    mem,
};
use page::{Block, PageFlags};
use segment::Segment;
use sptr::Strict;
use std::intrinsics::unlikely;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::{alloc::System, thread_local};

mod heap;
mod linked_list;
mod page;
mod segment;
mod system;

const PTR_SIZE: usize = mem::size_of::<*mut ()>();
const PTR_BITS: usize = PTR_SIZE * 8;

const SMALL_ALLOC_WORDS: usize = 128;
const SMALL_ALLOC: usize = SMALL_ALLOC_WORDS * PTR_SIZE;

const BINS: usize = 69;
const BIN_FULL: usize = BINS + 1;

// (must match REGION_MAX_ALLOC_SIZE in memory.c)
const HUGE_OBJ_SIZE_MAX: usize = 2 * PTR_SIZE * SEGMENT_SIZE;

// Used as a special value to encode block sizes in 32 bits.
const HUGE_BLOCK_SIZE: u32 = HUGE_OBJ_SIZE_MAX as u32;

const MAX_EXTEND_SIZE: usize = 4 * 1024; // heuristic, one OS page seems to work well.

const MIN_EXTEND: usize = 1;

const MAX_ALIGN_SIZE: usize = 16;

const PAGE_HUGE_ALIGN: usize = 256 * 1024;

// The max object size are checked to not waste more than 12.5% internally over the page sizes.
// (Except for large pages since huge objects are allocated in 4MiB chunks)
const SMALL_OBJ_SIZE_MAX: usize = SMALL_PAGE_SIZE / 4; // 16KiB
const MEDIUM_OBJ_SIZE_MAX: usize = MEDIUM_PAGE_SIZE / 4; // 128KiB
const LARGE_OBJ_SIZE_MAX: usize = LARGE_PAGE_SIZE / 2; // 2MiB
const LARGE_OBJ_WSIZE_MAX: usize = LARGE_OBJ_SIZE_MAX / PTR_SIZE;

const MAX_SLICE_SHIFT: usize = 6; // at most 64 slices
const MAX_SLICES: usize = 1 << MAX_SLICE_SHIFT;
const MIN_SLICES: usize = 2;

const SMALL_WSIZE_MAX: usize = 128;
const SMALL_SIZE_MAX: usize = SMALL_WSIZE_MAX * PTR_SIZE;

#[inline(always)]
const fn align_down(val: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    val & !(align - 1)
}

#[inline(always)]
const fn align_up(val: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (val + align - 1) & !(align - 1)
}

/// Returns the number of words in `size` rounded up.
#[inline(always)]
const fn word_count(size: usize) -> usize {
    align_up(size, PTR_SIZE) / PTR_SIZE
}

#[inline(always)]
unsafe fn index_array<T, const S: usize>(array: *mut [T; S], index: usize) -> *mut T {
    debug_assert!(index < S);
    unsafe { array.cast::<T>().add(index) }
}

#[inline]
fn bin_index(size: usize) -> usize {
    let w = word_count(size);
    if w <= 1 {
        1
    } else if w <= 8 {
        w
    } else if w > LARGE_OBJ_WSIZE_MAX {
        BINS
    } else {
        let w = w - 1;
        // find the highest bit
        let b = bit_scan_reverse(w); // note: w != 0
                                     // and use the top 3 bits to determine the bin (~12.5% worst internal fragmentation).
                                     // - adjust with 3 because we use do not round the first 8 sizes
                                     //   which each get an exact bin

        let c = ((w >> (b.saturating_sub(2) as u32)) & 0x03);

        ((b << 2) + c).saturating_sub(3)
    }
}

/// "bit scan reverse": Return index of the highest bit (or PTR_BITS if `x` is zero)
const fn bit_scan_reverse(x: usize) -> usize {
    if x == 0 {
        PTR_BITS
    } else {
        PTR_BITS - 1 - x.leading_zeros() as usize
    }
}

thread_local! {
    static LOCAL_HEAP: UnsafeCell<Heap> = const {
        UnsafeCell::new(Heap::INITIAL)
    };
}

#[cfg(windows)]
mod thread_exit {
    use crate::{heap::Heap, with_heap};
    use core::ptr::read_volatile;
    use windows_sys::Win32::System::SystemServices::{DLL_PROCESS_DETACH, DLL_THREAD_DETACH};

    // FIXME: Apparently this trick doesn't work for loaded DLLs.
    // The callback runs before main, so we could dynamically use another solution in DLLs?

    // Use the `.CRT$XLM` section as that us executed
    // later then the `.CRT$XLB` section used by `std`.
    #[link_section = ".CRT$XLM"]
    #[used]
    static THREAD_CALLBACK: unsafe extern "system" fn(*mut (), u32, *mut ()) = callback;

    pub fn register_thread() {}

    unsafe extern "system" fn callback(_h: *mut (), dw_reason: u32, _pv: *mut ()) {
        if dw_reason == DLL_THREAD_DETACH || dw_reason == DLL_PROCESS_DETACH {
            with_heap(|heap| Heap::done(heap));
        }
    }
}

#[cfg(miri)]
mod thread_exit {
    use crate::{heap::Heap, with_heap};

    thread_local! {
        static THREAD_EXIT: ThreadExit = const { ThreadExit };
    }
    struct ThreadExit;
    impl Drop for ThreadExit {
        fn drop(&mut self) {
            unsafe {
                with_heap(|heap| Heap::done(heap));
            }
        }
    }
    pub fn register_thread() {
        THREAD_EXIT.with(|_| {});
    }
}

fn with_heap<R>(f: impl FnOnce(*mut Heap) -> R) -> R {
    LOCAL_HEAP.with(|heap| f(heap.get()))
}

#[inline]
fn thread_id() -> usize {
    // FIXME: Is this valid with reuse?
    with_heap(|heap| heap.addr())
}

fn is_main_thread() -> bool {
    false
}

fn yield_now() {
    std::thread::yield_now();
}

pub struct Alloc;

#[inline]
fn fallback_alloc(layout: Layout) -> bool {
    layout.size() > SMALL_ALLOC || layout.align() > PTR_SIZE
}

#[inline]
fn compare_exchange_weak_acq_rel<T>(
    atomic: &AtomicPtr<T>,
    current: &mut *mut T,
    new: *mut T,
) -> bool {
    *current =
        match atomic.compare_exchange_weak(*current, new, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return true,
            Err(v) => v,
        };
    false
}

#[inline]
fn compare_exchange_weak_release<T>(
    atomic: &AtomicPtr<T>,
    current: &mut *mut T,
    new: *mut T,
) -> bool {
    *current =
        match atomic.compare_exchange_weak(*current, new, Ordering::Release, Ordering::Relaxed) {
            Ok(_) => return true,
            Err(v) => v,
        };
    false
}

#[inline(never)]
unsafe fn free_generic(segment: *mut Segment, page: *mut Page, is_local: bool, ptr: *mut u8) {
    // FIXME: `flags` Can race with other threads
    let block = if Page::flags(page).contains(PageFlags::HAS_ALIGNED) {
        Page::unalign_pointer(page, segment, ptr)
    } else {
        ptr.cast()
    };

    Page::free_block(page, is_local, block);
}

#[inline]
unsafe fn free(ptr: *mut u8) {
    let segment = Segment::from_pointer_checked(ptr);
    let is_local = Segment::is_local(segment);
    let page = Segment::page_from_pointer(segment, ptr);

    if likely(is_local) {
        // thread-local free?
        if likely(Page::flags(page) == PageFlags::empty())
        // and it is not a full page (full pages need to move from the full bin), nor has aligned blocks (aligned blocks need to be unaligned)
        {
            let block: *mut Block = ptr.cast();

            // mi_check_padding(page, block);
            (*block).next = (*page).local_free;
            (*page).local_free = block;
            (*page).used -= 1;
            if unlikely((*page).used == 0) {
                Page::retire(page);
            }
        } else {
            // page is full or contains (inner) aligned blocks; use generic path
            free_generic(segment, page, true, ptr);
        }
    } else {
        // not thread-local; use generic path
        free_generic(segment, page, false, ptr);
    }
}

unsafe impl GlobalAlloc for Alloc {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if likely(!fallback_alloc(layout)) {
            let result = with_heap(|heap| Heap::alloc_small(heap, layout));
            // `result` may be null due to oom or if the heap
            // is abandoned and later TLS destructors allocate.
            debug_assert!(result.is_aligned_to(layout.align()));
            result
        } else {
            System.alloc(layout)
        }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        debug_assert!(!ptr.is_null());
        debug_assert!(ptr.is_aligned_to(layout.align()));

        if likely(!fallback_alloc(layout)) {
            free(ptr)
        } else {
            System.dealloc(ptr, layout)
        }
    }
}
