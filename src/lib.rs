#![feature(
    core_intrinsics,
    const_refs_to_static,
    pointer_is_aligned,
    allocator_api
)]
#![allow(
    unstable_name_collisions,
    internal_features,
    clippy::missing_safety_doc,
    clippy::assertions_on_constants,
    clippy::comparison_chain
)]

use crate::heap::Heap;
use crate::page::{LARGE_PAGE_SIZE, MEDIUM_PAGE_SIZE, SMALL_PAGE_SIZE};
use crate::segment::Whole;
use core::cell::UnsafeCell;
use core::{
    alloc::{GlobalAlloc, Layout},
    mem,
};
use sptr::Strict;
use std::fmt::Debug;
use std::hint::unreachable_unchecked;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::ops::Deref;
use std::ptr::{null_mut, NonNull};
use std::sync::atomic::{AtomicPtr, Ordering};
use std::thread_local;

pub mod c;
mod heap;
mod linked_list;
mod page;
mod segment;
mod system;
mod tests;

// A word is the smallest allocation.
const WORD_SIZE: usize = if mem::size_of::<*mut ()>() > 16 {
    mem::size_of::<*mut ()>()
} else {
    16
};

const SMALL_ALLOC_WORDS: usize = 128;
const SMALL_ALLOC: usize = SMALL_ALLOC_WORDS * WORD_SIZE;

const BINS: usize = 61;
const BIN_HUGE: usize = BINS;
const BIN_FULL: usize = BINS + 1;

const BIN_FULL_BLOCK_SIZE: usize = LARGE_OBJ_SIZE_MAX + 2 * WORD_SIZE;
const BIN_HUGE_BLOCK_SIZE: usize = LARGE_OBJ_SIZE_MAX + WORD_SIZE;

const _: () = {
    assert!(BIN_HUGE_BLOCK_SIZE > LARGE_OBJ_SIZE_MAX);
};

const MAX_EXTEND_SIZE: usize = 4 * 1024; // heuristic, one OS page seems to work well.

const MIN_EXTEND: usize = 1;

/// The largest alignment which does not create dedicated huge pages.
const MEDIUM_ALIGN_MAX: usize = 0x10000;
/// The largest size for which we expand the size to align rather than create a huge page.
const MEDIUM_ALIGN_MAX_SIZE: usize = LARGE_OBJ_SIZE_MAX - (MEDIUM_ALIGN_MAX - 1);

// The max object size are checked to not waste more than 12.5% internally over the page sizes.
// (Except for large pages since huge objects are allocated in 4MiB chunks)
const SMALL_OBJ_SIZE_MAX: usize = SMALL_PAGE_SIZE / 4; // 16KiB
const MEDIUM_OBJ_SIZE_MAX: usize = MEDIUM_PAGE_SIZE / 4; // 128KiB
const LARGE_OBJ_SIZE_MAX: usize = LARGE_PAGE_SIZE / 2; // 2MiB
const LARGE_OBJ_WSIZE_MAX: usize = LARGE_OBJ_SIZE_MAX / WORD_SIZE;

const SMALL_WSIZE_MAX: usize = 128;
const SMALL_SIZE_MAX: usize = SMALL_WSIZE_MAX * WORD_SIZE;

/// A raw pointer which is non null. It implements `Deref` for convenience.
/// `A` may specify additional invariants.
#[repr(transparent)]
struct Ptr<T, A = ()> {
    ptr: NonNull<T>,
    phantom: PhantomData<A>,
}

impl<T, A> Debug for Ptr<T, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.ptr.as_ptr().fmt(f)
    }
}

impl<T, A> Copy for Ptr<T, A> {}
impl<T, A> Clone for Ptr<T, A> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, A> PartialEq for Ptr<T, A> {
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
}

impl<T, A> Deref for Ptr<T, A> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { self.ptr.as_ref() }
    }
}

unsafe impl<T, A> Send for Ptr<T, A> {}

impl<T, A> Ptr<T, A> {
    pub unsafe fn new(ptr: *mut T) -> Option<Self> {
        (!ptr.is_null()).then(|| Self::new_unchecked(ptr))
    }

    pub const unsafe fn new_unchecked(ptr: *mut T) -> Self {
        Self {
            ptr: NonNull::new_unchecked(ptr),
            phantom: PhantomData,
        }
    }

    pub const fn as_ptr(self) -> *mut T {
        self.ptr.as_ptr()
    }

    pub fn as_maybe_null_ptr(this: Option<Ptr<T, A>>) -> *mut T {
        this.map(|result| result.as_ptr()).unwrap_or(null_mut())
    }

    pub fn addr(self) -> usize {
        self.ptr.as_ptr().addr()
    }

    pub const unsafe fn cast<U>(self) -> Ptr<U, A> {
        Ptr::new_unchecked(self.as_ptr() as *mut U)
    }

    pub unsafe fn map_addr(self, f: impl FnOnce(usize) -> usize) -> Self {
        Self::new_unchecked(self.as_ptr().with_addr(f(self.as_ptr().addr())))
    }
}

#[inline(always)]
const fn rem(lhs: usize, rhs: NonZeroUsize) -> usize {
    match lhs.checked_rem(rhs.get()) {
        Some(r) => r,
        None => unsafe { unreachable_unchecked() },
    }
}

#[inline(always)]
const fn div(lhs: usize, rhs: NonZeroUsize) -> usize {
    match lhs.checked_div(rhs.get()) {
        Some(r) => r,
        None => unsafe { unreachable_unchecked() },
    }
}

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
    align_up(size, WORD_SIZE) / WORD_SIZE
}

#[inline(always)]
unsafe fn index_array<T, const S: usize>(array: *const [T; S], index: usize) -> *const T {
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
        BIN_HUGE
    } else {
        let w = w - 1;
        // find the highest bit
        let b = bit_scan_reverse(w); // note: w != 0
                                     // and use the top 3 bits to determine the bin (~12.5% worst internal fragmentation).
                                     // - adjust with 3 because we use do not round the first 8 sizes
                                     //   which each get an exact bin

        let c = (w >> ((b - 2) as u32)) & 0x03;

        ((b << 2) + c) - 3
    }
}

/// "bit scan reverse": Return index of the highest bit (or PTR_BITS if `x` is zero)
const fn bit_scan_reverse(x: usize) -> usize {
    if x == 0 {
        mem::size_of::<usize>() * 8
    } else {
        mem::size_of::<usize>() * 8 - 1 - x.leading_zeros() as usize
    }
}

thread_local! {
    static LOCAL_HEAP: UnsafeCell<Heap> = const {
        UnsafeCell::new(Heap::initial())
    };
}

fn with_heap<R>(f: impl FnOnce(Ptr<Heap>) -> R) -> R {
    LOCAL_HEAP.with(|heap| f(unsafe { Ptr::new_unchecked(heap.get()) }))
}

/// A thread id which may be reused when the thread exits.
#[inline]
fn thread_id() -> usize {
    with_heap(|heap| heap.addr())
}

/*
fn is_main_thread() -> bool {
    false
}
 */

fn yield_now() {
    std::thread::yield_now();
}

pub struct Alloc;

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

struct AbortOnPanic;

impl Drop for AbortOnPanic {
    #[inline]
    fn drop(&mut self) {
        panic!("panic in allocator");
    }
}

#[inline]
fn abort_on_panic<R>(f: impl FnOnce() -> R) -> R {
    let guard = AbortOnPanic;
    let result = f();
    mem::forget(guard);
    result
}

#[inline]
pub unsafe fn alloc(layout: Layout) -> *mut u8 {
    debug_assert!(layout.size() > 0);
    debug_assert!(layout.align() > 0);
    debug_assert!(layout.align().is_power_of_two());

    let result = with_heap(|heap| Whole::as_maybe_null_ptr(Heap::alloc(heap, layout)));
    // `result` may be null due to oom or if the heap
    // is abandoned and later TLS destructors allocate.
    debug_assert!(result.is_aligned_to(layout.align()));
    result.cast()
}

#[inline]
pub unsafe fn dealloc(ptr: *mut u8, layout: Layout) {
    debug_assert!(!ptr.is_null());
    debug_assert!(ptr.is_aligned_to(layout.align()));
    debug_assert!(layout.size() > 0);

    Heap::free(Whole::new_unchecked(ptr.cast()))
}

unsafe impl GlobalAlloc for Alloc {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        abort_on_panic(|| alloc(layout))
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        abort_on_panic(|| dealloc(ptr, layout));
    }
}
