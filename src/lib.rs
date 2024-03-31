#![feature(core_intrinsics, const_refs_to_static, lazy_cell)]
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
use core::fmt;
use core::{
    alloc::{GlobalAlloc, Layout},
    mem,
};
use page::AllocatedBlock;
#[cfg(debug_assertions)]
use segment::cookie;
use sptr::Strict;
use std::alloc::System;
use std::cell::Cell;
use std::cmp::max;
use std::cmp::min;
use std::fmt::Debug;
use std::hint::unreachable_unchecked;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::ops::{Deref, Range};
use std::panic::Location;
use std::ptr::{null_mut, NonNull};
use std::sync::atomic::{AtomicPtr, Ordering};
use std::thread_local;

#[allow(unused)]
macro_rules! internal_abort {
    () => (
        abort!("explicit abort");
    );
    ($($arg:tt)+) => ({
        crate::abort_fmt(format_args!($($arg)+));
    });
}

#[allow(unused)]
macro_rules! internal_assert_or_abort {
    ($cond:expr) => ({
        if !$cond {
            internal_abort!("assertion failed: {}", stringify!($cond));
        }
    });
    ($cond:expr, $($arg:tt)+) => ({
        if !$cond {
            internal_abort!("assertion failed: {}\nmessage:{}", stringify!($cond), format_args!($($arg)+));
        }
    });
}

macro_rules! expensive_assert {
    ($($arg:tt)*) => {
        internal_assert!($($arg)*)
    }
}

#[cfg(debug_assertions)]
macro_rules! internal_assert {
    ($($arg:tt)*) => {
        internal_assert_or_abort!($($arg)*)
    }
}

#[cfg(not(debug_assertions))]
macro_rules! internal_assert {
    ($cond:expr) => {{
        if false {
            let _: bool = $cond;
        }
    }};
    ($cond:expr, $($arg:tt)+) => {{
        if false {
            let _: bool = $cond;
        }
    }};
}

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

#[cold]
#[inline(never)]
#[track_caller]
#[allow(unused)]
fn abort_fmt(msg: fmt::Arguments<'_>) -> ! {
    let location = Location::caller();
    eprintln!("\nallocator aborted at {location}:\n{msg}");
    std::process::abort()
}

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

#[allow(unused)]
fn overlaps(a: Range<usize>, b: Range<usize>) -> bool {
    if a.start <= a.end || b.start <= b.end {
        return false;
    };
    max(a.start, b.start) <= min(a.end - 1, b.end - 1)
}

#[inline(always)]
fn align_down(val: usize, align: usize) -> usize {
    internal_assert!(align.is_power_of_two());
    val & !(align - 1)
}

#[inline(always)]
fn align_up(val: usize, align: usize) -> usize {
    internal_assert!(align.is_power_of_two());
    (val + align - 1) & !(align - 1)
}

#[allow(unused)]
#[inline(always)]
fn wrapping_align_up(val: usize, align: usize) -> usize {
    internal_assert!(align.is_power_of_two());
    (val.wrapping_add(align - 1)) & !(align - 1)
}

#[inline(always)]
#[cfg(debug_assertions)]
fn checked_align_up(val: usize, align: usize) -> Option<usize> {
    internal_assert!(align.is_power_of_two());
    Some((val.checked_add(align - 1)?) & !(align - 1))
}

#[inline(always)]
fn validate_align<T>(ptr: *mut T, align: usize) {
    internal_assert!(align.is_power_of_two());
    internal_assert!(ptr.addr() & (align - 1) == 0);
}

/// Returns the number of words in `size` rounded up.
#[inline(always)]
fn word_count(size: usize) -> usize {
    align_up(size, WORD_SIZE) / WORD_SIZE
}

#[inline(always)]
unsafe fn index_array<T, const S: usize>(array: *const [T; S], index: usize) -> *const T {
    internal_assert!(index < S);
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
    static IN_HEAP: Cell<bool> = const { Cell::new(false) };
}

fn with_heap<R>(f: impl FnOnce(Ptr<Heap>) -> R) -> R {
    if cfg!(debug_assertions) {
        // FIXME: std's stack overflow signal handler can invoke the allocator
        // which means that stack overflows in the allocator itself will trigger this.
        internal_assert!(!IN_HEAP.get());
        IN_HEAP.set(true);
    }
    let result = LOCAL_HEAP.with(|heap| f(unsafe { Ptr::new_unchecked(heap.get()) }));
    if cfg!(debug_assertions) {
        IN_HEAP.set(false);
    }
    result
}

/// A thread id which may be reused when the thread exits. This must be non-zero.
#[inline]
fn thread_id() -> usize {
    LOCAL_HEAP.with(|heap| heap.get().addr())
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

const PADDING: usize = 0;
// FIXME
//const PADDING: usize = WORD_SIZE;

#[inline]
#[cfg(debug_assertions)]
unsafe fn alloc_padded_end(heap: Ptr<Heap>, layout: Layout) -> Option<Whole<AllocatedBlock>> {
    use std::ptr;

    let result = Heap::alloc(
        heap,
        Layout::from_size_align_unchecked(layout.size().checked_add(PADDING)?, layout.align()),
    );
    result.inspect(|&allocation| {
        let size = Heap::usable_size(allocation);
        let ptr: *mut u8 = allocation.as_ptr().cast();
        let cookie = cookie(allocation);

        if PADDING > mem::size_of::<usize>() {
            ptr::copy_nonoverlapping(
                cookie.to_ne_bytes().as_ptr(),
                ptr.add(size - PADDING),
                mem::size_of::<usize>(),
            );
        }
    })
}

#[inline]
#[cfg(not(debug_assertions))]
unsafe fn alloc_padded_end(heap: Ptr<Heap>, layout: Layout) -> Option<Whole<AllocatedBlock>> {
    Heap::alloc(heap, layout)
}

#[inline]
#[cfg(debug_assertions)]
unsafe fn free_padded_end(allocation: Whole<AllocatedBlock>) {
    let size = Heap::usable_size(allocation);
    let ptr: *mut u8 = allocation.as_ptr().cast();
    let cookie = cookie(allocation);

    if PADDING > mem::size_of::<usize>() {
        internal_assert!(usize::from_ne_bytes(*ptr.add(size - PADDING).cast()) == cookie);
    }
    Heap::free(allocation)
}

#[inline]
#[cfg(not(debug_assertions))]
unsafe fn free_padded_end(ptr: Whole<AllocatedBlock>) {
    Heap::free(ptr)
}

#[inline]
#[cfg(debug_assertions)]
unsafe fn alloc_padded(heap: Ptr<Heap>, layout: Layout) -> Option<Whole<AllocatedBlock>> {
    let padding = max(PADDING, layout.align());
    let size = checked_align_up(layout.size(), padding)?.checked_add(padding.checked_mul(2)?)?;

    let result = Heap::alloc(
        heap,
        Layout::from_size_align_unchecked(size, layout.align()),
    );
    result.map(|allocation| {
        let ptr: *mut usize = allocation.as_ptr().cast();
        let cookie = cookie(allocation);
        if PADDING > mem::size_of::<usize>() {
            *ptr = cookie;
            let size = Heap::usable_size(allocation);
            *ptr.byte_add(size - PADDING) = cookie;
        }

        // Offset the allocation past padding
        Ptr::new_unchecked(allocation.as_ptr().byte_add(padding))
    })
}

#[inline]
#[cfg(not(debug_assertions))]
unsafe fn alloc_padded(heap: Ptr<Heap>, layout: Layout) -> Option<Whole<AllocatedBlock>> {
    Heap::alloc(heap, layout)
}

#[inline]
#[cfg(debug_assertions)]
unsafe fn free_padded(ptr: Whole<AllocatedBlock>, layout: Layout) {
    let padding = max(PADDING, layout.align());
    let allocation = Ptr::new_unchecked(ptr.as_ptr().byte_sub(padding));
    let size = Heap::usable_size(allocation);
    let ptr: *mut usize = allocation.as_ptr().cast();
    let cookie = cookie(allocation);
    if PADDING > mem::size_of::<usize>() {
        internal_assert!(*ptr == cookie);
        internal_assert!(*ptr.byte_add(size - PADDING) == cookie);
    }
    Heap::free(allocation)
}

#[inline]
#[cfg(not(debug_assertions))]
unsafe fn free_padded(ptr: Whole<AllocatedBlock>, _layout: Layout) {
    Heap::free(ptr)
}

#[inline]
pub unsafe fn alloc(layout: Layout) -> *mut u8 {
    internal_assert!(layout.size() > 0);
    internal_assert!(layout.align() > 0);
    internal_assert!(layout.align().is_power_of_two());

    let result = with_heap(|heap| Whole::as_maybe_null_ptr(alloc_padded(heap, layout)));
    // `result` may be null due to oom or if the heap
    // is abandoned and later TLS destructors allocate.
    validate_align(result, layout.align());
    result.cast()
}

#[inline]
pub unsafe fn dealloc(ptr: *mut u8, layout: Layout) {
    internal_assert!(!ptr.is_null());
    validate_align(ptr, layout.align());
    internal_assert!(layout.size() > 0);

    free_padded(Whole::new_unchecked(ptr.cast()), layout)
}

unsafe impl GlobalAlloc for Alloc {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if true {
            abort_on_panic(|| alloc(layout))
        } else {
            System.alloc(layout)
        }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if true {
            abort_on_panic(|| dealloc(ptr, layout));
        } else {
            System.dealloc(ptr, layout)
        }
    }
}
