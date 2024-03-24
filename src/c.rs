use crate::{alloc, heap::Heap, segment::Whole, WORD_SIZE};
use std::{
    alloc::Layout,
    cmp::{self, min},
    ffi::c_char,
    os::raw::{c_int, c_void},
    ptr::{self, null_mut},
};

#[inline]
pub extern "C" fn malloc(size: usize) -> *mut c_void {
    unsafe { alloc(Layout::from_size_align_unchecked(size, WORD_SIZE)).cast() }
}

#[inline]
pub unsafe extern "C" fn free(ptr: *mut c_void) {
    debug_assert!(!ptr.is_null());
    debug_assert!(ptr.is_aligned_to(WORD_SIZE));

    Heap::free(Whole::new_unchecked(ptr.cast()))
}

#[allow(clippy::redundant_closure)] // Not actually redundant
#[inline]
pub unsafe extern "C" fn calloc(items: usize, size: usize) -> *mut c_void {
    items
        .checked_mul(size)
        .map(|c| malloc(c))
        .unwrap_or(null_mut())
}

#[inline]
pub unsafe extern "C" fn posix_memalign(ptr: *mut *mut c_void, align: usize, size: usize) -> c_int {
    *ptr = aligned_alloc(align, size);
    if (*ptr).is_null() {
        libc::ENOMEM
    } else {
        0
    }
}

#[inline]
pub unsafe extern "C" fn aligned_alloc(align: usize, size: usize) -> *mut c_void {
    alloc(Layout::from_size_align_unchecked(size, align)).cast()
}

#[inline]
pub unsafe extern "C" fn realloc(ptr: *mut c_void, new_size: usize) -> *mut c_void {
    let size = Heap::usable_size(Whole::new_unchecked(ptr.cast()));
    let new_layout = Layout::from_size_align_unchecked(new_size, WORD_SIZE);
    let new_ptr = alloc(new_layout);
    if !new_ptr.is_null() {
        ptr::copy_nonoverlapping(ptr.cast(), new_ptr, cmp::min(size, new_size));
        free(ptr);
    }
    new_ptr.cast()
}

#[inline]
pub unsafe extern "C" fn strdup(ptr: *const c_char) -> *mut c_char {
    if ptr.is_null() {
        return null_mut();
    }
    let n = libc::strlen(ptr);
    let new: *mut c_char = malloc(n + 1).cast();
    if new.is_null() {
        return null_mut();
    }
    ptr::copy_nonoverlapping(ptr, new, n);
    *new.add(n) = 0;
    new
}

#[inline]
pub unsafe extern "C" fn strndup(ptr: *const c_char, length: usize) -> *mut c_char {
    if ptr.is_null() {
        return null_mut();
    }
    let n = min(libc::strlen(ptr), length);
    let new: *mut c_char = malloc(n + 1).cast();
    if new.is_null() {
        return null_mut();
    }
    ptr::copy_nonoverlapping(ptr, new, n);
    *new.add(n) = 0;
    new
}
