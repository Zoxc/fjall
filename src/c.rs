use crate::{
    alloc_padded_end, free_padded_end, heap::Heap, segment::Whole, with_heap, PADDING, WORD_SIZE,
};
use libc::{EINVAL, ENOMEM};
use sptr::Strict;
use std::{
    alloc::Layout,
    cmp::{self, max},
    ffi::c_char,
    mem,
    os::raw::{c_int, c_void},
    ptr::{self, null_mut},
};

#[inline]
pub extern "C" fn malloc(size: usize) -> *mut c_void {
    aligned_alloc(WORD_SIZE, size)
}

#[inline]
pub unsafe extern "C" fn free(ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }
    internal_assert!(ptr.is_aligned_to(WORD_SIZE));
    free_padded_end(Whole::new_unchecked(ptr.cast()))
}

#[allow(clippy::redundant_closure)] // Not actually redundant
#[inline]
pub unsafe extern "C" fn calloc(items: usize, size: usize) -> *mut c_void {
    items
        .checked_mul(size)
        .map(|c| {
            let result = malloc(c);
            if !result.is_null() {
                ptr::write_bytes(result, 0, c)
            }
            result
        })
        .unwrap_or(null_mut())
}

#[inline]
pub unsafe extern "C" fn posix_memalign(ptr: *mut *mut c_void, align: usize, size: usize) -> c_int {
    if ptr.is_null() {
        return EINVAL;
    }
    internal_assert!(align % mem::size_of::<*mut ()>() == 0);
    internal_assert!(align.is_power_of_two());
    let result = aligned_alloc(align, size);
    if result.is_null() {
        ENOMEM
    } else {
        *ptr = result;
        0
    }
}

#[inline]
pub extern "C" fn aligned_alloc(align: usize, size: usize) -> *mut c_void {
    unsafe {
        internal_assert!(align.is_power_of_two());

        let size = max(size, 1);
        let align = max(align, WORD_SIZE);

        let result = with_heap(|heap| {
            Whole::as_maybe_null_ptr(alloc_padded_end(
                heap,
                Layout::from_size_align_unchecked(size, align),
            ))
        });
        internal_assert!(result.is_aligned_to(align));
        result.cast()
    }
}

#[inline]
pub unsafe extern "C" fn realloc(ptr: *mut c_void, new_size: usize) -> *mut c_void {
    if ptr.is_null() {
        return malloc(new_size);
    }
    if new_size == 0 {
        free(ptr);
        return null_mut();
    }
    let new_ptr = malloc(new_size);
    if !new_ptr.is_null() {
        let size = Heap::usable_size(Whole::new_unchecked(ptr.cast())) - PADDING;
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
    let n = libc::strlen(ptr) + 1;
    let new: *mut c_char = malloc(n).cast();
    if new.is_null() {
        return null_mut();
    }
    ptr::copy_nonoverlapping(ptr, new, n);
    new
}

#[inline]
pub unsafe extern "C" fn strndup(ptr: *const c_char, length: usize) -> *mut c_char {
    if ptr.is_null() {
        return null_mut();
    }
    let end = libc::memchr(ptr.cast(), 0, length);
    let n = if !end.is_null() {
        end.addr() - ptr.addr()
    } else {
        length
    };
    internal_assert!(n <= length);
    let new: *mut c_char = malloc(n + 1).cast();
    if new.is_null() {
        return null_mut();
    }
    ptr::copy_nonoverlapping(ptr, new, n);
    *new.add(n) = 0;
    new
}
