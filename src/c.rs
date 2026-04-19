use crate::{
    alloc_padded_end, free_padded_end, heap::Heap, segment::Whole, validate_align, with_heap,
    PADDING, WORD_SIZE,
};
use libc::{EINVAL, ENOMEM};
use std::{
    alloc::Layout,
    cmp::{self, max},
    ffi::c_char,
    mem,
    os::raw::{c_int, c_void},
    ptr::{self, null_mut},
};

#[inline]
unsafe fn alloc_layout(layout: Layout) -> *mut c_void {
    let result = with_heap(|heap| Whole::as_maybe_null_ptr(alloc_padded_end(heap, layout)));
    validate_align(result, layout.align());
    result.cast()
}

// Corresponds to mimalloc `mi_malloc` in `src/alloc.c:175-176` and the public `malloc` override in `src/alloc-override.c:132`.
#[inline]
pub extern "C" fn malloc(size: usize) -> *mut c_void {
    aligned_alloc(WORD_SIZE, size)
}

// Corresponds to mimalloc `mi_free` in `src/alloc.c:570-603` and the public `free` override in `src/alloc-override.c:135`.
#[inline]
pub unsafe extern "C" fn free(ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }
    validate_align(ptr, WORD_SIZE);
    free_padded_end(Whole::new_unchecked(ptr.cast()))
}

#[allow(clippy::redundant_closure)] // Not actually redundant
// Corresponds to mimalloc `mi_calloc` / `mi_heap_calloc` in `src/alloc.c:680-688` and the public `calloc` override in `src/alloc-override.c:133`.
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

// Corresponds to mimalloc `mi_posix_memalign` in `src/alloc-posix.c:55-67`.
#[inline]
pub unsafe extern "C" fn posix_memalign(ptr: *mut *mut c_void, align: usize, size: usize) -> c_int {
    if ptr.is_null() {
        return EINVAL;
    }

    if !align.is_power_of_two() || align % mem::size_of::<*mut ()>() != 0 {
        return EINVAL;
    }

    let result = aligned_alloc(align, size);
    if result.is_null() {
        ENOMEM
    } else {
        *ptr = result;
        0
    }
}

// Corresponds to mimalloc `mi_aligned_alloc` in `src/alloc-posix.c:86-99`.
#[inline]
pub extern "C" fn aligned_alloc(align: usize, size: usize) -> *mut c_void {
    unsafe {
        if !align.is_power_of_two() {
            return null_mut();
        }

        let size = max(size, 1);
        let align = max(align, WORD_SIZE);
        let Ok(layout) = Layout::from_size_align(size, align) else {
            return null_mut();
        };

        alloc_layout(layout)
    }
}

// Corresponds to mimalloc `mi_realloc` / `_mi_heap_realloc_zero` in `src/alloc.c:715-745` and `src/alloc.c:776-777`.
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

// Corresponds to mimalloc `mi_strdup` / `mi_heap_strdup` in `src/alloc.c:804-816`.
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

// Corresponds to mimalloc `mi_strndup` / `mi_heap_strndup` in `src/alloc.c:819-833`.
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
