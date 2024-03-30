use crate::heap::Heap;
use crate::with_heap;
#[cfg(not(feature = "system-allocator"))]
use crate::{align_down, align_up, validate_align, wrapped_align_up, Ptr};
use core::mem;
use libc::c_void;
#[cfg(not(feature = "system-allocator"))]
use libc::{
    MADV_DONTNEED, MAP_ANONYMOUS, MAP_FAILED, MAP_PRIVATE, PROT_NONE, PROT_READ, PROT_WRITE,
};
use sptr::Strict;
#[cfg(not(feature = "system-allocator"))]
use std::alloc::Layout;
use std::ptr::null_mut;
use std::sync::LazyLock;
#[cfg(not(feature = "system-allocator"))]
use {core::sync::atomic::AtomicUsize, core::sync::atomic::Ordering};

unsafe extern "C" fn thread_done(value: *mut c_void) {
    if !value.is_null() {
        with_heap(|heap| Heap::done(heap));
    }
}

static KEY: LazyLock<libc::pthread_key_t> = LazyLock::new(|| unsafe {
    let mut key: libc::pthread_key_t = 0;
    internal_assert_or_abort!(libc::pthread_key_create(&mut key, Some(thread_done)) == 0);
    key
});

pub fn register_thread() {
    unsafe {
        internal_assert_or_abort!(
            libc::pthread_setspecific(*KEY, null_mut::<c_void>().with_addr(1)) == 0
        );
    }
}

#[cfg(not(feature = "system-allocator"))]
pub fn page_size() -> usize {
    static PAGE_SIZE: AtomicUsize = AtomicUsize::new(0);

    let cached = PAGE_SIZE.load(Ordering::Relaxed);
    if cached != 0 {
        cached as usize
    } else {
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        internal_assert_or_abort!(page_size >= 1);
        PAGE_SIZE.store(page_size as usize, Ordering::Relaxed);
        page_size as usize
    }
}

/// A clock in milliseconds.
pub fn clock_now() -> Option<u64> {
    unsafe {
        let mut t = mem::zeroed();
        (libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut t) == 0)
            .then_some((t.tv_sec as u64) * 1000 + (t.tv_nsec as u64 / 1000000))
    }
}

#[cfg(not(feature = "system-allocator"))]
#[derive(Clone, Copy, Debug)]
pub struct SystemAllocation {
    base: *mut c_void,
}

#[cfg(not(feature = "system-allocator"))]
pub unsafe fn commit(ptr: Ptr<u8>, size: usize) -> bool {
    /*
    let result = libc::mprotect(ptr.as_ptr().cast(), size, PROT_READ | PROT_WRITE);
    internal_assert!(result == 0);
    result == 0*/
    true
}

#[cfg(not(feature = "system-allocator"))]
pub unsafe fn decommit(ptr: Ptr<u8>, size: usize) -> bool {
    /*
    // decommit: use MADV_DONTNEED as it decreases rss immediately (unlike MADV_FREE)
    let result = libc::madvise(ptr.as_ptr().cast(), size, MADV_DONTNEED);
    internal_assert!(result == 0);*/
    false
}

#[cfg(not(feature = "system-allocator"))]
pub fn alloc(layout: Layout, commit: bool) -> Option<(SystemAllocation, Ptr<u8>, bool)> {
    let size = layout.size().checked_add(layout.align() - 1)?;
    let protect_flags = if true {
        PROT_WRITE | PROT_READ
    } else {
        PROT_NONE
    };
    let flags = MAP_PRIVATE | MAP_ANONYMOUS;
    unsafe {
        let result = libc::mmap(null_mut(), size, protect_flags, flags, -1, 0);
        if result == MAP_FAILED {
            return None;
        }
        let alloc = SystemAllocation { base: result };
        let result: *mut u8 = result.cast();

        let aligned = result.map_addr(|addr| align_up(addr, layout.align()));

        if result.is_null() || aligned.addr().checked_add(layout.size()).is_none() {
            // Either 0 or the last byte of the address space was allocated, so we can't use
            // this memory.
            dealloc(alloc, Ptr::new_unchecked(aligned), layout);
            return None;
        }

        let unmap = |start: usize, end: usize| {
            let len = end.wrapping_sub(start);
            if len > 0 && !cfg!(miri) {
                //   libc::munmap(result.with_addr(start).cast(), len);
            }
        };

        let page_size = page_size();
        internal_assert!(result.addr() == align_down(result.addr(), page_size));
        unmap(result.addr(), align_down(aligned.addr(), page_size));
        unmap(
            align_up(aligned.addr().wrapping_add(layout.size()), page_size),
            wrapped_align_up(result.addr().wrapping_add(size), page_size),
        );
        validate_align(aligned, layout.align());
        Some((alloc, Ptr::new_unchecked(aligned), commit))
    }
}

#[cfg(not(feature = "system-allocator"))]
pub unsafe fn dealloc(alloc: SystemAllocation, __ptr: Ptr<u8>, layout: Layout) {
    let size = layout.size() + layout.align() - 1;
    let result = libc::munmap(alloc.base, size);
    internal_assert!(result == 0);
}
