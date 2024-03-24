use crate::Ptr;
#[cfg(not(feature = "system-allocator"))]
use crate::{align_down, align_up, system::page_size};
use core::mem;
#[cfg(not(feature = "system-allocator"))]
use libc::{
    c_void, MADV_DONTNEED, MAP_ANONYMOUS, MAP_FAILED, MAP_PRIVATE, PROT_NONE, PROT_READ, PROT_WRITE,
};
#[cfg(not(feature = "system-allocator"))]
use sptr::Strict;
use std::alloc::Layout;
use std::ptr::null_mut;

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
    let result = libc::mprotect(ptr.as_ptr().cast(), size, PROT_READ | PROT_WRITE);
    debug_assert!(result == 0);
    result == 0
}

#[cfg(not(feature = "system-allocator"))]
pub unsafe fn decommit(ptr: Ptr<u8>, size: usize) -> bool {
    // decommit: use MADV_DONTNEED as it decreases rss immediately (unlike MADV_FREE)
    let result = libc::madvise(ptr.as_ptr().cast(), size, MADV_DONTNEED);
    debug_assert!(result == 0);
    result == 0
}

#[cfg(not(feature = "system-allocator"))]
pub fn alloc(layout: Layout, commit: bool) -> Option<(SystemAllocation, Ptr<u8>, bool)> {
    let size = layout.size().checked_add(layout.align() - 1)?;
    let protect_flags = if commit {
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
        let page_size = page_size();
        let aligned = result.map_addr(|addr| align_up(addr, layout.align()));

        let before = align_down(aligned.addr() - result.addr(), page_size);
        if before > 0 && !cfg!(miri) {
            libc::munmap(result.with_addr(result.addr() + before).cast(), before);
        }

        let after = align_up(aligned.addr() + layout.size(), page_size);
        let len = align_down(
            result.addr().wrapping_add(size).wrapping_sub(after),
            page_size,
        );
        if len > 0 && !cfg!(miri) {
            libc::munmap(result.with_addr(after).cast(), len);
        }

        debug_assert!(aligned.is_aligned_to(layout.align()));
        Some((alloc, Ptr::new_unchecked(aligned), commit))
    }
}

#[cfg(not(feature = "system-allocator"))]
pub unsafe fn dealloc(alloc: SystemAllocation, __ptr: Ptr<u8>, layout: Layout) {
    let size = layout.size() + layout.align() - 1;
    let result = libc::munmap(alloc.base, size);
    debug_assert!(result == 0);
}
