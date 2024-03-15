use crate::{
    align_down, align_up,
    segment::{OPTION_PURGE_DELAY, OPTION_PURGE_DOES_DECOMMIT},
};
use sptr::Strict;
use std::{
    alloc::Layout,
    sync::atomic::{AtomicI64, Ordering},
    time::Instant,
};

#[cfg(any(miri, unix))]
use {core::alloc::GlobalAlloc, std::alloc::System};

#[cfg(all(windows, not(miri)))]
mod windows;
#[cfg(all(windows, not(miri)))]
pub use windows::*;

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
#[cfg(miri)]
pub use thread_exit::register_thread;

/// A clock in milliseconds.
#[cfg(unix)]
pub fn clock_now() -> Option<u64> {
    use std::mem;

    unsafe {
        let mut t = mem::zeroed();
        (libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut t) == 0)
            .then_some((t.tv_sec as u64) * 1000 + (t.tv_nsec as u64 / 1000000))
    }
}

unsafe fn page_align_conservative(ptr: *mut u8, size: usize) -> (*mut u8, usize) {
    let page_size = page_size();
    let end = ptr.add(size).map_addr(|addr| align_down(addr, page_size));
    let ptr = ptr.map_addr(|addr| align_up(addr, page_size));
    (ptr, end.offset_from(ptr) as usize)
}

// either resets or decommits memory, returns true if the memory needs
// to be recommitted if it is to be re-used later on.
pub unsafe fn purge(ptr: *mut u8, size: usize, allow_reset: bool) -> bool {
    if OPTION_PURGE_DELAY < 0 {
        return false;
    } // is purging allowed?}

    // FIXME
    if OPTION_PURGE_DOES_DECOMMIT
    /*  &&   // should decommit?
    !_mi_preloading())                                     // don't decommit during preloading (unsafe)*/
    {
        let (ptr, size) = page_align_conservative(ptr, size);
        if size == 0 {
            false
        } else {
            decommit(ptr, size)
        }
    } else {
        /*if (allow_reset) {  // this can sometimes be not allowed if the range is not fully committed
          _mi_os_reset(p, size, stats);
        }*/
        false // needs no recommit
    }
}

pub fn has_overcommit() -> bool {
    #[cfg(target_os = "linux")]
    {
        return true;
    }
    false
}

#[cfg(any(miri, unix))]
pub unsafe fn commit(ptr: *mut u8, size: usize) -> bool {
    true
}

#[cfg(any(miri, unix))]
pub unsafe fn decommit(ptr: *mut u8, size: usize) -> bool {
    true
}

#[cfg(any(miri, unix))]
pub fn alloc(layout: Layout, _commit: bool) -> (*mut u8, bool) {
    unsafe { (System.alloc_zeroed(layout), true) }
}

#[cfg(any(miri, unix))]
pub unsafe fn dealloc(ptr: *mut u8, layout: Layout) {
    System.dealloc(ptr, layout);
}

fn page_size() -> usize {
    0x1000
}
