use crate::{
    align_down, align_up,
    segment::{OPTION_PURGE_DELAY, OPTION_PURGE_DOES_DECOMMIT},
    Ptr,
};
use sptr::Strict;

#[cfg(windows)]
mod windows;
#[cfg(windows)]
pub use windows::*;

#[cfg(unix)]
mod unix;
#[cfg(unix)]
pub use unix::*;

#[cfg(feature = "system-allocator")]
mod system_allocator;
#[cfg(feature = "system-allocator")]
pub use system_allocator::*;

#[cfg(any(miri, unix))]
mod miri;
#[cfg(any(miri, unix))]
pub use miri::*;

unsafe fn page_align_conservative(ptr: *mut u8, size: usize) -> (*mut u8, usize) {
    let page_size = page_size();
    let end = ptr.add(size).map_addr(|addr| align_down(addr, page_size));
    let ptr = ptr.map_addr(|addr| align_up(addr, page_size));
    (ptr, end.offset_from(ptr) as usize)
}

// either resets or decommits memory, returns true if the memory needs
// to be recommitted if it is to be re-used later on.
pub unsafe fn purge(ptr: *mut u8, size: usize, _allow_reset: bool) -> bool {
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
            decommit(Ptr::new_unchecked(ptr), size)
        }
    } else {
        /*if (allow_reset) {  // this can sometimes be not allowed if the range is not fully committed
          _mi_os_reset(p, size, stats);
        }*/
        false // needs no recommit
    }
}

pub fn has_overcommit() -> bool {
    cfg!(target_os = "linux")
}

fn page_size() -> usize {
    0x1000
}
