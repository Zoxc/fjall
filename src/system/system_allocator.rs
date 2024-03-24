use crate::Ptr;
use core::alloc::GlobalAlloc;
use std::alloc::{Layout, System};

#[derive(Clone, Copy, Debug)]
pub struct SystemAllocation;

pub fn page_size() -> usize {
    1
}

pub unsafe fn commit(_ptr: Ptr<u8>, _size: usize) -> bool {
    true
}

pub unsafe fn decommit(_ptr: Ptr<u8>, _size: usize) -> bool {
    true
}

pub fn alloc(layout: Layout, _commit: bool) -> Option<(SystemAllocation, Ptr<u8>, bool)> {
    unsafe {
        Some((
            SystemAllocation,
            Ptr::new(System.alloc_zeroed(layout))?,
            true,
        ))
    }
}

pub unsafe fn dealloc(_alloc: SystemAllocation, ptr: Ptr<u8>, layout: Layout) {
    System.dealloc(ptr.as_ptr(), layout);
}
