use crate::alloc;
use crate::segment::ALIGNMENT_MAX;
use crate::{dealloc, LARGE_OBJ_SIZE_MAX};
use std::alloc::Layout;

#[test]
fn large_alloc() {
    let layout = Layout::from_size_align(LARGE_OBJ_SIZE_MAX + 1, 1).unwrap();
    unsafe {
        let huge = alloc(layout);
        dealloc(huge, layout);
    }
}

#[test]
fn medium_align() {
    let layout = Layout::from_size_align(1, ALIGNMENT_MAX).unwrap();
    unsafe {
        let huge = alloc(layout);
        dealloc(huge, layout);
    }
}
