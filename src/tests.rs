#![cfg(test)]

use crate::segment::SEGMENT_ALIGN;
use crate::{
    alloc, MEDIUM_ALIGN_MAX, MEDIUM_ALIGN_MAX_SIZE, MEDIUM_OBJ_SIZE_MAX, SMALL_OBJ_SIZE_MAX,
};
use crate::{dealloc, LARGE_OBJ_SIZE_MAX};
use libc::EINVAL;
use std::alloc::Layout;
use std::cmp::{max, min};
use std::ops::Sub;
use std::ptr::null_mut;

fn test(size: usize, align: usize) {
    dbg!(size, align);
    let layout = Layout::from_size_align(size, align).unwrap();
    unsafe {
        let ptr = alloc(layout);
        assert!(!ptr.is_null());
        dealloc(ptr, layout);
    }
}

fn test_align(size: usize) {
    test(size, 1);
    test(size, MEDIUM_ALIGN_MAX);
    test(size, MEDIUM_ALIGN_MAX << 1);
    test(size, SEGMENT_ALIGN);
    // FIXME: Not yet supported
    //test(size, SEGMENT_ALIGN << 1);
}

fn test_size(size: usize) {
    test_align(max(size.sub(1), 1));
    test_align(size);
    test_align(min(size.saturating_add(1), isize::MAX as usize));
}

#[test]
fn allocs() {
    test_size(1);
    test_size(SMALL_OBJ_SIZE_MAX);
    test_size(MEDIUM_OBJ_SIZE_MAX);
    test_size(MEDIUM_ALIGN_MAX_SIZE);
    test_size(LARGE_OBJ_SIZE_MAX);
}

#[test]
fn c_api_rejects_invalid_layout_inputs() {
    unsafe {
        let mut ptr = 1usize as *mut _;

        assert_eq!(crate::c::posix_memalign(&mut ptr, 3, 64), EINVAL);
        assert_eq!(ptr, 1usize as *mut _);

        assert!(crate::c::aligned_alloc(3, 64).is_null());
        assert_eq!(crate::c::malloc(isize::MAX as usize + 1), null_mut());
    }
}
