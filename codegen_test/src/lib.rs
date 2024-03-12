use fjall::Alloc;
use std::alloc::GlobalAlloc;
use std::alloc::Layout;

#[no_mangle]
pub unsafe fn fjall_alloc(layout: Layout) -> *mut u8 {
    Alloc.alloc(layout)
}

#[no_mangle]
pub unsafe fn fjall_dealloc(ptr: *mut u8, layout: Layout) {
    Alloc.dealloc(ptr, layout)
}
