#[cfg(feature = "panic-check")]
use no_panic::no_panic;
use std::alloc::Layout;

#[cfg_attr(feature = "panic-check", no_panic)]
#[no_mangle]
pub unsafe fn fjall_alloc(layout: Layout) -> *mut u8 {
    fjall::alloc(layout)
}

#[cfg_attr(feature = "panic-check", no_panic)]
#[no_mangle]
pub unsafe fn fjall_dealloc(ptr: *mut u8, layout: Layout) {
    fjall::dealloc(ptr, layout)
}
