use std::alloc::Layout;

#[cfg(any(miri, unix))]
use {core::alloc::GlobalAlloc, std::alloc::System};

#[cfg(any(miri, unix))]
pub fn alloc(layout: Layout) -> *mut u8 {
    unsafe { System.alloc_zeroed(layout) }
}

#[cfg(any(miri, unix))]
pub unsafe fn dealloc(ptr: *mut u8, layout: Layout) {
    System.dealloc(ptr, layout);
}

#[cfg(all(windows, not(miri)))]
pub fn alloc(layout: Layout) -> *mut u8 {
    use std::{mem, ptr::null_mut};
    use windows_sys::Win32::System::Memory::{
        MemExtendedParameterAddressRequirements, VirtualAlloc2, MEM_ADDRESS_REQUIREMENTS,
        MEM_COMMIT, MEM_EXTENDED_PARAMETER, MEM_RESERVE, PAGE_READWRITE,
    };

    let mut address_reqs: MEM_ADDRESS_REQUIREMENTS = unsafe { mem::zeroed() };
    address_reqs.Alignment = layout.align();

    let mut param: MEM_EXTENDED_PARAMETER = unsafe { mem::zeroed() };
    param.Anonymous2.Pointer = (&address_reqs as *const MEM_ADDRESS_REQUIREMENTS)
        .cast_mut()
        .cast();
    param.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;

    let result = unsafe {
        VirtualAlloc2(
            0,
            null_mut(),
            layout.size(),
            MEM_RESERVE | MEM_COMMIT,
            PAGE_READWRITE,
            &mut param,
            1,
        )
    };
    if result.is_null() {
        return null_mut();
    }
    /*  eprintln!(
        "VirtualAlloc2 {:x}, size - {:x}",
        result as usize,
        layout.size()
    );*/
    debug_assert!(result.is_aligned_to(layout.align()));
    result.cast()
}

#[cfg(all(windows, not(miri)))]
pub unsafe fn dealloc(ptr: *mut u8, layout: Layout) {
    use std::io::Error;
    use windows_sys::Win32::System::Memory::{VirtualFree, MEM_DECOMMIT, MEM_RELEASE};
    //eprintln!("VirtualFree {:x}, size - {:x}", ptr as usize, layout.size());
    let result = VirtualFree(ptr.cast(), 0, MEM_RELEASE);
    if cfg!(debug_assertions) && result == 0 {
        //   let os_error = Error::last_os_error();
        //  eprintln!("VirtualFree failed: {os_error:?}  ");
    }
    debug_assert_ne!(result, 0);
}

/*
#[cfg(unix)]
pub fn alloc(layout: Layout) -> *mut u8 {
    let result = aligned_alloc(layout.align(), layout.size());
    let result = mmap(
        NULL,
        layout.size(),
        PROT_READ | PROT_WRITE,
        MAP_SHARED | MAP_ANONYMOUS,
        -1,
        0,
    );
    if result == MAP_FAILED {
        return null_mut();
    }
    debug_assert!(result.is_aligned_to(layout.align()));
    result
}

#[cfg(unix)]
pub unsafe fn dealloc(ptr: *mut u8, layout: Layout) {
    use windows_sys::Win32::System::Memory::{VirtualFree, MEM_RELEASE};
    let result = VirtualFree(ptr.cast(), layout.size(), MEM_RELEASE);
    debug_assert_ne!(result, 0);
}
*/
