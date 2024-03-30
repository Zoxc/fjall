use crate::{abort_on_panic, heap::Heap, with_heap};
use std::cmp::max;
use std::sync::atomic::{AtomicI64, Ordering};
use windows_sys::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
use windows_sys::Win32::System::SystemServices::{DLL_PROCESS_DETACH, DLL_THREAD_DETACH};
#[cfg(not(feature = "system-allocator"))]
use {
    crate::Ptr,
    crate::{align_up, system, validate_align},
    sptr::Strict,
    std::alloc::Layout,
    std::cmp::min,
    std::os::raw::c_void,
    std::sync::atomic::{AtomicU32, AtomicUsize},
    std::{mem, ptr::null_mut},
    windows_sys::Win32::System::Memory::{
        MemExtendedParameterAddressRequirements, VirtualAlloc, VirtualAlloc2, VirtualFree,
        MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_DECOMMIT, MEM_EXTENDED_PARAMETER, MEM_RELEASE,
        MEM_RESERVE, PAGE_READWRITE,
    },
    windows_sys::Win32::System::SystemInformation::GetSystemInfo,
};

// Apparently this trick doesn't work for loaded DLLs, that but it works on Window 10 22H2.
// Maybe it didn't work on some older Windows / CRT versions?

// Use the `.CRT$XLM` section as that us executed
// later then the `.CRT$XLB` section used by `std`.
#[link_section = ".CRT$XLM"]
#[used]
static THREAD_CALLBACK: unsafe extern "system" fn(*mut (), u32, *mut ()) = callback;

#[cfg(not(miri))]
pub fn register_thread() {}

unsafe extern "system" fn callback(_h: *mut (), dw_reason: u32, _pv: *mut ()) {
    if dw_reason == DLL_THREAD_DETACH || dw_reason == DLL_PROCESS_DETACH {
        abort_on_panic(|| {
            with_heap(|heap| Heap::done(heap));
        })
    }
}

/// A clock in milliseconds.
pub fn clock_now() -> Option<u64> {
    let frequency = {
        static FREQUENCY: AtomicI64 = AtomicI64::new(0);

        let cached = FREQUENCY.load(Ordering::Relaxed);
        if cached != 0 {
            cached
        } else {
            let mut frequency = 0;
            unsafe {
                QueryPerformanceFrequency(&mut frequency);
            }
            FREQUENCY.store(frequency, Ordering::Relaxed);
            frequency
        }
    };

    let mut clock: i64 = 0;
    (unsafe { QueryPerformanceCounter(&mut clock) });
    Some(mul_div_u64(clock as u64, 1000, max(frequency, 1) as u64))
}

pub fn mul_div_u64(value: u64, numer: u64, denom: u64) -> u64 {
    let q = value / denom;
    let r = value % denom;
    // Decompose value as (value/denom*denom + value%denom),
    // substitute into (value*numer)/denom and simplify.
    // r < denom, so (denom*numer) is the upper bound of (r*numer)
    q * numer + r * numer / denom
}

#[cfg(not(feature = "system-allocator"))]
pub fn page_size() -> usize {
    static PAGE_SIZE: AtomicU32 = AtomicU32::new(0);

    let cached = PAGE_SIZE.load(Ordering::Relaxed);
    if cached != 0 {
        cached as usize
    } else {
        let mut info = unsafe { mem::zeroed() };
        unsafe {
            GetSystemInfo(&mut info);
        }
        PAGE_SIZE.store(info.dwPageSize, Ordering::Relaxed);
        info.dwPageSize as usize
    }
}

#[cfg(not(feature = "system-allocator"))]
fn highest_address() -> usize {
    static HIGHEST_ADDRESS: AtomicUsize = AtomicUsize::new(0);

    let cached = HIGHEST_ADDRESS.load(Ordering::Relaxed);
    if cached != 0 {
        cached
    } else {
        let mut info = unsafe { mem::zeroed() };
        unsafe {
            GetSystemInfo(&mut info);
        }
        // Exclude the last byte in the address space
        let highest_address = min(info.lpMaximumApplicationAddress.addr(), usize::MAX - 1);
        HIGHEST_ADDRESS.store(highest_address, Ordering::Relaxed);
        highest_address
    }
}

#[cfg(not(feature = "system-allocator"))]
#[derive(Clone, Copy, Debug)]
pub struct SystemAllocation;

#[cfg(not(feature = "system-allocator"))]
pub unsafe fn commit(ptr: Ptr<u8>, size: usize) -> bool {
    !VirtualAlloc(
        ptr.as_ptr().cast_const().cast(),
        size,
        MEM_COMMIT,
        PAGE_READWRITE,
    )
    .is_null()
}

#[cfg(not(feature = "system-allocator"))]
pub unsafe fn decommit(ptr: Ptr<u8>, size: usize) -> bool {
    let result = VirtualFree(ptr.as_ptr().cast(), size, MEM_DECOMMIT);
    assert_ne!(result, 0);
    result != 0
}

#[cfg(not(feature = "system-allocator"))]
pub fn alloc(layout: Layout, commit: bool) -> Option<(SystemAllocation, Ptr<u8>, bool)> {
    let mut address_reqs: MEM_ADDRESS_REQUIREMENTS = unsafe { mem::zeroed() };
    address_reqs.Alignment = layout.align();
    address_reqs.HighestEndingAddress = null_mut::<c_void>().with_addr(highest_address());

    let mut param: MEM_EXTENDED_PARAMETER = unsafe { mem::zeroed() };
    param.Anonymous2.Pointer = (&address_reqs as *const MEM_ADDRESS_REQUIREMENTS)
        .cast_mut()
        .cast();
    param.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;

    let result = unsafe {
        VirtualAlloc2(
            0,
            null_mut(),
            align_up(layout.size(), system::page_size()),
            if commit {
                MEM_RESERVE | MEM_COMMIT
            } else {
                MEM_RESERVE
            },
            PAGE_READWRITE,
            &mut param,
            1,
        )
    };
    let result: Ptr<u8> = unsafe { Ptr::new(result.cast())? };
    validate_align(result.as_ptr(), layout.align());
    Some((SystemAllocation, result, commit))
}

#[cfg(not(feature = "system-allocator"))]
pub unsafe fn dealloc(_alloc: SystemAllocation, ptr: Ptr<u8>, _layout: Layout) {
    let result = VirtualFree(ptr.as_ptr().cast(), 0, MEM_RELEASE);
    assert_ne!(result, 0);
}
