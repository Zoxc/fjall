use crate::{abort_on_panic, heap::Heap, with_heap};
use crate::{
    align_down, align_up,
    segment::{OPTION_PURGE_DELAY, OPTION_PURGE_DOES_DECOMMIT},
    system,
};
use core::ptr::read_volatile;
use sptr::Strict;
use std::{
    alloc::Layout,
    sync::atomic::{AtomicI64, Ordering},
    time::Instant,
};
use windows_sys::Win32::System::SystemServices::{DLL_PROCESS_DETACH, DLL_THREAD_DETACH};

// FIXME: Apparently this trick doesn't work for loaded DLLs.
// The callback runs before main, so we could dynamically use another solution in DLLs?

// Use the `.CRT$XLM` section as that us executed
// later then the `.CRT$XLB` section used by `std`.
#[link_section = ".CRT$XLM"]
#[used]
static THREAD_CALLBACK: unsafe extern "system" fn(*mut (), u32, *mut ()) = callback;

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
    use std::cmp::max;
    use windows_sys::Win32::System::Performance::{
        QueryPerformanceCounter, QueryPerformanceFrequency,
    };
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

use std::{mem, ptr::null_mut};
use windows_sys::Win32::System::Memory::{
    VirtualAlloc, MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, PAGE_READWRITE,
};

pub unsafe fn commit(ptr: *mut u8, size: usize) -> bool {
    !VirtualAlloc(ptr.cast_const().cast(), size, MEM_COMMIT, PAGE_READWRITE).is_null()
}

pub unsafe fn decommit(ptr: *mut u8, size: usize) -> bool {
    use std::{mem, ptr::null_mut};
    use windows_sys::Win32::System::Memory::{
        VirtualAlloc, VirtualFree, MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_DECOMMIT,
        PAGE_READWRITE,
    };
    let result = VirtualFree(ptr.cast(), size, MEM_DECOMMIT);
    debug_assert_ne!(result, 0);
    result != 0
}

pub fn alloc(layout: Layout, commit: bool) -> (*mut u8, bool) {
    use std::{mem, ptr::null_mut};
    use windows_sys::Win32::System::Memory::{
        MemExtendedParameterAddressRequirements, VirtualAlloc2, MEM_ADDRESS_REQUIREMENTS,
        MEM_COMMIT, MEM_EXTENDED_PARAMETER, MEM_RESERVE, PAGE_READWRITE,
    };

    use crate::align_up;

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
            align_up(layout.size(), system::page_size()), // FIXME: Round
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
    if result.is_null() {
        return (null_mut(), false);
    }
    /*  eprintln!(
        "VirtualAlloc2 {:x}, size - {:x}",
        result as usize,
        layout.size()
    );*/
    debug_assert!(result.is_aligned_to(layout.align()));
    (result.cast(), commit)
}

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
