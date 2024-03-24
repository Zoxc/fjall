#![no_main]

use std::alloc::Layout;

use fjall_fuzz::Size;
use libfuzzer_sys::arbitrary::{self, Arbitrary};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
enum Operation {
    Malloc { index: usize, layout: Size },
    Free { index: usize },
    Realloc { index: usize, new_size: usize },
}

const SEGMENT_SIZE: usize = 1 << 21;

const MAX_HEAP_SIZE: usize = 4 * 1024 * 1024 * 1024;

fuzz_target!(|methods: Vec<Operation>| {
    let mut allocs = Vec::new();

    let mut heap_size = 0;

    for method in methods {
        match method {
            Operation::Malloc { index, layout } => {
                let Some(total) = layout.size.checked_add(layout.align) else {
                    continue;
                };
                if !layout.valid() || layout.align > SEGMENT_SIZE || total > MAX_HEAP_SIZE {
                    continue;
                }
                let Ok(layout) = Layout::from_size_align(layout.size, layout.align) else {
                    continue;
                };

                // Remove allocations to fit the new one
                while heap_size + total > MAX_HEAP_SIZE {
                    let (ptr, size) = allocs.remove(index % allocs.len());
                    unsafe {
                        fjall::c::free(ptr);
                    }
                    heap_size -= size;
                }

                let ptr = fjall::c::aligned_alloc(layout.align(), layout.size());
                if !ptr.is_null() {
                    allocs.push((ptr, total));
                    heap_size += total;
                }
            }
            Operation::Free { index } => match allocs.get(index) {
                Some(&(ptr, size)) => {
                    unsafe {
                        fjall::c::free(ptr);
                    }
                    allocs.remove(index);
                    heap_size -= size;
                }
                _ => {}
            },
            Operation::Realloc { index, new_size } => match allocs.get(index) {
                Some(&(ptr, size)) => {
                    if new_size > MAX_HEAP_SIZE {
                        continue;
                    }

                    if new_size > size {
                        let delta = new_size - size;

                        // Remove allocations to fit the new one
                        while heap_size + delta > MAX_HEAP_SIZE {
                            let (ptr, size) = allocs.remove(index % allocs.len());
                            unsafe {
                                fjall::c::free(ptr);
                            }
                            heap_size -= size;
                        }
                    }

                    let new_ptr = unsafe { fjall::c::realloc(ptr, new_size) };
                    if new_size == 0 {
                        assert!(new_ptr.is_null());
                        heap_size -= size;
                        allocs.remove(index);
                    } else if !new_ptr.is_null() {
                        heap_size -= size;
                        heap_size += new_size;
                        allocs[index] = (new_ptr, new_size);
                    }
                }
                _ => {}
            },
        }
    }

    for (ptr, _) in allocs {
        unsafe {
            fjall::c::free(ptr);
        }
    }
});
