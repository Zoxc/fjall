#![no_main]

use fjall_fuzz::Allocation;
use fjall_fuzz::Size;
use fjall_fuzz::__asan_poison_memory_region;
use libfuzzer_sys::arbitrary::{self, Arbitrary};
use libfuzzer_sys::fuzz_target;
use std::alloc::Layout;
use std::sync::atomic::AtomicPtr;
use std::sync::mpsc::channel;

#[derive(Arbitrary, Debug)]
enum Operation {
    Alloc {
        index: usize,
        layout: Size,
    },
    Dealloc {
        index: usize,
    },
    Send {
        index: usize,
        remove: usize,
        thread: usize,
    },
    DeallocSent {
        index: usize,
    },
}

const SEGMENT_SIZE: usize = 1 << 21;

const THREADS: usize = 8;

const MAX_LOCAL_HEAP_SIZE: usize = 4 * 1024 * 1024 * 1024 / THREADS;

fuzz_target!(|methods: [Vec<Operation>; THREADS]| {
    std::thread::scope(|scope| {
        let (senders, receivers): (Vec<_>, Vec<_>) = (0..THREADS)
            .map(|_| channel::<(Allocation, usize)>())
            .unzip();
        for (rx, methods) in receivers.into_iter().zip(methods) {
            let senders = senders.clone();
            scope.spawn(move || {
                let mut allocs = Vec::new();
                let mut sent_allocs = Vec::new();

                let mut heap_size = 0;

                let trim = |fit,
                            index,
                            heap_size: &mut usize,
                            allocs: &mut Vec<Allocation>,
                            sent_allocs: &mut Vec<Allocation>| {
                    // Remove allocations to fit the new one
                    loop {
                        if *heap_size + fit <= MAX_LOCAL_HEAP_SIZE {
                            break;
                        }
                        {
                            let alloc = allocs.swap_remove(index % allocs.len());
                            *heap_size -= alloc.cost();
                        }
                        if *heap_size + fit <= MAX_LOCAL_HEAP_SIZE {
                            break;
                        }
                        {
                            let alloc = sent_allocs.swap_remove(index % sent_allocs.len());
                            *heap_size -= alloc.cost();
                        }
                    }
                };

                for method in methods {
                    if let Some((alloc, remove)) = rx.try_recv().ok() {
                        trim(
                            alloc.cost(),
                            remove,
                            &mut heap_size,
                            &mut allocs,
                            &mut sent_allocs,
                        );
                        sent_allocs.push(alloc);
                    }

                    match method {
                        Operation::Alloc { index, layout } => {
                            let Some(total) = layout.size.checked_add(layout.align) else {
                                continue;
                            };
                            if !layout.valid()
                                || layout.align > SEGMENT_SIZE
                                || total > MAX_LOCAL_HEAP_SIZE
                            {
                                continue;
                            }
                            let Ok(layout) = Layout::from_size_align(layout.size, layout.align)
                            else {
                                continue;
                            };

                            trim(total, index, &mut heap_size, &mut allocs, &mut sent_allocs);

                            let ptr = unsafe { fjall::alloc(layout) };
                            if !ptr.is_null() {
                                unsafe { __asan_poison_memory_region(ptr, layout.size()) };
                                heap_size += total;
                                allocs.push(Allocation {
                                    ptr: AtomicPtr::new(ptr),
                                    layout,
                                });
                            }
                        }
                        Operation::Dealloc { index } => match allocs.get(index) {
                            Some(alloc) => {
                                heap_size -= alloc.cost();
                                allocs.swap_remove(index);
                            }
                            _ => {}
                        },
                        Operation::DeallocSent { index } => match sent_allocs.get(index) {
                            Some(alloc) => {
                                heap_size -= alloc.cost();
                                sent_allocs.swap_remove(index);
                            }
                            _ => {}
                        },
                        Operation::Send {
                            index,
                            thread,
                            remove,
                        } => {
                            if thread < THREADS && index < allocs.len() {
                                let alloc = allocs.swap_remove(index);
                                heap_size -= alloc.cost();
                                senders[thread].send((alloc, remove)).ok();
                            }
                        }
                    }
                }
            });
        }
    });
});
