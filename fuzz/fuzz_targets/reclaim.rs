#![no_main]

use fjall_fuzz::Allocation;
use fjall_fuzz::Size;
use fjall_fuzz::MAX_HEAP_SIZE;
use fjall_fuzz::SEGMENT_SIZE;
use libfuzzer_sys::arbitrary::{self, Arbitrary};
use libfuzzer_sys::fuzz_target;
use std::alloc::Layout;
use std::sync::Arc;
use std::sync::Condvar;
use std::sync::Mutex;
use std::thread;
use std::thread::JoinHandle;

#[derive(Arbitrary, Debug)]
enum Operation {
    Alloc { index: usize, layout: Size },
    Dealloc { index: usize },
    Migrate,
}

fn run_thread(
    mut allocs: Vec<Allocation>,
    mut heap_size: usize,
    mut methods: Vec<Operation>,
    limit: usize,
    data: Arc<Threads>,
) {
    loop {
        let Some(method) = methods.pop() else {
            let mut guard = data.data.lock().unwrap();
            guard.1 = true;
            data.cond.notify_one();
            return;
        };
        match method {
            Operation::Alloc { index, layout } => {
                if layout.size > limit || layout.align > limit {
                    continue;
                }
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
                    let alloc = allocs.swap_remove(index % allocs.len());
                    heap_size -= alloc.cost();
                }

                if let Some(alloc) = Allocation::new(layout) {
                    heap_size += alloc.cost();
                    allocs.push(alloc);
                }
            }
            Operation::Dealloc { index } => {
                if index < allocs.len() {
                    heap_size -= allocs.swap_remove(index).cost();
                }
            }
            Operation::Migrate => {
                let data_ = data.clone();
                // Take the lock before spawning the thread to ensure we add it to the
                // thread list before we mark the test as done.
                let mut guard = data.data.lock().unwrap();
                let handle =
                    thread::spawn(move || run_thread(allocs, heap_size, methods, limit, data_));
                guard.0.push(handle);
                return;
            }
        }
    }
}

struct Threads {
    data: Mutex<(Vec<JoinHandle<()>>, bool)>,
    cond: Condvar,
}

fuzz_target!(|methods: Vec<Operation>| {
    let threads = Arc::new(Threads {
        data: Mutex::new((Vec::new(), false)),
        cond: Condvar::new(),
    });
    run_thread(Vec::new(), 0, methods, 4 * 2 * 1024 * 1024, threads.clone());
    let mut guard = threads
        .cond
        .wait_while(threads.data.lock().unwrap(), |t| !t.1)
        .unwrap();
    for thread in guard.0.drain(..) {
        thread.join().unwrap();
    }
});
