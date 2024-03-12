use core::slice;
use fjall::Alloc;
use rand::Rng;
use std::alloc::GlobalAlloc;
use std::alloc::Layout;
use std::sync::atomic::AtomicPtr;
use std::sync::mpsc::channel;
use std::thread::yield_now;

struct Allocation {
    layout: Layout,
    ptr: AtomicPtr<u8>,
    seed: u8,
}

impl Allocation {
    fn new() -> Self {
        unsafe {
            let mut rng = rand::thread_rng();
            let size: usize = rng.gen_range(0..1024);
            let layout = Layout::from_size_align(size as usize, 4).unwrap();
            let ptr = Alloc.alloc(layout);
            let seed: u8 = rng.gen();
            if !ptr.is_null() {
                slice::from_raw_parts_mut(ptr, size).fill(seed);
            }
            Allocation {
                ptr: AtomicPtr::new(ptr),
                layout,
                seed,
            }
        }
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        let ptr = *self.ptr.get_mut();
        if !ptr.is_null() {
            unsafe {
                assert!(slice::from_raw_parts(ptr, self.layout.size())
                    .iter()
                    .all(|v| *v == self.seed));

                Alloc.dealloc(ptr, self.layout);
            }
        }
    }
}

const THREADS: usize = if cfg!(miri) { 4 } else { 16 };
const MIRI_FACTOR: usize = if cfg!(miri) { 1 } else { 10000 };

fn main() {
    unsafe {
        let a = Alloc.alloc(Layout::new::<u8>());
        let b = Alloc.alloc(Layout::new::<String>());
        Alloc.dealloc(a, Layout::new::<u8>());
        Alloc.dealloc(b, Layout::new::<String>());

        std::thread::scope(|scope| {
            for _ in 0..THREADS {
                scope.spawn(|| {
                    let mut rng = rand::thread_rng();

                    let mut allocs = Vec::new();

                    for _ in 0..(10 * MIRI_FACTOR) {
                        //  if rng.gen() {
                        allocs.push(Allocation::new());
                        //   }

                        //   if rng.gen() && allocs.len() > 50 {
                        allocs.remove(rng.gen_range(0..allocs.len()));
                        //   }
                    }
                });
            }
        });

        std::thread::scope(|scope| {
            let (senders, receivers): (Vec<_>, Vec<_>) =
                (0..THREADS).map(|_| channel::<Allocation>()).unzip();
            for rx in receivers {
                let senders = senders.clone();
                scope.spawn(move || {
                    let mut rng = rand::thread_rng();

                    let mut allocs = Vec::new();

                    for _ in 0..(10 * MIRI_FACTOR) {
                        if rng.gen() {
                            allocs.push(Allocation::new());
                        }

                        rx.try_recv().ok();

                        senders[rng.gen_range(0..THREADS)]
                            .send(Allocation::new())
                            .ok();

                        yield_now();

                        if rng.gen() && allocs.len() > 50 {
                            allocs.remove(rng.gen_range(0..allocs.len()));
                        }
                    }
                });
            }
        });
    }
}
