#![allow(unstable_name_collisions)]

use libfuzzer_sys::arbitrary::{self, Arbitrary};
use libfuzzer_sys::Corpus;
use sptr::Strict;
use std::alloc::Layout;
use std::sync::atomic::AtomicPtr;

extern "C" {
    pub fn __asan_poison_memory_region(addr: *mut u8, size: usize);
    pub fn __asan_unpoison_memory_region(addr: *mut u8, size: usize);
}

#[derive(Arbitrary, Debug)]
pub struct Size {
    pub size: usize,
    pub align: usize,
}

impl Size {
    pub fn valid(&self) -> bool {
        self.size > 0 && self.align > 0 && self.align.is_power_of_two()
    }
}

pub struct Allocation {
    pub layout: Layout,
    pub ptr: AtomicPtr<u8>,
}

impl Allocation {
    pub fn cost(&self) -> usize {
        self.layout.size() + self.layout.align()
    }

    pub fn new(layout: Layout) -> Option<Allocation> {
        let ptr = unsafe { fjall::alloc(layout) };
        (!ptr.is_null()).then(|| {
            unsafe { __asan_poison_memory_region(ptr, layout.size()) };
            Allocation {
                ptr: AtomicPtr::new(ptr),
                layout,
            }
        })
    }

    fn overlaps(&self, other: &Allocation) -> bool {
        let a = self.ptr.as_ptr().addr();
        let b = other.ptr.as_ptr().addr();
        let b_end = b + other.layout.size() - 1;
        (a..self.layout.size()).contains(&b) || (a..self.layout.size()).contains(&b_end)
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        unsafe {
            __asan_unpoison_memory_region(*self.ptr.get_mut(), self.layout.size());
            fjall::dealloc(*self.ptr.get_mut(), self.layout);
        }
    }
}

#[derive(Arbitrary, Debug)]
pub enum Operation {
    Alloc { index: usize, layout: Size },
    Dealloc { index: usize },
}

impl Operation {
    fn valid(&self, limit: usize) -> bool {
        match self {
            Operation::Alloc { index: _, layout } => {
                if layout.size > limit || layout.align > limit {
                    return false;
                }
                let Some(total) = layout.size.checked_add(layout.align) else {
                    return false;
                };
                if !layout.valid() || layout.align > SEGMENT_SIZE || total > MAX_HEAP_SIZE {
                    return false;
                }
                Layout::from_size_align(layout.size, layout.align).is_ok()
            }
            Operation::Dealloc { .. } => true,
        }
    }
}

pub const SEGMENT_SIZE: usize = 1 << 21;

pub const MAX_HEAP_SIZE: usize = 4 * 1024 * 1024 * 1024;

pub fn run(methods: Vec<Operation>, limit: usize) -> Corpus {
    if !methods.iter().all(|o| o.valid(limit)) {
        return Corpus::Reject;
    }

    let mut allocs: Vec<Allocation> = Vec::new();

    let mut heap_size = 0;

    for method in methods {
        match method {
            Operation::Alloc { index, layout } => {
                let total = layout.size + layout.align;

                let layout = Layout::from_size_align(layout.size, layout.align).unwrap();

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
                if !allocs.is_empty() {
                    let index = index % allocs.len();
                    heap_size -= allocs.swap_remove(index).cost();
                }
            }
        }
    }

    Corpus::Keep
}

pub fn overlap(methods: Vec<Operation>, limit: usize) {
    let mut allocs: Vec<Allocation> = Vec::new();

    let mut heap_size = 0;

    for method in methods {
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
                    let point = allocs
                        .partition_point(|x| x.ptr.as_ptr().addr() < alloc.ptr.as_ptr().addr());

                    allocs.get(point).map(|a| {
                        assert!(!a.overlaps(&alloc));
                    });
                    allocs.get(point + 1).map(|a| {
                        assert!(!a.overlaps(&alloc));
                    });

                    heap_size += alloc.cost();
                    allocs.insert(point, alloc);
                }
            }
            Operation::Dealloc { index } => {
                if index < allocs.len() {
                    heap_size -= allocs.swap_remove(index).cost();
                }
            }
        }
    }
}
