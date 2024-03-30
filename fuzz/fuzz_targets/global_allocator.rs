#![no_main]

#[global_allocator]
static GLOBAL: fjall::Alloc = fjall::Alloc;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|methods: Vec<fjall_fuzz::Operation>| {
    fjall_fuzz::run(methods, 16 * 1024 * 1024);
});
