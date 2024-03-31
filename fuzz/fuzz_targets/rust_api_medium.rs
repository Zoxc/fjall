#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|methods: Vec<fjall_fuzz::Operation>| {
    fjall_fuzz::run(methods, |layout| {
        layout.size() <= 16 * 1024 * 1024 && layout.align() < 16 * 1024 * 1024
    });
});
