#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|methods: Vec<fjall_fuzz::Operation>| {
    fjall_fuzz::run(methods, |layout| layout.size() <= 48 && layout.align() < 48);
});
