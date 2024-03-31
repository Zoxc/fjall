#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|methods: Vec<fjall_fuzz::Operation>| {
    fjall_fuzz::run(methods, |_| true);
});
