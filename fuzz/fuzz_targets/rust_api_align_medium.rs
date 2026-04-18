#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|methods: Vec<fjall_fuzz::Operation>| {
    fjall_fuzz::run(methods, |layout| {
        layout.align() <= 0x10000 && layout.size() <= 1 * 1024 * 1024
        /*
        layout.size() <= 1 * 1024 * 1024 - (0x10000 - 1)
            && layout.align() <= 0x10000
            && layout.align() > 16 */
    });
});
