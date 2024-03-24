use crate::{heap::Heap, with_heap};

thread_local! {
    static THREAD_EXIT: ThreadExit = const { ThreadExit };
}
struct ThreadExit;
impl Drop for ThreadExit {
    fn drop(&mut self) {
        unsafe {
            with_heap(|heap| Heap::done(heap));
        }
    }
}
pub fn register_thread() {
    THREAD_EXIT.with(|_| {});
}
