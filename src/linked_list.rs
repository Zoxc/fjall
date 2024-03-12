use core::ptr::null_mut;

#[derive(Copy, Clone, Debug)]
pub struct Node<T> {
    pub prev: *mut T,
    pub next: *mut T,
}

impl<T> Node<T> {
    pub const UNUSED: Node<T> = Node {
        prev: null_mut(),
        next: null_mut(),
    };
}

#[derive(Debug)]
pub struct List<T> {
    pub first: *mut T,
    pub last: *mut T,
}

impl<T> Copy for List<T> {}
impl<T> Clone for List<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> List<T> {
    pub const EMPTY: List<T> = List {
        first: null_mut(),
        last: null_mut(),
    };

    #[inline]
    pub unsafe fn iter<N: Fn(*mut T) -> *mut Node<T>>(&self, node: N) -> Iter<T, N> {
        Iter {
            current: self.first,
            node,
        }
    }

    pub unsafe fn remove(&mut self, element: *mut T, node: impl Fn(*mut T) -> *mut Node<T>) {
        let element_node = &mut *node(element);
        if !element_node.prev.is_null() {
            (*node(element_node.prev)).next = element_node.next;
        }
        if !element_node.next.is_null() {
            (*node(element_node.next)).prev = element_node.prev;
        }
        if element == self.last {
            self.last = element_node.prev;
        }
        if element == self.first {
            self.first = element_node.next;
        }
        element_node.next = null_mut();
        element_node.prev = null_mut();
    }

    pub unsafe fn push_front(&mut self, element: *mut T, node: impl Fn(*mut T) -> *mut Node<T>) {
        let element_node = &mut *node(element);
        element_node.next = self.first;
        element_node.prev = null_mut();
        if self.first.is_null() {
            debug_assert!(self.last.is_null());
            self.first = element;
            self.last = element;
        } else {
            (*node(self.first)).prev = element;
            self.first = element;
        }
    }

    pub unsafe fn push_back(&mut self, element: *mut T, node: impl Fn(*mut T) -> *mut Node<T>) {
        let element_node = &mut *node(element);
        element_node.next = null_mut();
        element_node.prev = self.last;
        if !self.last.is_null() {
            debug_assert!((*node(self.last)).next.is_null());
            (*node(self.last)).next = element;
            self.last = element;
        } else {
            self.first = element;
            self.last = element;
        }
    }
}

pub struct Iter<T, N> {
    current: *mut T,
    node: N,
}

impl<T, N: Fn(*mut T) -> *mut Node<T>> Iterator for Iter<T, N> {
    type Item = *mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.current.is_null() {
            let result = self.current;
            self.current = unsafe { (*(self.node)(self.current)).next };
            Some(result)
        } else {
            None
        }
    }
}
