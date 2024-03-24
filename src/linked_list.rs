use std::cell::Cell;

#[derive(Clone, Debug)]
pub struct Node<T: Copy> {
    pub prev: Cell<Option<T>>,
    pub next: Cell<Option<T>>,
}

impl<T: Copy> Node<T> {
    pub const UNUSED: Node<T> = Node {
        prev: Cell::new(None),
        next: Cell::new(None),
    };

    pub unsafe fn used(&self) -> bool {
        self.prev.get().is_some() || self.next.get().is_some()
    }
}

#[derive(Debug)]
pub struct List<T: Copy> {
    pub first: Cell<Option<T>>,
    pub last: Cell<Option<T>>,
}

impl<T: Copy> Clone for List<T> {
    fn clone(&self) -> Self {
        Self {
            first: self.first.clone(),
            last: self.last.clone(),
        }
    }
}

impl<T: Copy + PartialEq> List<T> {
    pub const fn empty() -> Self {
        List {
            first: Cell::new(None),
            last: Cell::new(None),
        }
    }

    #[inline]
    pub unsafe fn iter<N: Fn(T) -> *mut Node<T>>(&self, node: N) -> Iter<true, T, N> {
        Iter {
            current: self.first.get(),
            node,
        }
    }

    #[inline]
    pub unsafe fn iter_rev<N: Fn(T) -> *mut Node<T>>(&self, node: N) -> Iter<false, T, N> {
        Iter {
            current: self.last.get(),
            node,
        }
    }

    pub unsafe fn only_entry(&self, element: T) -> bool {
        self.first.get() == Some(element) && self.last.get() == Some(element)
    }

    /// Checks if `element` is in some list or this list.
    pub unsafe fn may_contain(&self, element: T, node: impl Fn(T) -> *mut Node<T>) -> bool {
        (*node(element)).next.get().is_some()
            || (*node(element)).prev.get().is_some()
            || self.first.get() == Some(element)
    }

    pub unsafe fn remove(&self, element: T, node: impl Fn(T) -> *mut Node<T>) {
        let element_node = &mut *node(element);
        if let Some(prev) = element_node.prev.get() {
            (*node(prev)).next.set(element_node.next.get());
        }
        if let Some(next) = element_node.next.get() {
            (*node(next)).prev.set(element_node.prev.get());
        }
        if Some(element) == self.last.get() {
            self.last.set(element_node.prev.get());
        }
        if Some(element) == self.first.get() {
            self.first.set(element_node.next.get());
        }
        element_node.next.set(None);
        element_node.prev.set(None);
    }

    pub unsafe fn push_front(&self, element: T, node: impl Fn(T) -> *mut Node<T>) {
        let element_node = &mut *node(element);
        element_node.next.set(self.first.get());
        element_node.prev.set(None);
        if let Some(first) = self.first.get() {
            (*node(first)).prev.set(Some(element));
            self.first.set(Some(element));
        } else {
            debug_assert!(self.last.get().is_none());
            self.first.set(Some(element));
            self.last.set(Some(element));
        }
    }

    pub unsafe fn push_back(&self, element: T, node: impl Fn(T) -> *mut Node<T>) {
        let element_node = &mut *node(element);
        element_node.next.set(None);
        element_node.prev.set(self.last.get());
        if let Some(last) = self.last.get() {
            debug_assert!((*node(last)).next.get().is_none());
            (*node(last)).next.set(Some(element));
            self.last.set(Some(element));
        } else {
            self.first.set(Some(element));
            self.last.set(Some(element));
        }
    }
}

pub struct Iter<const FORWARD: bool, T, N> {
    current: Option<T>,
    node: N,
}

impl<const FORWARD: bool, T: Copy, N: Fn(T) -> *mut Node<T>> Iterator for Iter<FORWARD, T, N> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(current) = self.current {
            self.current = unsafe {
                if FORWARD {
                    (*(self.node)(current)).next.get()
                } else {
                    (*(self.node)(current)).prev.get()
                }
            };
            Some(current)
        } else {
            None
        }
    }
}
