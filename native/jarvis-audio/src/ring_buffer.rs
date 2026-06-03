//! A fixed-capacity ring buffer of 16-bit PCM samples. The capture callback
//! pushes samples in; the consumer (the wake/VAD/endpointer, next increment)
//! drains them. When full, the oldest sample is overwritten so capture never
//! blocks. Pure logic — fully unit-tested (no audio hardware needed).

/// A simple overwrite-on-full ring buffer of `i16` samples.
pub struct RingBuffer {
    buf: Vec<i16>,
    capacity: usize,
    head: usize, // index of the next write
    len: usize,  // number of valid samples currently held
}

impl RingBuffer {
    /// Create a ring holding up to `capacity` samples (min 1).
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        RingBuffer { buf: vec![0; cap], capacity: cap, head: 0, len: 0 }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Push one sample. Returns true iff it overwrote the oldest (buffer full).
    pub fn push(&mut self, sample: i16) -> bool {
        let overwrote = self.len == self.capacity;
        self.buf[self.head] = sample;
        self.head = (self.head + 1) % self.capacity;
        if !overwrote {
            self.len += 1;
        }
        overwrote
    }

    /// Push many samples in order.
    pub fn push_slice(&mut self, samples: &[i16]) {
        for &s in samples {
            self.push(s);
        }
    }

    /// Drain everything in FIFO order, leaving the buffer empty.
    pub fn drain(&mut self) -> Vec<i16> {
        let mut out = Vec::with_capacity(self.len);
        let start = (self.head + self.capacity - self.len) % self.capacity;
        for i in 0..self.len {
            out.push(self.buf[(start + i) % self.capacity]);
        }
        self.len = 0;
        self.head = 0;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_drain_fifo() {
        let mut r = RingBuffer::new(8);
        assert!(r.is_empty());
        r.push_slice(&[1, 2, 3]);
        assert_eq!(r.len(), 3);
        assert_eq!(r.drain(), vec![1, 2, 3]);
        assert!(r.is_empty());
    }

    #[test]
    fn overwrites_when_full() {
        let mut r = RingBuffer::new(3);
        assert!(!r.push(1));
        assert!(!r.push(2));
        assert!(!r.push(3));
        assert!(r.push(4)); // full -> overwrote the oldest (1)
        assert_eq!(r.len(), 3);
        assert_eq!(r.drain(), vec![2, 3, 4]);
    }

    #[test]
    fn zero_capacity_is_clamped_to_one() {
        let mut r = RingBuffer::new(0);
        assert_eq!(r.capacity(), 1);
        r.push(7);
        assert_eq!(r.drain(), vec![7]);
    }

    #[test]
    fn wraps_around_cleanly() {
        let mut r = RingBuffer::new(4);
        r.push_slice(&[1, 2, 3, 4, 5, 6]); // 1,2 overwritten
        assert_eq!(r.drain(), vec![3, 4, 5, 6]);
    }
}
