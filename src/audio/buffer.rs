//! Circular audio buffer for streaming
//!
//! Optimized for continuous writing and chunked reading.

/// Circular buffer for audio samples
pub struct AudioBuffer {
    buffer: Vec<f32>,
    capacity: usize,
    write_pos: usize,
    read_pos: usize,
    count: usize,
}

impl AudioBuffer {
    /// Create a new audio buffer with specified capacity (in samples)
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0.0; capacity],
            capacity,
            write_pos: 0,
            read_pos: 0,
            count: 0,
        }
    }

    /// Push samples into the buffer
    /// Returns number of samples written
    /// If buffer is full, it will overwrite old data if overwrite is true,
    /// otherwise it returns number of samples written up to capacity.
    pub fn push(&mut self, samples: &[f32]) -> usize {
        let _written = 0;

        let samples_len = samples.len();

        // If chunk is larger than capacity, only write the last capacity samples
        let (samples_to_write, _start_offset) = if samples_len > self.capacity {
            (&samples[samples_len - self.capacity..], 0)
        } else {
            (samples, 0)
        };

        let len = samples_to_write.len();

        // Calculate how much space is available before wrapping
        let space_at_end = self.capacity - self.write_pos;

        if len <= space_at_end {
            // No wrap needed
            self.buffer[self.write_pos..self.write_pos + len].copy_from_slice(samples_to_write);
            self.write_pos += len;
        } else {
            // Wrap around
            self.buffer[self.write_pos..self.capacity]
                .copy_from_slice(&samples_to_write[..space_at_end]);
            self.buffer[0..len - space_at_end].copy_from_slice(&samples_to_write[space_at_end..]);
            self.write_pos = len - space_at_end;
        }

        if self.write_pos >= self.capacity {
            self.write_pos = 0;
        }

        // Update count and read_pos
        self.count = (self.count + len).min(self.capacity);

        // If we overwrote data, move read_pos
        // This logic is tricky for a true ring buffer where we overwrite.
        // For simplicity, let's assume we just update count.
        // If count == capacity, read_pos should effectively move to write_pos (oldest data).
        if self.count == self.capacity {
            self.read_pos = self.write_pos;
        }

        len
    }

    /// Read exactly `size` samples
    /// Returns None if not enough data
    /// Advances read position
    pub fn read_chunk(&mut self, size: usize) -> Option<Vec<f32>> {
        if self.count < size {
            return None;
        }

        let mut chunk = vec![0.0; size];

        let space_at_end = self.capacity - self.read_pos;

        if size <= space_at_end {
            chunk.copy_from_slice(&self.buffer[self.read_pos..self.read_pos + size]);
            self.read_pos += size;
        } else {
            chunk[..space_at_end].copy_from_slice(&self.buffer[self.read_pos..self.capacity]);
            chunk[space_at_end..].copy_from_slice(&self.buffer[0..size - space_at_end]);
            self.read_pos = size - space_at_end;
        }

        if self.read_pos >= self.capacity {
            self.read_pos = 0;
        }

        self.count -= size;

        Some(chunk)
    }

    /// Peek at `size` samples without advancing read position
    pub fn peek_chunk(&self, size: usize) -> Option<Vec<f32>> {
        if self.count < size {
            return None;
        }

        let mut chunk = vec![0.0; size];
        let read_pos = self.read_pos;
        let space_at_end = self.capacity - read_pos;

        if size <= space_at_end {
            chunk.copy_from_slice(&self.buffer[read_pos..read_pos + size]);
        } else {
            chunk[..space_at_end].copy_from_slice(&self.buffer[read_pos..self.capacity]);
            chunk[space_at_end..].copy_from_slice(&self.buffer[0..size - space_at_end]);
        }

        Some(chunk)
    }

    /// Get current number of samples
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Clear buffer
    pub fn clear(&mut self) {
        self.write_pos = 0;
        self.read_pos = 0;
        self.count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_push_read() {
        let mut buf = AudioBuffer::new(10);

        buf.push(&[1.0, 2.0, 3.0]);
        assert_eq!(buf.len(), 3);

        let chunk = buf.read_chunk(2).unwrap();
        assert_eq!(chunk, vec![1.0, 2.0]);
        assert_eq!(buf.len(), 1);

        let chunk = buf.read_chunk(1).unwrap();
        assert_eq!(chunk, vec![3.0]);
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn test_buffer_wrap() {
        let mut buf = AudioBuffer::new(5);

        buf.push(&[1.0, 2.0, 3.0]);
        buf.read_chunk(2); // Read 1, 2. Left: 3. read_pos=2, write_pos=3

        buf.push(&[4.0, 5.0, 6.0]); // Write 4, 5 at 3,4. 6 wraps to 0. write_pos=1.
                                    // Buffer: [6.0, 0.0, 3.0, 4.0, 5.0]
                                    // read_pos=2 (val 3.0)
                                    // count = 1 + 3 = 4

        assert_eq!(buf.len(), 4);

        let chunk = buf.read_chunk(4).unwrap();
        assert_eq!(chunk, vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_overwrite() {
        let mut buf = AudioBuffer::new(3);
        buf.push(&[1.0, 2.0]);
        buf.push(&[3.0, 4.0]); // Should overwrite 1. Buffer: [4.0, 2.0, 3.0] or similar
                               // Logic:
                               // push 1,2 -> buf=[1,2,0], w=2, r=0, c=2
                               // push 3,4 ->
                               //   write 3 at 2 -> buf=[1,2,3], w=0 (wrap)
                               //   write 4 at 0 -> buf=[4,2,3], w=1
                               //   c = 2 + 2 = 4 -> min(3) = 3
                               //   c == cap, so r = w = 1

        assert_eq!(buf.len(), 3);
        let chunk = buf.read_chunk(3).unwrap();
        // r=1 -> 2.0
        // r=2 -> 3.0
        // r=0 -> 4.0
        assert_eq!(chunk, vec![2.0, 3.0, 4.0]);
    }
}
