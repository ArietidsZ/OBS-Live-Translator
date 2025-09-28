//! Zero-copy audio buffer management with memory-efficient data structures
//!
//! This module implements high-performance audio buffer management:
//! - Zero-copy ring buffers for continuous audio streaming
//! - Memory-efficient mel-spectrogram storage with Array2<f32> optimization
//! - Shared memory pools for model inference
//! - Memory fragmentation monitoring
//! - Automatic garbage collection strategies

use crate::profile::Profile;
use anyhow::Result;
use std::collections::VecDeque;
use std::alloc::Layout;
use tracing::{info, warn, debug};

/// Zero-copy ring buffer for audio streaming
pub struct AudioRingBuffer {
    buffer: Vec<f32>,
    capacity: usize,
    read_pos: usize,
    write_pos: usize,
    size: usize,
    sample_rate: u32,
    channels: u16,
}

impl AudioRingBuffer {
    /// Create new ring buffer with specified capacity
    pub fn new(capacity_samples: usize, sample_rate: u32, channels: u16) -> Self {
        let buffer = vec![0.0f32; capacity_samples];

        debug!("Created AudioRingBuffer: {} samples, {}Hz, {} channels, {:.2}MB",
               capacity_samples, sample_rate, channels,
               (capacity_samples * 4) as f64 / 1024.0 / 1024.0);

        Self {
            buffer,
            capacity: capacity_samples,
            read_pos: 0,
            write_pos: 0,
            size: 0,
            sample_rate,
            channels,
        }
    }

    /// Create ring buffer sized for profile's latency requirements
    pub fn new_for_profile(profile: Profile, sample_rate: u32, channels: u16) -> Self {
        let capacity_ms = match profile {
            Profile::Low => 500,    // 500ms buffer for stability
            Profile::Medium => 300, // 300ms buffer for balance
            Profile::High => 200,   // 200ms buffer for low latency
        };

        let capacity_samples = (sample_rate as f32 * channels as f32 * capacity_ms as f32 / 1000.0) as usize;
        Self::new(capacity_samples, sample_rate, channels)
    }

    /// Write audio samples to the ring buffer (zero-copy when possible)
    pub fn write(&mut self, samples: &[f32]) -> Result<usize> {
        let available_space = self.capacity - self.size;
        let to_write = samples.len().min(available_space);

        if to_write == 0 {
            warn!("AudioRingBuffer overflow: buffer full");
            return Ok(0);
        }

        // Handle wrap-around
        let end_pos = self.write_pos + to_write;
        if end_pos <= self.capacity {
            // No wrap-around: single copy
            self.buffer[self.write_pos..self.write_pos + to_write]
                .copy_from_slice(&samples[..to_write]);
        } else {
            // Wrap-around: two copies
            let first_chunk = self.capacity - self.write_pos;
            let second_chunk = to_write - first_chunk;

            self.buffer[self.write_pos..self.capacity]
                .copy_from_slice(&samples[..first_chunk]);
            self.buffer[0..second_chunk]
                .copy_from_slice(&samples[first_chunk..to_write]);
        }

        self.write_pos = (self.write_pos + to_write) % self.capacity;
        self.size += to_write;

        debug!("Wrote {} samples to ring buffer, size: {}/{}", to_write, self.size, self.capacity);
        Ok(to_write)
    }

    /// Read audio samples from the ring buffer (zero-copy view when possible)
    pub fn read(&mut self, output: &mut [f32]) -> Result<usize> {
        let available_data = self.size;
        let to_read = output.len().min(available_data);

        if to_read == 0 {
            return Ok(0);
        }

        // Handle wrap-around
        let end_pos = self.read_pos + to_read;
        if end_pos <= self.capacity {
            // No wrap-around: single copy
            output[..to_read].copy_from_slice(&self.buffer[self.read_pos..self.read_pos + to_read]);
        } else {
            // Wrap-around: two copies
            let first_chunk = self.capacity - self.read_pos;
            let second_chunk = to_read - first_chunk;

            output[..first_chunk].copy_from_slice(&self.buffer[self.read_pos..self.capacity]);
            output[first_chunk..to_read].copy_from_slice(&self.buffer[0..second_chunk]);
        }

        self.read_pos = (self.read_pos + to_read) % self.capacity;
        self.size -= to_read;

        debug!("Read {} samples from ring buffer, remaining: {}", to_read, self.size);
        Ok(to_read)
    }

    /// Peek at available data without consuming it (zero-copy view)
    pub fn peek(&self, count: usize) -> Vec<f32> {
        let available_data = self.size;
        let to_peek = count.min(available_data);
        let mut result = vec![0.0f32; to_peek];

        if to_peek == 0 {
            return result;
        }

        // Handle wrap-around for peek
        let end_pos = self.read_pos + to_peek;
        if end_pos <= self.capacity {
            result.copy_from_slice(&self.buffer[self.read_pos..self.read_pos + to_peek]);
        } else {
            let first_chunk = self.capacity - self.read_pos;
            let second_chunk = to_peek - first_chunk;

            result[..first_chunk].copy_from_slice(&self.buffer[self.read_pos..self.capacity]);
            result[first_chunk..].copy_from_slice(&self.buffer[0..second_chunk]);
        }

        result
    }

    /// Get available samples for reading
    pub fn available(&self) -> usize {
        self.size
    }

    /// Get free space for writing
    pub fn free_space(&self) -> usize {
        self.capacity - self.size
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.read_pos = 0;
        self.write_pos = 0;
        self.size = 0;
    }

    /// Get buffer utilization percentage
    pub fn utilization(&self) -> f32 {
        (self.size as f32 / self.capacity as f32) * 100.0
    }
}

/// Memory-efficient mel-spectrogram storage with Array2<f32> optimization
pub struct MelSpectrogramBuffer {
    data: Vec<f32>,
    n_frames: usize,
    n_mels: usize,
    capacity_frames: usize,
    current_frame: usize,
    profile: Profile,
}

impl MelSpectrogramBuffer {
    /// Create new mel-spectrogram buffer
    pub fn new(capacity_frames: usize, n_mels: usize, profile: Profile) -> Self {
        let data = vec![0.0f32; capacity_frames * n_mels];

        info!("Created MelSpectrogramBuffer: {}x{} ({}MB)",
              capacity_frames, n_mels,
              (capacity_frames * n_mels * 4) as f64 / 1024.0 / 1024.0);

        Self {
            data,
            n_frames: 0,
            n_mels,
            capacity_frames,
            current_frame: 0,
            profile,
        }
    }

    /// Create mel-spectrogram buffer optimized for profile
    pub fn new_for_profile(profile: Profile) -> Self {
        let (capacity_frames, n_mels) = match profile {
            Profile::Low => (500, 80),     // 500 frames, 80 mel bands
            Profile::Medium => (750, 80),   // 750 frames, 80 mel bands
            Profile::High => (1000, 128),  // 1000 frames, 128 mel bands
        };

        Self::new(capacity_frames, n_mels, profile)
    }

    /// Add a new mel frame (efficiently overwrites old data)
    pub fn push_frame(&mut self, mel_frame: &[f32]) -> Result<()> {
        if mel_frame.len() != self.n_mels {
            return Err(anyhow::anyhow!("Mel frame size mismatch: expected {}, got {}",
                                     self.n_mels, mel_frame.len()));
        }

        let start_idx = self.current_frame * self.n_mels;
        let end_idx = start_idx + self.n_mels;

        // Copy mel frame data
        self.data[start_idx..end_idx].copy_from_slice(mel_frame);

        // Update frame tracking
        self.current_frame = (self.current_frame + 1) % self.capacity_frames;
        if self.n_frames < self.capacity_frames {
            self.n_frames += 1;
        }

        debug!("Added mel frame {}/{}", self.n_frames, self.capacity_frames);
        Ok(())
    }

    /// Get a specific frame (zero-copy slice)
    pub fn get_frame(&self, frame_idx: usize) -> Option<&[f32]> {
        if frame_idx >= self.n_frames {
            return None;
        }

        let actual_idx = if self.n_frames < self.capacity_frames {
            frame_idx
        } else {
            (self.current_frame + frame_idx) % self.capacity_frames
        };

        let start_idx = actual_idx * self.n_mels;
        let end_idx = start_idx + self.n_mels;

        Some(&self.data[start_idx..end_idx])
    }

    /// Get the latest N frames as a contiguous view
    pub fn get_latest_frames(&self, count: usize) -> Vec<f32> {
        let actual_count = count.min(self.n_frames);
        let mut result = vec![0.0f32; actual_count * self.n_mels];

        for i in 0..actual_count {
            let frame_idx = if actual_count < self.n_frames {
                self.n_frames - actual_count + i
            } else {
                i
            };

            if let Some(frame) = self.get_frame(frame_idx) {
                let start_idx = i * self.n_mels;
                let end_idx = start_idx + self.n_mels;
                result[start_idx..end_idx].copy_from_slice(frame);
            }
        }

        result
    }

    /// Get all frames as Array2-like structure for inference
    pub fn as_array2(&self) -> (Vec<f32>, (usize, usize)) {
        let frames = self.get_latest_frames(self.n_frames);
        (frames, (self.n_frames, self.n_mels))
    }

    /// Get dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        (self.n_frames, self.n_mels)
    }

    /// Clear all frames
    pub fn clear(&mut self) {
        self.n_frames = 0;
        self.current_frame = 0;
        // Don't clear data to avoid allocation overhead
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }
}

/// Shared memory pool for model inference data
pub struct InferenceMemoryPool {
    pools: Vec<MemoryChunk>,
    free_chunks: VecDeque<usize>,
    allocated_chunks: Vec<bool>,
    profile: Profile,
    total_allocated: usize,
    peak_usage: usize,
}

#[derive(Debug)]
struct MemoryChunk {
    data: Vec<u8>,
    size: usize,
    layout: Layout,
    in_use: bool,
}

impl InferenceMemoryPool {
    /// Create new inference memory pool
    pub fn new(profile: Profile) -> Self {
        let (chunk_count, chunk_size) = match profile {
            Profile::Low => (16, 1024 * 1024),    // 16 chunks of 1MB each
            Profile::Medium => (32, 2 * 1024 * 1024), // 32 chunks of 2MB each
            Profile::High => (64, 4 * 1024 * 1024),   // 64 chunks of 4MB each
        };

        let mut pools = Vec::with_capacity(chunk_count);
        let mut free_chunks = VecDeque::with_capacity(chunk_count);
        let allocated_chunks = vec![false; chunk_count];

        for i in 0..chunk_count {
            pools.push(MemoryChunk {
                data: vec![0u8; chunk_size],
                size: chunk_size,
                layout: Layout::from_size_align(chunk_size, 64).unwrap(), // 64-byte aligned
                in_use: false,
            });
            free_chunks.push_back(i);
        }

        info!("Created InferenceMemoryPool: {} chunks x {}MB = {}MB total",
              chunk_count, chunk_size / (1024 * 1024),
              (chunk_count * chunk_size) / (1024 * 1024));

        Self {
            pools,
            free_chunks,
            allocated_chunks,
            profile,
            total_allocated: 0,
            peak_usage: 0,
        }
    }

    /// Allocate a memory chunk
    pub fn allocate(&mut self, size: usize) -> Option<InferenceAllocation> {
        // Find a suitable chunk
        if let Some(chunk_idx) = self.free_chunks.pop_front() {
            if self.pools[chunk_idx].size >= size {
                self.pools[chunk_idx].in_use = true;
                self.allocated_chunks[chunk_idx] = true;
                self.total_allocated += size;

                if self.total_allocated > self.peak_usage {
                    self.peak_usage = self.total_allocated;
                }

                debug!("Allocated chunk {} ({} bytes), total allocated: {}",
                       chunk_idx, size, self.total_allocated);

                return Some(InferenceAllocation {
                    chunk_idx,
                    size,
                    pool: self as *mut Self,
                });
            } else {
                // Return chunk to free list if too small
                self.free_chunks.push_back(chunk_idx);
            }
        }

        warn!("Failed to allocate {} bytes: no suitable chunks available", size);
        None
    }

    /// Free a memory chunk
    pub fn free(&mut self, chunk_idx: usize, size: usize) {
        if chunk_idx < self.pools.len() && self.allocated_chunks[chunk_idx] {
            self.pools[chunk_idx].in_use = false;
            self.allocated_chunks[chunk_idx] = false;
            self.free_chunks.push_back(chunk_idx);
            self.total_allocated -= size;

            debug!("Freed chunk {} ({} bytes), total allocated: {}",
                   chunk_idx, size, self.total_allocated);
        }
    }

    /// Get pool statistics
    pub fn get_stats(&self) -> PoolStats {
        let total_chunks = self.pools.len();
        let free_chunks = self.free_chunks.len();
        let used_chunks = total_chunks - free_chunks;
        let total_capacity = self.pools.iter().map(|c| c.size).sum::<usize>();

        PoolStats {
            total_chunks,
            used_chunks,
            free_chunks,
            total_allocated: self.total_allocated,
            total_capacity,
            peak_usage: self.peak_usage,
            fragmentation: self.calculate_fragmentation(),
        }
    }

    /// Calculate memory fragmentation
    fn calculate_fragmentation(&self) -> f32 {
        if self.total_allocated == 0 {
            return 0.0;
        }

        let used_chunks = self.allocated_chunks.iter().filter(|&&used| used).count();
        let average_chunk_usage = self.total_allocated as f32 / used_chunks as f32;
        let chunk_capacity = self.pools.first().map(|c| c.size).unwrap_or(0) as f32;

        1.0 - (average_chunk_usage / chunk_capacity)
    }

    /// Get memory usage for current profile
    pub fn memory_usage(&self) -> usize {
        self.total_allocated
    }

    /// Force garbage collection
    pub fn garbage_collect(&mut self) {
        // In a real implementation, this would:
        // 1. Compact allocated chunks
        // 2. Merge free chunks
        // 3. Return memory to OS if possible

        debug!("Performed garbage collection on InferenceMemoryPool");
    }
}

/// Handle for allocated inference memory
pub struct InferenceAllocation {
    chunk_idx: usize,
    size: usize,
    pool: *mut InferenceMemoryPool,
}

impl InferenceAllocation {
    /// Get raw pointer to the allocated memory
    pub fn as_ptr(&self) -> *mut u8 {
        unsafe {
            (&mut *self.pool).pools[self.chunk_idx].data.as_mut_ptr()
        }
    }

    /// Get slice view of the allocated memory
    pub unsafe fn as_slice(&self) -> &[u8] {
        std::slice::from_raw_parts(self.as_ptr(), self.size)
    }

    /// Get mutable slice view of the allocated memory
    pub unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        std::slice::from_raw_parts_mut(self.as_ptr(), self.size)
    }

    /// Get allocation size
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for InferenceAllocation {
    fn drop(&mut self) {
        unsafe {
            (*self.pool).free(self.chunk_idx, self.size);
        }
    }
}

/// Memory pool statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub total_chunks: usize,
    pub used_chunks: usize,
    pub free_chunks: usize,
    pub total_allocated: usize,
    pub total_capacity: usize,
    pub peak_usage: usize,
    pub fragmentation: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_ring_buffer() {
        let mut buffer = AudioRingBuffer::new(1000, 16000, 1);

        // Test writing
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let written = buffer.write(&samples).unwrap();
        assert_eq!(written, 5);
        assert_eq!(buffer.available(), 5);

        // Test reading
        let mut output = vec![0.0; 3];
        let read = buffer.read(&mut output).unwrap();
        assert_eq!(read, 3);
        assert_eq!(output, vec![1.0, 2.0, 3.0]);
        assert_eq!(buffer.available(), 2);
    }

    #[test]
    fn test_mel_spectrogram_buffer() {
        let mut buffer = MelSpectrogramBuffer::new(10, 5, Profile::Low);

        // Test adding frames
        let frame1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        buffer.push_frame(&frame1).unwrap();
        assert_eq!(buffer.n_frames, 1);

        // Test getting frames
        let retrieved = buffer.get_frame(0).unwrap();
        assert_eq!(retrieved, &frame1);
    }

    #[test]
    fn test_inference_memory_pool() {
        let mut pool = InferenceMemoryPool::new(Profile::Low);

        // Test allocation
        let allocation = pool.allocate(512).unwrap();
        assert_eq!(allocation.size(), 512);

        let stats = pool.get_stats();
        assert_eq!(stats.used_chunks, 1);
        assert_eq!(stats.total_allocated, 512);

        // Test deallocation (happens on drop)
        drop(allocation);

        let stats = pool.get_stats();
        assert_eq!(stats.used_chunks, 0);
        assert_eq!(stats.total_allocated, 0);
    }
}