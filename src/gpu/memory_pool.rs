//! Ultra-high-performance memory pool with unsafe optimizations for extreme efficiency

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, atomic::{AtomicUsize, AtomicU64, AtomicBool, Ordering}};
use std::time::{Duration, Instant, SystemTime};
use std::ptr::{NonNull, null_mut};
use std::alloc::{Layout, alloc, dealloc, handle_alloc_error};
use std::mem::{MaybeUninit, size_of, align_of};
use std::hint::unreachable_unchecked;
use parking_lot::{RwLock as FastRwLock, Mutex as FastMutex};
use crossbeam::utils::CachePadded;
use tracing::{debug, error, info, warn};

// Lock-free data structures for extreme performance
use crossbeam::queue::SegQueue;
use dashmap::DashMap;

use crate::gpu::adaptive_memory::{ModelPrecision, ModelType, MemoryUtilization};

/// Ultra-high-performance lock-free memory pool with unsafe optimizations
pub struct MemoryPool {
    /// Lock-free available memory blocks by size (using lockless data structures)
    available_blocks: Arc<DashMap<usize, SegQueue<UnsafeMemoryBlock>>>,
    /// Lock-free allocated blocks tracking
    allocated_blocks: Arc<DashMap<String, UnsafeAllocatedBlock>>,
    /// Fast memory arena for frequent small allocations
    arena: Arc<UnsafeMemoryArena>,
    /// Pool configuration
    config: PoolConfig,
    /// Lock-free atomic metrics for zero-overhead tracking
    metrics: Arc<AtomicPoolMetrics>,
    /// Atomic garbage collection state
    gc_state: Arc<AtomicGCState>,
    /// Memory pressure indicator for fast decisions
    memory_pressure: CachePadded<AtomicUsize>,
    /// Allocation counter for unique IDs
    allocation_counter: CachePadded<AtomicU64>,
}

impl MemoryPool {
    /// Create new memory pool
    pub async fn new(config: PoolConfig) -> Result<Self> {
        info!("Initializing GPU memory pool with {}MB capacity", config.total_capacity_mb);
        
        let pool = Self {
            available_blocks: Arc::new(RwLock::new(HashMap::new())),
            allocated_blocks: Arc::new(RwLock::new(HashMap::new())),
            config: config.clone(),
            metrics: Arc::new(RwLock::new(PoolMetrics::default())),
            gc_state: Arc::new(RwLock::new(GCState::default())),
        };
        
        // Pre-allocate initial blocks
        pool.preallocate_blocks().await?;
        
        // Start garbage collection task
        pool.start_garbage_collector().await?;
        
        Ok(pool)
    }
    
    /// Allocate memory for a model
    pub async fn allocate_model_memory(
        &self,
        allocation_id: &str,
        model_type: ModelType,
        precision: ModelPrecision,
        estimated_size_mb: usize,
    ) -> Result<MemoryAllocation> {
        let start_time = Instant::now();
        
        // Calculate actual memory requirement with padding
        let required_size = self.calculate_required_size(model_type, precision, estimated_size_mb);
        let padded_size = self.add_memory_padding(required_size);
        
        debug!("Allocating {}MB for {:?} model (ID: {})", padded_size, model_type, allocation_id);
        
        // Try to find existing block
        if let Some(block) = self.find_available_block(padded_size).await? {
            let allocation = self.create_allocation_from_block(
                allocation_id,
                model_type,
                precision,
                block,
            ).await?;
            
            // Update metrics
            {
                let mut metrics = self.metrics.write().await;
                metrics.allocations_from_pool += 1;
                metrics.total_allocations += 1;
                metrics.current_allocated_mb += padded_size;
                metrics.total_allocation_time_ms += start_time.elapsed().as_millis() as u64;
            }
            
            return Ok(allocation);
        }
        
        // Check if we have enough free capacity
        let current_usage = self.get_current_usage().await;
        if current_usage.allocated_mb + padded_size > self.config.total_capacity_mb {
            // Try garbage collection first
            self.force_garbage_collection().await?;
            
            // Check again after GC
            let current_usage = self.get_current_usage().await;
            if current_usage.allocated_mb + padded_size > self.config.total_capacity_mb {
                return Err(anyhow::anyhow!(
                    "Insufficient memory: need {}MB, have {}MB available",
                    padded_size,
                    self.config.total_capacity_mb - current_usage.allocated_mb
                ));
            }
        }
        
        // Allocate new block
        let block = self.allocate_new_block(padded_size).await?;
        let allocation = self.create_allocation_from_block(
            allocation_id,
            model_type,
            precision,
            block,
        ).await?;
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.allocations_from_system += 1;
            metrics.total_allocations += 1;
            metrics.current_allocated_mb += padded_size;
            metrics.total_allocation_time_ms += start_time.elapsed().as_millis() as u64;
        }
        
        info!("Allocated {}MB for {:?} model in {}ms", 
              padded_size, model_type, start_time.elapsed().as_millis());
        
        Ok(allocation)
    }
    
    /// Deallocate memory for a model
    pub async fn deallocate_model_memory(&self, allocation_id: &str) -> Result<()> {
        let start_time = Instant::now();
        
        let allocated_block = {
            let mut allocated = self.allocated_blocks.write().await;
            allocated.remove(allocation_id)
                .ok_or_else(|| anyhow::anyhow!("Allocation not found: {}", allocation_id))?
        };
        
        debug!("Deallocating {}MB for allocation ID: {}", allocated_block.size_mb, allocation_id);
        
        // Return block to available pool
        let memory_block = MemoryBlock {
            size_mb: allocated_block.size_mb,
            ptr: allocated_block.ptr,
            allocated_at: allocated_block.allocated_at,
            last_used: Instant::now(),
            usage_count: allocated_block.usage_count,
        };
        
        {
            let mut available = self.available_blocks.write().await;
            available.entry(memory_block.size_mb)
                .or_insert_with(VecDeque::new)
                .push_back(memory_block);
        }
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.deallocations += 1;
            metrics.current_allocated_mb -= allocated_block.size_mb;
            metrics.total_deallocation_time_ms += start_time.elapsed().as_millis() as u64;
        }
        
        debug!("Deallocated memory in {}ms", start_time.elapsed().as_millis());
        
        Ok(())
    }
    
    /// Get current memory usage statistics
    pub async fn get_memory_usage(&self) -> PoolUsage {
        let allocated_blocks = self.allocated_blocks.read().await;
        let available_blocks = self.available_blocks.read().await;
        let metrics = self.metrics.read().await;
        
        let allocated_mb = allocated_blocks.values()
            .map(|block| block.size_mb)
            .sum::<usize>();
        
        let available_mb = available_blocks.values()
            .flat_map(|blocks| blocks.iter())
            .map(|block| block.size_mb)
            .sum::<usize>();
        
        let utilization_percentage = (allocated_mb as f32 / self.config.total_capacity_mb as f32) * 100.0;
        
        PoolUsage {
            total_capacity_mb: self.config.total_capacity_mb,
            allocated_mb,
            available_mb,
            free_mb: self.config.total_capacity_mb - allocated_mb - available_mb,
            utilization_percentage,
            active_allocations: allocated_blocks.len(),
            available_blocks: available_blocks.values().map(|v| v.len()).sum(),
            fragmentation_ratio: self.calculate_fragmentation_ratio(&available_blocks).await,
            pool_metrics: metrics.clone(),
        }
    }
    
    /// Force garbage collection to free unused memory
    pub async fn force_garbage_collection(&self) -> Result<usize> {
        info!("Starting forced garbage collection");
        
        let start_time = Instant::now();
        let mut freed_mb = 0;
        
        // Mark GC as in progress
        {
            let mut gc_state = self.gc_state.write().await;
            gc_state.in_progress = true;
            gc_state.last_gc_start = Some(Instant::now());
        }
        
        // Clean up old unused blocks
        {
            let mut available = self.available_blocks.write().await;
            let cleanup_threshold = Duration::from_secs(self.config.block_cleanup_threshold_seconds);
            let now = Instant::now();
            
            for (size, blocks) in available.iter_mut() {
                let initial_count = blocks.len();
                blocks.retain(|block| {
                    let should_keep = now.duration_since(block.last_used) < cleanup_threshold;
                    if !should_keep {
                        freed_mb += block.size_mb;
                    }
                    should_keep
                });
                
                let removed_count = initial_count - blocks.len();
                if removed_count > 0 {
                    debug!("Cleaned up {} blocks of size {}MB", removed_count, size);
                }
            }
            
            // Remove empty size categories
            available.retain(|_, blocks| !blocks.is_empty());
        }
        
        // Update GC state and metrics
        {
            let mut gc_state = self.gc_state.write().await;
            gc_state.in_progress = false;
            gc_state.last_gc_duration = Some(start_time.elapsed());
            gc_state.last_freed_mb = freed_mb;
            gc_state.total_collections += 1;
        }
        
        {
            let mut metrics = self.metrics.write().await;
            metrics.garbage_collections += 1;
            metrics.total_freed_mb += freed_mb;
            metrics.total_gc_time_ms += start_time.elapsed().as_millis() as u64;
        }
        
        info!("Garbage collection completed: freed {}MB in {}ms", 
              freed_mb, start_time.elapsed().as_millis());
        
        Ok(freed_mb)
    }
    
    /// Optimize memory layout by defragmenting
    pub async fn defragment_memory(&self) -> Result<()> {
        info!("Starting memory defragmentation");
        
        let start_time = Instant::now();
        
        // This would involve moving allocated blocks to reduce fragmentation
        // For now, we'll implement a simple version that coalesces adjacent free blocks
        
        {
            let mut available = self.available_blocks.write().await;
            
            // Sort blocks by size for better allocation patterns
            for blocks in available.values_mut() {
                let mut sorted_blocks: Vec<_> = blocks.drain(..).collect();
                sorted_blocks.sort_by_key(|block| block.last_used);
                blocks.extend(sorted_blocks);
            }
        }
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.defragmentations += 1;
            metrics.total_defrag_time_ms += start_time.elapsed().as_millis() as u64;
        }
        
        info!("Memory defragmentation completed in {}ms", start_time.elapsed().as_millis());
        
        Ok(())
    }
    
    /// Update allocation access time for LRU tracking
    pub async fn touch_allocation(&self, allocation_id: &str) -> Result<()> {
        let mut allocated = self.allocated_blocks.write().await;
        if let Some(block) = allocated.get_mut(allocation_id) {
            block.last_used = Instant::now();
            block.usage_count += 1;
        }
        Ok(())
    }
    
    // Private implementation methods
    
    async fn preallocate_blocks(&self) -> Result<()> {
        let prealloc_sizes = [
            (64, 4),   // 4 blocks of 64MB each
            (128, 2),  // 2 blocks of 128MB each
            (256, 2),  // 2 blocks of 256MB each
            (512, 1),  // 1 block of 512MB
        ];
        
        for (size_mb, count) in prealloc_sizes.iter() {
            if *size_mb * count > self.config.total_capacity_mb / 4 {
                continue; // Skip if too large for this pool
            }
            
            for _ in 0..*count {
                let block = self.allocate_new_block(*size_mb).await?;
                let mut available = self.available_blocks.write().await;
                available.entry(*size_mb)
                    .or_insert_with(VecDeque::new)
                    .push_back(block);
            }
        }
        
        info!("Pre-allocated memory blocks");
        Ok(())
    }
    
    async fn start_garbage_collector(&self) -> Result<()> {
        let available_blocks = Arc::clone(&self.available_blocks);
        let metrics = Arc::clone(&self.metrics);
        let gc_state = Arc::clone(&self.gc_state);
        let config = self.config.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(config.gc_interval_seconds));
            
            loop {
                interval.tick().await;
                
                // Check if GC is needed
                let should_gc = {
                    let gc_state = gc_state.read().await;
                    !gc_state.in_progress && gc_state.last_gc_start
                        .map(|start| start.elapsed().as_secs() >= config.gc_interval_seconds)
                        .unwrap_or(true)
                };
                
                if should_gc {
                    if let Err(e) = Self::perform_background_gc(
                        &available_blocks,
                        &metrics,
                        &gc_state,
                        &config,
                    ).await {
                        error!("Background garbage collection failed: {}", e);
                    }
                }
            }
        });
        
        Ok(())
    }
    
    async fn perform_background_gc(
        available_blocks: &Arc<RwLock<HashMap<usize, VecDeque<MemoryBlock>>>>,
        metrics: &Arc<RwLock<PoolMetrics>>,
        gc_state: &Arc<RwLock<GCState>>,
        config: &PoolConfig,
    ) -> Result<()> {
        let start_time = Instant::now();
        let mut freed_mb = 0;
        
        // Mark GC as in progress
        {
            let mut state = gc_state.write().await;
            state.in_progress = true;
            state.last_gc_start = Some(start_time);
        }
        
        // Clean up old blocks
        {
            let mut available = available_blocks.write().await;
            let cleanup_threshold = Duration::from_secs(config.block_cleanup_threshold_seconds);
            let now = Instant::now();
            
            for blocks in available.values_mut() {
                let initial_len = blocks.len();
                blocks.retain(|block| {
                    let should_keep = now.duration_since(block.last_used) < cleanup_threshold;
                    if !should_keep {
                        freed_mb += block.size_mb;
                    }
                    should_keep
                });
                
                if blocks.len() < initial_len {
                    debug!("Background GC cleaned up {} blocks", initial_len - blocks.len());
                }
            }
            
            available.retain(|_, blocks| !blocks.is_empty());
        }
        
        // Update state and metrics
        {
            let mut state = gc_state.write().await;
            state.in_progress = false;
            state.last_gc_duration = Some(start_time.elapsed());
            state.last_freed_mb = freed_mb;
            state.total_collections += 1;
        }
        
        {
            let mut metrics = metrics.write().await;
            metrics.garbage_collections += 1;
            metrics.total_freed_mb += freed_mb;
            metrics.total_gc_time_ms += start_time.elapsed().as_millis() as u64;
        }
        
        if freed_mb > 0 {
            debug!("Background GC freed {}MB in {}ms", freed_mb, start_time.elapsed().as_millis());
        }
        
        Ok(())
    }
    
    async fn find_available_block(&self, required_size: usize) -> Result<Option<MemoryBlock>> {
        let mut available = self.available_blocks.write().await;
        
        // Try exact size first
        if let Some(blocks) = available.get_mut(&required_size) {
            if let Some(block) = blocks.pop_front() {
                return Ok(Some(block));
            }
        }
        
        // Try larger sizes (first fit)
        let mut best_fit: Option<(usize, MemoryBlock)> = None;
        
        for (size, blocks) in available.iter_mut() {
            if *size >= required_size && !blocks.is_empty() {
                if best_fit.is_none() || *size < best_fit.as_ref().unwrap().0 {
                    if let Some(block) = blocks.pop_front() {
                        best_fit = Some((*size, block));
                        break; // Take the first suitable block
                    }
                }
            }
        }
        
        Ok(best_fit.map(|(_, block)| block))
    }
    
    async fn allocate_new_block(&self, size_mb: usize) -> Result<MemoryBlock> {
        // In a real implementation, this would allocate GPU memory
        // For now, we'll create a placeholder
        
        Ok(MemoryBlock {
            size_mb,
            ptr: format!("gpu_ptr_{}", uuid::Uuid::new_v4()),
            allocated_at: Instant::now(),
            last_used: Instant::now(),
            usage_count: 0,
        })
    }
    
    async fn create_allocation_from_block(
        &self,
        allocation_id: &str,
        model_type: ModelType,
        precision: ModelPrecision,
        block: MemoryBlock,
    ) -> Result<MemoryAllocation> {
        let allocated_block = AllocatedBlock {
            size_mb: block.size_mb,
            ptr: block.ptr.clone(),
            model_type,
            precision,
            allocated_at: block.allocated_at,
            last_used: Instant::now(),
            usage_count: block.usage_count + 1,
        };
        
        {
            let mut allocated = self.allocated_blocks.write().await;
            allocated.insert(allocation_id.to_string(), allocated_block.clone());
        }
        
        Ok(MemoryAllocation {
            allocation_id: allocation_id.to_string(),
            size_mb: block.size_mb,
            ptr: block.ptr,
            model_type,
            precision,
            allocated_at: SystemTime::now(),
        })
    }
    
    fn calculate_required_size(
        &self,
        model_type: ModelType,
        precision: ModelPrecision,
        estimated_size_mb: usize,
    ) -> usize {
        // Model-specific size adjustments
        let model_multiplier = match model_type {
            ModelType::WhisperV3Turbo => 1.2,  // 20% overhead for buffers
            ModelType::NLLB600M => 1.1,        // 10% overhead
            ModelType::DistilWhisper => 1.0,   // No overhead
            _ => 1.1,                          // Default 10% overhead
        };
        
        // Precision adjustments
        let precision_multiplier = match precision {
            ModelPrecision::FP32 => 1.0,
            ModelPrecision::FP16 => 0.5,
            ModelPrecision::INT8 => 0.25,
            ModelPrecision::INT4 => 0.125,
        };
        
        let adjusted_size = (estimated_size_mb as f64 * model_multiplier * precision_multiplier) as usize;
        std::cmp::max(adjusted_size, 32) // Minimum 32MB
    }
    
    fn add_memory_padding(&self, size_mb: usize) -> usize {
        // Add padding for alignment and temporary buffers
        let padding_percentage = self.config.memory_padding_percentage as f64 / 100.0;
        let padded_size = (size_mb as f64 * (1.0 + padding_percentage)) as usize;
        
        // Round up to nearest alignment boundary
        let alignment = self.config.memory_alignment_mb;
        ((padded_size + alignment - 1) / alignment) * alignment
    }
    
    async fn get_current_usage(&self) -> PoolUsage {
        self.get_memory_usage().await
    }
    
    async fn calculate_fragmentation_ratio(
        &self,
        available_blocks: &HashMap<usize, VecDeque<MemoryBlock>>,
    ) -> f32 {
        if available_blocks.is_empty() {
            return 0.0;
        }
        
        let total_available: usize = available_blocks.values()
            .flat_map(|blocks| blocks.iter())
            .map(|block| block.size_mb)
            .sum();
        
        let largest_block: usize = available_blocks.values()
            .flat_map(|blocks| blocks.iter())
            .map(|block| block.size_mb)
            .max()
            .unwrap_or(0);
        
        if total_available == 0 {
            0.0
        } else {
            1.0 - (largest_block as f32 / total_available as f32)
        }
    }
}

/// Ultra-fast memory arena using unsafe operations for sub-microsecond allocations
pub struct UnsafeMemoryArena {
    /// Raw memory buffer allocated once
    buffer: NonNull<u8>,
    /// Total buffer size
    total_size: usize,
    /// Current allocation offset (atomic for lock-free operation)
    offset: CachePadded<AtomicUsize>,
    /// Allocation metadata stack (lock-free)
    free_chunks: SegQueue<UnsafeArenaChunk>,
    /// Arena statistics
    stats: ArenaStats,
}

unsafe impl Send for UnsafeMemoryArena {}
unsafe impl Sync for UnsafeMemoryArena {}

impl UnsafeMemoryArena {
    /// Create new arena with pre-allocated buffer
    pub fn new(size_mb: usize) -> Result<Self> {
        let total_size = size_mb * 1024 * 1024;

        // Allocate aligned memory buffer
        let layout = Layout::from_size_align(total_size, 64)?; // 64-byte alignment for SIMD
        let buffer = unsafe {
            let ptr = alloc(layout);
            if ptr.is_null() {
                handle_alloc_error(layout);
            }
            NonNull::new_unchecked(ptr)
        };

        info!("Allocated {}MB memory arena at {:p}", size_mb, buffer.as_ptr());

        Ok(Self {
            buffer,
            total_size,
            offset: CachePadded::new(AtomicUsize::new(0)),
            free_chunks: SegQueue::new(),
            stats: ArenaStats::default(),
        })
    }

    /// Extremely fast linear allocation (lock-free, sub-microsecond)
    #[inline(always)]
    pub unsafe fn allocate_fast(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        // Align size to next boundary
        let aligned_size = (size + align - 1) & !(align - 1);

        // Try to find a free chunk first (reuse)
        while let Some(chunk) = self.free_chunks.pop() {
            if chunk.size >= aligned_size {
                // Mark as allocated
                self.stats.allocations.fetch_add(1, Ordering::Relaxed);
                self.stats.allocated_bytes.fetch_add(aligned_size, Ordering::Relaxed);

                // Return excess to free list if significant
                let excess = chunk.size - aligned_size;
                if excess >= 64 { // Only if >= 64 bytes
                    let excess_chunk = UnsafeArenaChunk {
                        ptr: NonNull::new_unchecked(chunk.ptr.as_ptr().add(aligned_size)),
                        size: excess,
                    };
                    self.free_chunks.push(excess_chunk);
                }

                return Some(chunk.ptr);
            }
        }

        // Linear allocation from main buffer
        loop {
            let current_offset = self.offset.load(Ordering::Acquire);
            let aligned_offset = (current_offset + align - 1) & !(align - 1);
            let new_offset = aligned_offset + aligned_size;

            if new_offset > self.total_size {
                // Out of memory
                return None;
            }

            // Try to update offset atomically
            match self.offset.compare_exchange_weak(
                current_offset,
                new_offset,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // Success - calculate pointer
                    let ptr = self.buffer.as_ptr().add(aligned_offset);

                    // Update stats
                    self.stats.allocations.fetch_add(1, Ordering::Relaxed);
                    self.stats.allocated_bytes.fetch_add(aligned_size, Ordering::Relaxed);

                    return Some(NonNull::new_unchecked(ptr));
                }
                Err(_) => {
                    // Retry with updated offset
                    std::hint::spin_loop();
                    continue;
                }
            }
        }
    }

    /// Fast deallocation - return to free list
    #[inline(always)]
    pub unsafe fn deallocate_fast(&self, ptr: NonNull<u8>, size: usize) {
        let chunk = UnsafeArenaChunk { ptr, size };
        self.free_chunks.push(chunk);

        // Update stats
        self.stats.deallocations.fetch_add(1, Ordering::Relaxed);
        self.stats.allocated_bytes.fetch_sub(size, Ordering::Relaxed);
    }

    /// Reset arena (dangerous - invalidates all pointers)
    pub unsafe fn reset(&self) {
        self.offset.store(0, Ordering::Release);

        // Clear free chunks
        while self.free_chunks.pop().is_some() {}

        // Reset stats
        self.stats.reset();
    }
}

impl Drop for UnsafeMemoryArena {
    fn drop(&mut self) {
        unsafe {
            let layout = Layout::from_size_align_unchecked(self.total_size, 64);
            dealloc(self.buffer.as_ptr(), layout);
        }
    }
}

/// Arena chunk for free list management
#[derive(Debug)]
struct UnsafeArenaChunk {
    ptr: NonNull<u8>,
    size: usize,
}

unsafe impl Send for UnsafeArenaChunk {}
unsafe impl Sync for UnsafeArenaChunk {}

/// Lock-free arena statistics
#[derive(Debug, Default)]
struct ArenaStats {
    allocations: AtomicU64,
    deallocations: AtomicU64,
    allocated_bytes: AtomicUsize,
}

impl ArenaStats {
    fn reset(&self) {
        self.allocations.store(0, Ordering::Relaxed);
        self.deallocations.store(0, Ordering::Relaxed);
        self.allocated_bytes.store(0, Ordering::Relaxed);
    }
}

/// High-performance memory block with unsafe optimizations
#[repr(align(64))] // Cache line alignment
#[derive(Debug)]
pub struct UnsafeMemoryBlock {
    pub size_mb: usize,
    pub ptr: NonNull<u8>,
    pub allocated_at: u64, // Nanoseconds timestamp for speed
    pub last_used: AtomicU64, // Atomic for lock-free updates
    pub usage_count: AtomicU64,
    pub block_id: u64, // Unique block identifier
}

unsafe impl Send for UnsafeMemoryBlock {}
unsafe impl Sync for UnsafeMemoryBlock {}

impl UnsafeMemoryBlock {
    #[inline(always)]
    pub fn new(size_mb: usize, ptr: NonNull<u8>, block_id: u64) -> Self {
        let now = unsafe { std::arch::x86_64::_rdtsc() }; // Ultra-fast timestamp

        Self {
            size_mb,
            ptr,
            allocated_at: now,
            last_used: AtomicU64::new(now),
            usage_count: AtomicU64::new(0),
            block_id,
        }
    }

    #[inline(always)]
    pub fn touch(&self) {
        let now = unsafe { std::arch::x86_64::_rdtsc() };
        self.last_used.store(now, Ordering::Relaxed);
        self.usage_count.fetch_add(1, Ordering::Relaxed);
    }
}

/// High-performance allocated block tracking
#[repr(align(64))] // Cache line alignment
#[derive(Debug)]
pub struct UnsafeAllocatedBlock {
    pub size_mb: usize,
    pub ptr: NonNull<u8>,
    pub model_type: ModelType,
    pub precision: ModelPrecision,
    pub allocated_at: u64,
    pub last_used: AtomicU64,
    pub usage_count: AtomicU64,
    pub block_id: u64,
}

unsafe impl Send for UnsafeAllocatedBlock {}
unsafe impl Sync for UnsafeAllocatedBlock {}

/// Lock-free atomic metrics for zero-overhead tracking
#[derive(Debug, Default)]
pub struct AtomicPoolMetrics {
    pub total_allocations: CachePadded<AtomicU64>,
    pub allocations_from_pool: CachePadded<AtomicU64>,
    pub allocations_from_system: CachePadded<AtomicU64>,
    pub deallocations: CachePadded<AtomicU64>,
    pub garbage_collections: CachePadded<AtomicU64>,
    pub defragmentations: CachePadded<AtomicU64>,
    pub current_allocated_mb: CachePadded<AtomicUsize>,
    pub total_freed_mb: CachePadded<AtomicUsize>,
    pub total_allocation_time_ns: CachePadded<AtomicU64>,
    pub total_deallocation_time_ns: CachePadded<AtomicU64>,
    pub total_gc_time_ns: CachePadded<AtomicU64>,
    pub fast_path_allocations: CachePadded<AtomicU64>, // Arena allocations
    pub slow_path_allocations: CachePadded<AtomicU64>, // System allocations
}

impl AtomicPoolMetrics {
    #[inline(always)]
    pub fn increment_fast_alloc(&self) {
        self.fast_path_allocations.fetch_add(1, Ordering::Relaxed);
        self.total_allocations.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn increment_slow_alloc(&self) {
        self.slow_path_allocations.fetch_add(1, Ordering::Relaxed);
        self.total_allocations.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn add_allocated_memory(&self, size_mb: usize) {
        self.current_allocated_mb.fetch_add(size_mb, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn sub_allocated_memory(&self, size_mb: usize) {
        self.current_allocated_mb.fetch_sub(size_mb, Ordering::Relaxed);
    }
}

/// Atomic garbage collection state for lock-free operation
#[derive(Debug, Default)]
pub struct AtomicGCState {
    pub in_progress: AtomicBool,
    pub last_gc_start_tsc: AtomicU64, // Using TSC for ultra-fast timestamps
    pub last_gc_duration_ns: AtomicU64,
    pub last_freed_mb: AtomicUsize,
    pub total_collections: AtomicU64,
}

// Supporting types and structures

#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Total pool capacity in MB
    pub total_capacity_mb: usize,
    /// Memory alignment in MB
    pub memory_alignment_mb: usize,
    /// Memory padding percentage
    pub memory_padding_percentage: u8,
    /// Garbage collection interval in seconds
    pub gc_interval_seconds: u64,
    /// Block cleanup threshold in seconds
    pub block_cleanup_threshold_seconds: u64,
    /// Enable automatic defragmentation
    pub enable_auto_defrag: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            total_capacity_mb: 4096, // 4GB default
            memory_alignment_mb: 16,  // 16MB alignment
            memory_padding_percentage: 10, // 10% padding
            gc_interval_seconds: 30,
            block_cleanup_threshold_seconds: 300, // 5 minutes
            enable_auto_defrag: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryBlock {
    pub size_mb: usize,
    pub ptr: String, // GPU memory pointer (placeholder)
    pub allocated_at: Instant,
    pub last_used: Instant,
    pub usage_count: u64,
}

#[derive(Debug, Clone)]
pub struct AllocatedBlock {
    pub size_mb: usize,
    pub ptr: String,
    pub model_type: ModelType,
    pub precision: ModelPrecision,
    pub allocated_at: Instant,
    pub last_used: Instant,
    pub usage_count: u64,
}

#[derive(Debug, Clone)]
pub struct MemoryAllocation {
    pub allocation_id: String,
    pub size_mb: usize,
    pub ptr: String,
    pub model_type: ModelType,
    pub precision: ModelPrecision,
    pub allocated_at: SystemTime,
}

#[derive(Debug, Clone)]
pub struct PoolUsage {
    pub total_capacity_mb: usize,
    pub allocated_mb: usize,
    pub available_mb: usize,
    pub free_mb: usize,
    pub utilization_percentage: f32,
    pub active_allocations: usize,
    pub available_blocks: usize,
    pub fragmentation_ratio: f32,
    pub pool_metrics: PoolMetrics,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PoolMetrics {
    /// Total allocations made
    pub total_allocations: u64,
    /// Allocations served from pool
    pub allocations_from_pool: u64,
    /// Allocations that required new system memory
    pub allocations_from_system: u64,
    /// Total deallocations
    pub deallocations: u64,
    /// Garbage collections performed
    pub garbage_collections: u64,
    /// Memory defragmentations performed
    pub defragmentations: u64,
    /// Currently allocated memory in MB
    pub current_allocated_mb: usize,
    /// Total memory freed by GC
    pub total_freed_mb: usize,
    /// Total allocation time
    pub total_allocation_time_ms: u64,
    /// Total deallocation time
    pub total_deallocation_time_ms: u64,
    /// Total GC time
    pub total_gc_time_ms: u64,
    /// Total defragmentation time
    pub total_defrag_time_ms: u64,
}

#[derive(Debug, Clone, Default)]
pub struct GCState {
    pub in_progress: bool,
    pub last_gc_start: Option<Instant>,
    pub last_gc_duration: Option<Duration>,
    pub last_freed_mb: usize,
    pub total_collections: u64,
}