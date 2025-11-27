use ndarray::{ArrayD, IxDyn};
use obs_live_translator::{optimization::kv_cache::KvCacheManager, types::KvCacheConfig, Result};

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    tracing::info!("Starting KV Cache Manager test...");

    // Configure cache: max 10 tokens
    let config = KvCacheConfig {
        enabled: true,
        low_precision: false,
        max_cache_size: 10,
    };

    let mut manager = KvCacheManager::new(config);

    // Test 1: Add initial key/value
    tracing::info!("Test 1: Adding initial key/value (length 5)");
    // Shape: [1, 8, 5, 64] (Batch=1, Heads=8, Seq=5, Dim=64)
    let key = ArrayD::zeros(IxDyn(&[1, 8, 5, 64]));
    let value = ArrayD::zeros(IxDyn(&[1, 8, 5, 64]));

    manager.update("layer_0", key, value);

    let (k, v) = manager.get("layer_0").unwrap();
    assert_eq!(k.shape(), &[1, 8, 5, 64]);
    assert_eq!(v.shape(), &[1, 8, 5, 64]);
    tracing::info!("Test 1 passed: Cache shape correct");

    // Test 2: Append more data (length 3) -> Total 8
    tracing::info!("Test 2: Appending data (length 3) -> Total 8");
    let key_new = ArrayD::zeros(IxDyn(&[1, 8, 3, 64]));
    let value_new = ArrayD::zeros(IxDyn(&[1, 8, 3, 64]));

    manager.update("layer_0", key_new, value_new);

    let (k, _v) = manager.get("layer_0").unwrap();
    assert_eq!(k.shape(), &[1, 8, 8, 64]);
    tracing::info!("Test 2 passed: Cache grew to length 8");

    // Test 3: Append causing eviction (length 4) -> Total 12 -> Evict to 10
    tracing::info!("Test 3: Appending data (length 4) -> Total 12 -> Evict to 10");
    let key_overflow = ArrayD::zeros(IxDyn(&[1, 8, 4, 64]));
    let value_overflow = ArrayD::zeros(IxDyn(&[1, 8, 4, 64]));

    manager.update("layer_0", key_overflow, value_overflow);

    let (k, _v) = manager.get("layer_0").unwrap();
    assert_eq!(k.shape(), &[1, 8, 10, 64]);
    tracing::info!("Test 3 passed: Cache evicted correctly to max_size 10");

    Ok(())
}
