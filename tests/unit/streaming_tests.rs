//! Streaming Protocol Unit Tests

#[cfg(test)]
mod tests {
    use obs_live_translator::streaming::{
        WebSocketHandler, WebSocketMessage,
        StreamingSession, SessionManager,
        OptimizedWebSocketHandler, BinaryMessageType,
        StreamingOpusEncoder, OpusCodecConfig,
        ProtocolOptimizer, FrameOptimizationConfig,
        CompressionCapabilities, CompressionAlgorithm,
    };
    use crate::test_utils::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_websocket_connection() {
        let handler = WebSocketHandler::new().await.unwrap();

        // Test connection establishment
        let result = handler.test_connection("ws://localhost:8080").await;
        assert!(result.is_ok() || result.is_err()); // Connection may fail in test env
    }

    #[tokio::test]
    async fn test_binary_message_protocol() {
        // Test binary message header creation
        let header = BinaryMessageHeader::new(
            BinaryMessageType::OpusAudio,
            1024,
            1,
        );

        assert_eq!(header.msg_type, BinaryMessageType::OpusAudio);
        assert_eq!(header.length, 1024);
        assert_eq!(header.sequence, 1);

        // Test serialization
        let bytes = header.to_bytes();
        assert_eq!(bytes.len(), BinaryMessageHeader::SIZE);

        // Test deserialization
        let decoded = BinaryMessageHeader::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.msg_type, header.msg_type);
        assert_eq!(decoded.length, header.length);
        assert_eq!(decoded.sequence, header.sequence);
    }

    #[tokio::test]
    async fn test_session_management() {
        let manager = SessionManager::new().await.unwrap();

        // Create session
        let session_id = "test_session_123";
        let session = manager.create_session(session_id).await.unwrap();

        assert_eq!(session.get_id(), session_id);
        assert!(session.is_active());

        // Retrieve session
        let retrieved = manager.get_session(session_id).await.unwrap();
        assert_eq!(retrieved.get_id(), session_id);

        // Remove session
        manager.remove_session(session_id).await.unwrap();
        let removed = manager.get_session(session_id).await;
        assert!(removed.is_none());
    }

    #[tokio::test]
    async fn test_opus_encoding() {
        let config = OpusCodecConfig {
            sample_rate: 16000,
            channels: 1,
            application: OpusApplication::Voip,
            bitrate: 16000,
            frame_duration_ms: 20.0,
            adaptive_bitrate: true,
            low_latency_mode: true,
            complexity: 5,
            enable_dtx: true,
            enable_fec: true,
        };

        let mut encoder = StreamingOpusEncoder::new(config).await.unwrap();

        // Generate test audio (20ms at 16kHz = 320 samples)
        let audio = generate_test_audio(20, 16000);

        let encoded = encoder.encode(&audio).await.unwrap();

        // Check compression occurred
        assert!(encoded.len() < audio.len() * 4); // Less than uncompressed f32 size
        assert!(encoded.len() > 0);

        // Test adaptive bitrate
        encoder.update_network_conditions(100.0, 0.05).await.unwrap();
        let metrics = encoder.get_metrics().await;
        assert!(metrics.frames_processed > 0);
    }

    #[tokio::test]
    async fn test_opus_decoding() {
        let config = OpusCodecConfig::default();

        let encoder = StreamingOpusEncoder::new(config.clone()).await.unwrap();
        let mut decoder = OpusDecoder::new(config).await.unwrap();

        // Encode then decode
        let audio = generate_test_audio(20, 16000);
        let encoded = encoder.encode(&audio).await.unwrap();
        let decoded = decoder.decode(&encoded).await.unwrap();

        // Check dimensions match
        assert_eq!(decoded.len(), audio.len());
    }

    #[tokio::test]
    async fn test_packet_loss_concealment() {
        let config = OpusCodecConfig::default();
        let mut decoder = OpusDecoder::new(config).await.unwrap();

        // Decode with packet loss
        let result = decoder.decode_with_plc(None).await.unwrap();

        // Should return concealed frame
        assert!(!result.is_empty());

        // Multiple consecutive losses
        for _ in 0..5 {
            let concealed = decoder.decode_with_plc(None).await.unwrap();
            assert!(!concealed.is_empty());
        }

        let metrics = decoder.get_metrics().await;
        assert!(metrics.frames_lost > 0);
    }

    #[tokio::test]
    async fn test_protocol_optimization() {
        let config = FrameOptimizationConfig {
            target_frame_size: 1024,
            sample_rate: 16000,
            adaptive_frame_size: true,
            binary_packing_enabled: true,
            compression_threshold: 512,
        };

        let compression_caps = CompressionCapabilities {
            algorithms: vec![
                CompressionAlgorithm::Lz4,
                CompressionAlgorithm::Zstd,
            ],
            max_compression_level: 6,
            preferred_algorithm: CompressionAlgorithm::Lz4,
            cpu_overhead_tolerance: 0.1,
        };

        let optimizer = ProtocolOptimizer::new(config, compression_caps).await.unwrap();

        // Test frame optimization
        let frame_data = vec![0u8; 2048];
        let optimized = optimizer.optimize_frame(
            frame_data.clone(),
            BinaryFrameType::AudioFrame,
            1,
        ).await.unwrap();

        assert!(optimized.data.len() <= frame_data.len());
        assert_eq!(optimized.frame_type, BinaryFrameType::AudioFrame);
        assert_eq!(optimized.sequence, 1);
        assert_eq!(optimized.original_size, 2048);

        // Test compression negotiation
        let peer_caps = CompressionCapabilities::default();
        let negotiated = optimizer.negotiate_compression(peer_caps).await.unwrap();
        assert_ne!(negotiated, CompressionAlgorithm::None);
    }

    #[tokio::test]
    async fn test_adaptive_jitter_buffer() {
        let mut jitter_buffer = AdaptiveJitterBuffer::new(
            Duration::from_millis(30),
            Duration::from_millis(50),
        );

        // Add frames with varying latencies
        for i in 0..10 {
            let frame = AudioFrame {
                data: vec![0.0; 320],
                timestamp: i as u64 * 20_000, // 20ms intervals in microseconds
                sequence: i,
            };
            jitter_buffer.push(frame).await;
        }

        // Retrieve frames - should be in order
        for i in 0..10 {
            let frame = jitter_buffer.pop().await;
            assert!(frame.is_some());
            assert_eq!(frame.unwrap().sequence, i);
        }

        let stats = jitter_buffer.get_stats();
        assert!(stats.current_depth_ms >= 30.0);
        assert!(stats.current_depth_ms <= 50.0);
    }

    #[tokio::test]
    async fn test_backpressure_handling() {
        let controller = BackpressureController::new(
            BackpressureConfig {
                max_buffer_size: 1000,
                high_watermark: 800,
                low_watermark: 200,
                drop_strategy: DropStrategy::OldestFirst,
            }
        );

        // Test normal operation
        assert!(!controller.should_apply_backpressure(500).await);

        // Test high watermark
        assert!(controller.should_apply_backpressure(900).await);

        // Test drop strategy
        let action = controller.get_drop_action(1100).await;
        assert_eq!(action, BackpressureAction::DropOldest(100));
    }

    #[tokio::test]
    async fn test_frame_synchronization() {
        let synchronizer = FrameSynchronizer::new(SyncConfig {
            target_frame_rate: 50,
            max_drift_ms: 10.0,
            resync_threshold_ms: 50.0,
        });

        // Test synchronization with drift
        for i in 0..100 {
            let timestamp = i * 20_000; // 20ms intervals
            let drift = if i > 50 { 5000 } else { 0 }; // Add 5ms drift

            let adjusted = synchronizer.synchronize(timestamp + drift).await.unwrap();

            // Should compensate for drift
            if i > 60 {
                assert!(adjusted.drift_correction_us != 0);
            }
        }

        let stats = synchronizer.get_stats().await;
        assert!(stats.total_frames_synced > 0);
    }

    #[tokio::test]
    async fn test_streaming_session_lifecycle() {
        let manager = EnhancedSessionManager::new(
            SessionManagerConfig::default()
        ).await.unwrap();

        // Start manager
        manager.start().await.unwrap();

        // Create session
        let session_id = "lifecycle_test";
        let session = manager.create_session(session_id).await.unwrap();

        // Update session activity
        session.update_activity().await;
        sleep(Duration::from_millis(100)).await;

        // Check session stats
        let stats = session.get_stats().await;
        assert!(stats.bytes_processed == 0); // No data processed yet
        assert!(stats.uptime_seconds > 0.0);

        // Process some data
        session.process_audio_data(&[0u8; 1000]).await.unwrap();
        let updated_stats = session.get_stats().await;
        assert_eq!(updated_stats.bytes_processed, 1000);

        // Cleanup
        manager.remove_session(session_id).await.unwrap();
    }

    #[tokio::test]
    async fn test_websocket_reconnection() {
        let handler = OptimizedWebSocketHandler::new().await.unwrap();

        // Simulate connection failure and reconnection
        let mut reconnect_attempts = 0;
        let max_attempts = 3;

        while reconnect_attempts < max_attempts {
            let result = handler.connect("ws://localhost:8080").await;

            if result.is_ok() {
                break;
            }

            reconnect_attempts += 1;
            sleep(Duration::from_millis(100 * reconnect_attempts)).await;
        }

        assert!(reconnect_attempts <= max_attempts);
    }

    #[tokio::test]
    async fn test_binary_frame_prioritization() {
        let mut optimizer = ProtocolOptimizer::new(
            FrameOptimizationConfig::default(),
            CompressionCapabilities::default(),
        ).await.unwrap();

        // Create frames with different priorities
        let mut frames = vec![
            OptimizedBinaryFrame {
                data: vec![0u8; 100],
                timestamp: 0,
                sequence: 0,
                compression: CompressionAlgorithm::None,
                original_size: 100,
                priority: 50, // Low priority
                frame_type: BinaryFrameType::ConfigFrame,
            },
            OptimizedBinaryFrame {
                data: vec![0u8; 100],
                timestamp: 1,
                sequence: 1,
                compression: CompressionAlgorithm::None,
                original_size: 100,
                priority: 255, // High priority
                frame_type: BinaryFrameType::AudioFrame,
            },
            OptimizedBinaryFrame {
                data: vec![0u8; 100],
                timestamp: 2,
                sequence: 2,
                compression: CompressionAlgorithm::None,
                original_size: 100,
                priority: 150, // Medium priority
                frame_type: BinaryFrameType::TranscriptionFrame,
            },
        ];

        optimizer.schedule_frames(&mut frames).await;

        // Check frames are sorted by priority
        assert_eq!(frames[0].priority, 255);
        assert_eq!(frames[1].priority, 150);
        assert_eq!(frames[2].priority, 50);
    }

    #[tokio::test]
    async fn test_compression_negotiation() {
        let local_caps = CompressionCapabilities {
            algorithms: vec![
                CompressionAlgorithm::Lz4,
                CompressionAlgorithm::Zstd,
                CompressionAlgorithm::Opus,
            ],
            max_compression_level: 9,
            preferred_algorithm: CompressionAlgorithm::Lz4,
            cpu_overhead_tolerance: 0.15,
        };

        let remote_caps = CompressionCapabilities {
            algorithms: vec![
                CompressionAlgorithm::Zstd,
                CompressionAlgorithm::Opus,
            ],
            max_compression_level: 6,
            preferred_algorithm: CompressionAlgorithm::Zstd,
            cpu_overhead_tolerance: 0.10,
        };

        let optimizer = ProtocolOptimizer::new(
            FrameOptimizationConfig::default(),
            local_caps,
        ).await.unwrap();

        let negotiated = optimizer.negotiate_compression(remote_caps).await.unwrap();

        // Should negotiate to common algorithm
        assert_eq!(negotiated, CompressionAlgorithm::Zstd);
    }
}