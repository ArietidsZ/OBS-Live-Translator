# OBS Live Translator API Documentation

## Base URL
```
http://localhost:8080
```

## Health & Monitoring Endpoints

### GET /health
Returns the overall health status of the service.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "uptime_seconds": 3600,
  "version": "2.0.0",
  "components": [
    {
      "name": "StreamProcessor",
      "status": "healthy",
      "latency_ms": 5.2,
      "error_rate": 0.01
    }
  ],
  "metrics": {
    "active_streams": 5,
    "cpu_usage_percent": 45.2,
    "memory_usage_mb": 2048,
    "gpu_usage_percent": 60.5,
    "cache_hit_rate": 0.85
  }
}
```

### GET /readiness
Kubernetes readiness probe endpoint.

### GET /liveness
Kubernetes liveness probe endpoint.

### GET /metrics
Prometheus metrics endpoint.

## Stream Management

### POST /streams
Create a new translation stream.

**Request Body:**
```json
{
  "stream_id": "stream_123",
  "source_language": "ja",
  "target_language": "en",
  "priority": "high",
  "settings": {
    "enable_vad": true,
    "enable_noise_reduction": true,
    "max_latency_ms": 100
  }
}
```

**Response:**
```json
{
  "stream_id": "stream_123",
  "status": "active",
  "websocket_url": "ws://localhost:8080/ws/stream_123"
}
```

### GET /streams/{stream_id}
Get stream information and statistics.

### DELETE /streams/{stream_id}
Stop and remove a stream.

### PUT /streams/{stream_id}/priority
Update stream priority.

**Request Body:**
```json
{
  "priority": "critical"
}
```

## WebSocket Endpoints

### WS /ws/{stream_id}
Real-time audio streaming and translation.

**Client → Server (Binary):**
- Raw audio data (PCM format, 16kHz, mono)

**Server → Client (JSON):**
```json
{
  "type": "transcription",
  "timestamp": "2024-01-01T00:00:00Z",
  "original_text": "こんにちは",
  "translated_text": "Hello",
  "confidence": 0.95,
  "language_detected": "ja"
}
```

## Cache Management

### GET /cache/stats
Get cache statistics.

**Response:**
```json
{
  "total_entries": 1000,
  "total_size_mb": 512,
  "hit_rate": 0.85,
  "miss_rate": 0.15,
  "eviction_count": 50
}
```

### POST /cache/warm
Pre-warm cache with common phrases.

**Request Body:**
```json
{
  "phrases": [
    "Hello world",
    "Thank you"
  ],
  "languages": ["en", "ja", "zh"]
}
```

### DELETE /cache
Clear the entire cache.

## Resource Management

### GET /resources
Get current resource allocation and usage.

**Response:**
```json
{
  "cpu": {
    "allocated_cores": 4,
    "usage_percent": 45.2
  },
  "memory": {
    "allocated_mb": 4096,
    "used_mb": 2048
  },
  "gpu": {
    "allocated_mb": 2048,
    "used_mb": 1024,
    "utilization_percent": 60.5
  }
}
```

### PUT /resources/optimize
Optimize resource allocation.

**Request Body:**
```json
{
  "optimization_target": "latency" // or "throughput"
}
```

## Performance Monitoring

### GET /monitoring/metrics
Get detailed performance metrics.

### GET /monitoring/alerts
Get active alerts.

### POST /monitoring/alerts/clear
Clear all alerts.

## Configuration

### GET /config
Get current configuration.

### PUT /config
Update configuration (requires restart).

**Request Body:**
```json
{
  "streaming": {
    "max_concurrent_streams": 30,
    "default_priority": "normal"
  },
  "cache": {
    "max_size_mb": 4096,
    "eviction_policy": "lru"
  }
}
```

## Error Responses

All endpoints may return error responses:

```json
{
  "error": {
    "code": "RESOURCE_EXHAUSTED",
    "message": "Maximum concurrent streams limit reached",
    "details": {
      "current_streams": 20,
      "max_streams": 20
    }
  }
}
```

### Error Codes
- `INVALID_REQUEST` - Malformed request
- `NOT_FOUND` - Resource not found
- `RESOURCE_EXHAUSTED` - Resource limits exceeded
- `INTERNAL_ERROR` - Server error
- `SERVICE_UNAVAILABLE` - Service temporarily unavailable

## Rate Limiting

- Default: 1000 requests per minute per IP
- WebSocket connections: 100 concurrent per IP
- Stream creation: 10 per minute per IP

## Authentication (Optional)

When authentication is enabled:

**Header:**
```
Authorization: Bearer <token>
```

## SDK Examples

### JavaScript/TypeScript
```javascript
const translator = new OBSLiveTranslator({
  host: 'localhost',
  port: 8080
});

const stream = await translator.createStream({
  sourceLanguage: 'ja',
  targetLanguage: 'en',
  priority: 'high'
});

stream.on('translation', (data) => {
  console.log(`${data.original_text} → ${data.translated_text}`);
});

stream.sendAudio(audioBuffer);
```

### Python
```python
from obs_translator import Client

client = Client(host='localhost', port=8080)
stream = client.create_stream(
    source_language='ja',
    target_language='en'
)

for translation in stream.translations():
    print(f"{translation.original} → {translation.translated}")
```