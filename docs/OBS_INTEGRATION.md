# OBS Studio Integration Guide

This guide provides comprehensive instructions for integrating OBS Live Translator with OBS Studio for seamless real-time translation during streaming.

## Quick Start

### Automated Setup (Recommended)

```bash
# Run automated setup
cargo run --bin obs-setup auto

# Interactive setup with progress
cargo run --bin obs-setup interactive
```

### Manual Setup

1. **Add Browser Source in OBS:**
   - URL: `http://localhost:8080/overlay`
   - Width: 1920, Height: 1080
   - Enable hardware acceleration

2. **Configure Audio:**
   - Ensure microphone permissions
   - Set sample rate to 48kHz in OBS

3. **Enable WebSocket:**
   - Tools → WebSocket Server Settings
   - Port: 4455 (default)

## Integration Features

- Real-time translation overlay
- Voice preservation technology
- Cultural adaptation
- Hardware acceleration support
- Multi-language support (1,107 languages)

## Troubleshooting

Run diagnostics:
```bash
cargo run --bin obs-setup check
cargo run --bin obs-setup test
```

For detailed documentation, see the full integration guide.