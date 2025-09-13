# OBS Live Translator Documentation

## Overview

OBS Live Translator is a real-time AI translation system designed for streamers, educators, and content creators. It provides low-latency translation with emotion detection, running entirely on local hardware.

## Table of Contents

- [Getting Started](./getting-started.md)
- [Installation Guide](./installation.md)
- [Configuration](./configuration.md)
- [Performance Optimization](./OPTIMIZATION_GUIDE.md)
- [GPU Requirements](./GPU_REQUIREMENTS.md)
- [VRAM Usage Guide](./VRAM_USAGE_GUIDE.md)
- [Integrated GPU Performance](./INTEGRATED_GPU_PERFORMANCE.md)
- [API Reference](./api-reference.md)
- [Troubleshooting](./troubleshooting.md)

## Key Features

### Real-Time Translation
- Low-latency processing suitable for live streaming
- Support for multiple languages
- Automatic language detection

### Emotion Analysis
- Real-time emotion detection
- Tone analysis for better context
- Visual feedback through particle effects

### Hardware Optimization
- Runs on consumer-grade GPUs (2GB+ VRAM)
- Optimized for both discrete and integrated GPUs
- Automatic hardware detection and configuration

### Privacy First
- 100% local processing
- No cloud dependencies
- No data collection

## System Requirements

### Minimum Requirements
- **GPU**: 2GB VRAM (integrated or discrete)
- **RAM**: 8GB system memory
- **CPU**: 4-core processor
- **OS**: Windows 10/11, macOS 10.15+, Linux (Ubuntu 20.04+)

### Recommended Requirements
- **GPU**: 4GB+ VRAM discrete GPU
- **RAM**: 16GB system memory
- **CPU**: 6+ core processor
- **Network**: Stable connection for OBS streaming

## Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/obs-live-translator
cd obs-live-translator
```

2. **Run optimization script**
```bash
./scripts/optimize.sh --auto-detect
```

3. **Build and run**
```bash
cargo run --release
```

4. **Access control panel**
```
http://localhost:8080/control_panel.html
```

## Architecture

The system consists of several key components:

- **Audio Pipeline**: Captures and processes audio from OBS
- **Whisper Model**: Performs speech-to-text conversion
- **NLLB Model**: Handles language translation
- **Emotion Analyzer**: Detects emotions and tone
- **WebSocket Server**: Broadcasts translations to clients
- **Web Interface**: Provides control and visualization

## Performance

Performance varies based on hardware:

| GPU Type | Expected Latency | VRAM Usage |
|----------|-----------------|------------|
| RTX 4090 | Low | 4-6GB |
| RTX 4070 | Low | 3-4GB |
| RTX 4060 | Medium | 2-3GB |
| Integrated | High | 1-2GB |

## Contributing

We welcome contributions! Please see our [Contributing Guide](../CONTRIBUTING.md) for details.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/obs-live-translator/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/obs-live-translator/discussions)
- **Documentation**: [Full Docs](https://yourusername.github.io/obs-live-translator)

## Acknowledgments

- OpenAI Whisper team for speech recognition models
- Meta AI for NLLB translation models
- OBS Studio community for the streaming platform
- Open source contributors