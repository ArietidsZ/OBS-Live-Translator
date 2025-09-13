# Multi-stage build for OBS Live Translator
# Stage 1: Builder with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder

# Install Rust and build dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    cmake \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

# Set working directory
WORKDIR /app

# Copy dependency files
COPY Cargo.toml Cargo.lock ./
COPY package.json package-lock.json ./

# Build dependencies separately for caching
RUN cargo fetch
RUN npm ci --only=production

# Copy source code
COPY src ./src
COPY tests ./tests
COPY benches ./benches

# Build release binary with all features
RUN cargo build --release --features "cuda tensorrt acceleration"

# Stage 2: Runtime image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    libgomp1 \
    libnuma1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js runtime
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1000 translator
USER translator

# Set working directory
WORKDIR /home/translator/app

# Copy built artifacts from builder
COPY --from=builder --chown=translator:translator /app/target/release/libobs_live_translator_core.so ./lib/
COPY --from=builder --chown=translator:translator /app/node_modules ./node_modules
COPY --chown=translator:translator package.json ./
COPY --chown=translator:translator config ./config

# Environment variables
ENV RUST_LOG=info
ENV OBS_TRANSLATOR_PORT=8080
ENV OBS_TRANSLATOR_HOST=0.0.0.0
ENV CUDA_VISIBLE_DEVICES=0
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD node -e "require('http').get('http://localhost:8080/health', (r) => process.exit(r.statusCode === 200 ? 0 : 1))"

# Expose ports
EXPOSE 8080 8081

# Start command
CMD ["node", "src/server.js"]