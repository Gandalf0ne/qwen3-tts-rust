# =============================================================================
# Stage 1: Build the Rust binary
# =============================================================================
FROM rust:1-bookworm AS builder

# Build dependencies for Rust crates:
#   - pkg-config + libssl-dev: reqwest (HTTP client)
#   - libasound2-dev: cpal (audio)
#   - cmake + g++: tokenizers crate (may need C++ compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    libasound2-dev \
    cmake \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy manifests first for layer caching
COPY Cargo.toml Cargo.lock* build.rs .cargo/ ./
COPY src/ src/
COPY webui/ webui/
COPY speakers/ speakers/

# Build the server binary in release mode
RUN cargo build --release --bin qwen3_tts_server

# Find and stage the ort-managed ORT library (with WebGPU support)
RUN mkdir -p /build/ort-libs && \
    find /root/.cache/ort -name "libonnxruntime.so*" -exec cp {} /build/ort-libs/ \; && \
    find /root/.cache/ort -name "libdawn*" -exec cp {} /build/ort-libs/ \; && \
    find /root/.cache/ort -name "libwebgpu*" -exec cp {} /build/ort-libs/ \; || true

# =============================================================================
# Stage 2: Runtime image
# =============================================================================
FROM debian:bookworm-slim

# Runtime dependencies:
#   - libvulkan1 + mesa-vulkan-drivers: Vulkan ICD loader + AMD RADV driver
#   - ca-certificates + libssl3: HTTPS for model downloads on first run
#   - libasound2: ALSA runtime (cpal audio crate)
#   - libgomp1: OpenMP (used by ONNX Runtime)
#   - curl: container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libvulkan1 \
    mesa-vulkan-drivers \
    ca-certificates \
    libssl3 \
    libasound2 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json

# Working directory — critical: the app resolves "runtime/" and "models/"
# relative to CWD via hardcoded paths in the source code
WORKDIR /app

# Copy the compiled binary
COPY --from=builder /build/target/release/qwen3_tts_server /app/qwen3_tts_server

# Copy default speaker profiles (can be overridden via volume mount)
COPY --from=builder /build/speakers/ /app/speakers/

# Copy the ort-managed ORT libraries alongside the binary
COPY --from=builder /build/ort-libs/ /app/

# Tell ort where to find the WebGPU-enabled binary
ENV ORT_DYLIB_PATH=/app/libonnxruntime.so

EXPOSE 3000

# On first run, the binary auto-downloads into:
#   /app/models/   — GGUF + ONNX model files (~4-6 GB)
#   /app/runtime/  — llama.cpp shared libs (~200 MB)
# Both directories should be persistent volumes.
ENTRYPOINT ["/app/qwen3_tts_server"]
CMD ["--host", "0.0.0.0", "--port", "3000", "--model-dir", "models", "--speakers-dir", "speakers", "--threads", "8"]
