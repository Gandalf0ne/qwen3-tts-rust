# =============================================================================
# Stage 1: Build the Rust binary
# =============================================================================
# Trixie (Debian 13) required: the ort WebGPU prebuilt binary needs glibc >= 2.38
FROM rust:1-trixie AS builder

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

# Build the server binary in release mode.
# ort-sys downloads and statically links the WebGPU-enabled ORT binary.
RUN cargo build --release --bin qwen3_tts_server

# =============================================================================
# Stage 2: Runtime image
# =============================================================================
# Must match build stage's glibc (Trixie) for the statically-linked ORT binary
FROM debian:trixie-slim

# Runtime dependencies:
#   - libvulkan1 + mesa-vulkan-drivers: Vulkan ICD loader + AMD RADV driver
#   - ca-certificates + libssl3t64: HTTPS for model downloads on first run
#   - libasound2t64: ALSA runtime (cpal audio crate)
#   - libgomp1: OpenMP (used by ONNX Runtime)
#   - curl: container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libvulkan1 \
    mesa-vulkan-drivers \
    ca-certificates \
    libssl3t64 \
    libasound2t64 \
    libgomp1 \
    curl \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/app:/app/runtime
# Point Vulkan loader to the AMD RADV ICD (primary deployment target).
# On NVIDIA hosts, mount the host nvidia_icd.json via docker-compose instead.
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json
# Suppress Dawn validation noise in logs
ENV DAWN_DISABLE_VALIDATION=1

# Working directory — critical: the app resolves "runtime/" and "models/"
# relative to CWD via hardcoded paths in the source code
WORKDIR /app

# Copy the compiled binary and Dawn WebGPU shared lib (required by ort WebGPU EP)
COPY --from=builder /build/target/release/qwen3_tts_server /app/qwen3_tts_server
COPY --from=builder /build/target/release/libwebgpu_dawn.so /app/libwebgpu_dawn.so

# Copy default speaker profiles (can be overridden via volume mount)
COPY --from=builder /build/speakers/ /app/speakers/

EXPOSE 3000

# On first run, the binary auto-downloads into:
#   /app/models/   — GGUF + ONNX model files (~4-6 GB)
#   /app/runtime/  — llama.cpp shared libs (~200 MB)
# Both directories should be persistent volumes.
ENTRYPOINT ["/app/qwen3_tts_server"]
CMD ["--host", "0.0.0.0", "--port", "3000", "--model-dir", "models", "--speakers-dir", "speakers", "--threads", "8"]
