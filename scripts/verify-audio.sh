#!/usr/bin/env bash
# Verify that the TTS API produces real audio, not empty/silent WAV files.
# Usage: ./scripts/verify-audio.sh [host:port]
set -euo pipefail

HOST="${1:-localhost:3100}"
OUT="/tmp/tts-test-output.wav"

echo "=== Qwen3-TTS Audio Verification ==="
echo "Target: http://$HOST"
echo ""

# 1. Health check
echo "[1/5] Health check..."
if ! curl -sf "http://$HOST/health" > /dev/null 2>&1; then
    echo "FAIL: Server not responding at http://$HOST/health"
    exit 1
fi
echo "  OK: Server is healthy"

# 2. Check available speakers
echo "[2/5] Checking speakers..."
SPEAKERS=$(curl -sf "http://$HOST/api/speakers")
echo "  Speakers: $SPEAKERS"

# 3. Generate audio via OpenAI-compatible endpoint
echo "[3/5] Generating audio via /v1/audio/speech..."
HTTP_CODE=$(curl -sf -o "$OUT" -w "%{http_code}" \
    -X POST "http://$HOST/v1/audio/speech" \
    -H "Content-Type: application/json" \
    -d '{"model":"tts-1","input":"Hello world. This is a test of the text to speech system.","voice":"vivian"}')

if [ "$HTTP_CODE" != "200" ]; then
    echo "FAIL: HTTP $HTTP_CODE"
    cat "$OUT" 2>/dev/null
    exit 1
fi
echo "  OK: HTTP 200"

# 4. Check file size and WAV header
echo "[4/5] Checking WAV file..."
FILE_SIZE=$(stat -c%s "$OUT")
echo "  File size: $FILE_SIZE bytes"

if [ "$FILE_SIZE" -lt 1000 ]; then
    echo "FAIL: File too small ($FILE_SIZE bytes) — likely empty or error response"
    xxd "$OUT" | head -5
    exit 1
fi

# Check WAV header (RIFF....WAVE)
HEADER=$(xxd -l 12 -p "$OUT")
if [[ "$HEADER" != 52494646*57415645 ]] && [[ "${HEADER:0:8}" != "52494646" ]]; then
    echo "WARN: Doesn't look like a WAV file. Header: $HEADER"
fi

# 5. Check for actual audio content (not all zeros)
echo "[5/5] Analyzing audio samples..."

# Use python to check if samples are non-zero
python3 -c "
import wave, struct, sys

with wave.open('$OUT', 'rb') as w:
    params = w.getparams()
    n_frames = params.nframes
    sample_rate = params.framerate
    n_channels = params.nchannels
    sample_width = params.sampwidth
    duration = n_frames / sample_rate

    print(f'  Sample rate: {sample_rate} Hz')
    print(f'  Channels: {n_channels}')
    print(f'  Sample width: {sample_width} bytes')
    print(f'  Frames: {n_frames}')
    print(f'  Duration: {duration:.2f}s')

    if n_frames == 0:
        print('FAIL: Zero frames in WAV')
        sys.exit(1)

    if duration < 0.1:
        print(f'FAIL: Audio too short ({duration:.3f}s)')
        sys.exit(1)

    # Read all samples and check for silence
    raw = w.readframes(n_frames)
    if sample_width == 2:
        fmt = f'<{n_frames * n_channels}h'
        samples = struct.unpack(fmt, raw)
    elif sample_width == 4:
        fmt = f'<{n_frames * n_channels}f'
        samples = struct.unpack(fmt, raw)
    else:
        print(f'WARN: Unusual sample width {sample_width}, skipping sample analysis')
        sys.exit(0)

    # Check statistics
    max_val = max(abs(s) for s in samples)
    nonzero = sum(1 for s in samples if s != 0)
    nonzero_pct = nonzero / len(samples) * 100

    print(f'  Max amplitude: {max_val}')
    print(f'  Non-zero samples: {nonzero}/{len(samples)} ({nonzero_pct:.1f}%)')

    if max_val == 0:
        print('FAIL: ALL SAMPLES ARE ZERO — audio is completely silent!')
        sys.exit(1)

    if nonzero_pct < 5:
        print(f'WARN: Only {nonzero_pct:.1f}% non-zero samples — mostly silent')
        sys.exit(1)

    # RMS energy
    rms = (sum(s*s for s in samples) / len(samples)) ** 0.5
    if sample_width == 2:
        rms_db = 20 * __import__('math').log10(max(rms / 32768, 1e-10))
    else:
        rms_db = 20 * __import__('math').log10(max(rms, 1e-10))
    print(f'  RMS energy: {rms:.1f} ({rms_db:.1f} dB)')

    print('')
    print('PASS: Audio contains real content!')
    print(f'  Output saved to: $OUT')
" || {
    echo "FAIL: Audio analysis failed"
    exit 1
}

echo ""
echo "=== Verification Complete ==="
