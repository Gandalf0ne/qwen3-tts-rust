#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://localhost:3100}"
VOICE="${VOICE:-vivian}"
SMOKE_TIMEOUT_SECONDS="${SMOKE_TIMEOUT_SECONDS:-1800}"
SMOKE_POLL_SECONDS="${SMOKE_POLL_SECONDS:-10}"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

log() {
  printf '[smoke] %s\n' "$*"
}

deadline=$((SECONDS + SMOKE_TIMEOUT_SECONDS))
log "Waiting for ${BASE_URL}/health (timeout ${SMOKE_TIMEOUT_SECONDS}s)"
until curl -fsS "${BASE_URL}/health" >/dev/null; do
  if (( SECONDS >= deadline )); then
    log "Timed out waiting for service readiness"
    exit 1
  fi
  sleep "${SMOKE_POLL_SECONDS}"
done
log "Health endpoint is ready"

models_json="${tmpdir}/models.json"
curl -fsS "${BASE_URL}/v1/models" -o "${models_json}"
python3 - <<'PY' "${models_json}"
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    payload = json.load(fh)

assert payload.get("object") == "list", payload
model_ids = [entry.get("id") for entry in payload.get("data", [])]
assert "tts-1" in model_ids, model_ids
print("models_ok")
PY

speakers_json="${tmpdir}/speakers.json"
curl -fsS "${BASE_URL}/api/speakers" -o "${speakers_json}"
python3 - <<'PY' "${speakers_json}" "${VOICE}"
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    payload = json.load(fh)

speakers = payload.get("speakers", [])
assert isinstance(speakers, list), payload
assert sys.argv[2] in speakers, speakers
print("speakers_ok")
PY

invalid_status="$(
  curl -sS -o "${tmpdir}/invalid.json" -w '%{http_code}' \
    -H 'Content-Type: application/json' \
    -d '{"model":"tts-1","input":"invalid voice test","voice":"not-a-real-speaker"}' \
    "${BASE_URL}/v1/audio/speech"
)"
if [[ "${invalid_status}" != "400" ]]; then
  log "Expected invalid voice request to return HTTP 400, got ${invalid_status}"
  cat "${tmpdir}/invalid.json"
  exit 1
fi
python3 - <<'PY' "${tmpdir}/invalid.json"
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    payload = json.load(fh)

assert payload["error"]["type"] == "invalid_request_error", payload
print("invalid_voice_ok")
PY

openai_status="$(
  curl -sS -o "${tmpdir}/speech.wav" -w '%{http_code}' \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"tts-1\",\"input\":\"Local CPU smoke test.\",\"voice\":\"${VOICE}\"}" \
    "${BASE_URL}/v1/audio/speech"
)"
if [[ "${openai_status}" != "200" ]]; then
  log "Expected /v1/audio/speech to return HTTP 200, got ${openai_status}"
  exit 1
fi

if [[ ! -s "${tmpdir}/speech.wav" ]]; then
  log "OpenAI speech endpoint returned an empty WAV payload"
  exit 1
fi

riff_header="$(xxd -l 4 -p "${tmpdir}/speech.wav")"
if [[ "${riff_header}" != "52494646" ]]; then
  log "OpenAI speech endpoint did not return a WAV file"
  exit 1
fi
log "OpenAI speech endpoint returned a WAV file"

legacy_status="$(
  curl -sS -o "${tmpdir}/legacy.json" -w '%{http_code}' \
    -H 'Content-Type: application/json' \
    -d "{\"text\":\"Legacy API smoke test.\",\"speaker\":\"${VOICE}\"}" \
    "${BASE_URL}/api/tts"
)"
if [[ "${legacy_status}" != "200" ]]; then
  log "Expected /api/tts to return HTTP 200, got ${legacy_status}"
  cat "${tmpdir}/legacy.json"
  exit 1
fi
python3 - <<'PY' "${tmpdir}/legacy.json"
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    payload = json.load(fh)

assert payload.get("success") is True, payload
assert payload.get("audio_base64"), payload
assert int(payload.get("sample_rate", 0)) > 0, payload
print("legacy_api_ok")
PY

log "Smoke test completed successfully"
