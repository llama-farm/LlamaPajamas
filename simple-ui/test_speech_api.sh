#!/bin/bash
# Test speech API with a test audio file

set -e

echo "======================================================================"
echo "Testing Speech Transcription API"
echo "======================================================================"

# Create a test audio file (5 seconds of silence at 16kHz)
TEST_AUDIO="/tmp/test_audio_$(uuidgen).wav"
echo "Creating test audio file: $TEST_AUDIO"
ffmpeg -f lavfi -i "sine=frequency=440:duration=1" -ar 16000 -ac 1 "$TEST_AUDIO" -y 2>/dev/null

echo ""
echo "Test: Whisper-tiny CoreML Transcription"
echo "----------------------------------------------------------------------"

curl -N -X POST http://localhost:3001/api/speech/transcribe \
  -F "modelPath=../quant/models/whisper-tiny/coreml/int8/encoder.mlpackage" \
  -F "backend=coreml" \
  -F "audio=@${TEST_AUDIO}" \
  2>&1 | while IFS= read -r line; do
    if [[ $line == data:* ]]; then
      # Remove "data: " prefix and parse JSON
      json="${line#data: }"
      echo "$json" | python3 -m json.tool 2>/dev/null || echo "$json"
    fi
  done

# Clean up
rm -f "$TEST_AUDIO"

echo ""
echo "======================================================================"
echo "Speech API Tests Complete"
echo "======================================================================"
