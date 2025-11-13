#!/bin/bash
# Quick speech API test

set -e

echo "Testing Speech API..."

# Create a short test audio (1 second tone)
TEST_AUDIO="/tmp/test_speech_quick.wav"
ffmpeg -f lavfi -i "sine=frequency=440:duration=1" -ar 16000 -ac 1 "$TEST_AUDIO" -y 2>/dev/null

echo "Sending audio to API..."

curl -N -X POST http://localhost:3001/api/speech/transcribe \
  -F "modelPath=../quant/models/whisper-tiny/coreml/int8/encoder.mlpackage" \
  -F "backend=coreml" \
  -F "audio=@${TEST_AUDIO}" \
  2>&1 | while IFS= read -r line; do
    if [[ $line == data:* ]]; then
      json="${line#data: }"
      # Check for errors
      if echo "$json" | grep -q '"error"'; then
        echo "❌ ERROR: $json"
        rm -f "$TEST_AUDIO"
        exit 1
      fi
      # Check for result
      if echo "$json" | grep -q '"result"' && echo "$json" | grep -q '"text"'; then
        text=$(echo "$json" | python3 -c "import json,sys; data=json.load(sys.stdin); print(data.get('result', {}).get('text', 'N/A'))" 2>/dev/null || echo "N/A")
        confidence=$(echo "$json" | python3 -c "import json,sys; data=json.load(sys.stdin); print(data.get('result', {}).get('confidence', 'N/A'))" 2>/dev/null || echo "N/A")
        echo "✅ SUCCESS!"
        echo "   Text: $text"
        echo "   Confidence: $confidence"
        rm -f "$TEST_AUDIO"
        exit 0
      fi
      # Show progress
      if echo "$json" | grep -q '"progress"'; then
        progress=$(echo "$json" | python3 -c "import json,sys; print(json.load(sys.stdin).get('progress', ''))" 2>/dev/null || echo "$json")
        # Skip torch warnings
        if [[ ! "$progress" =~ "Torch version" && ! "$progress" =~ "scikit-learn" ]]; then
          echo "  $progress"
        fi
      fi
      if echo "$json" | grep -q '"status":"complete"'; then
        echo "✅ Test completed"
        rm -f "$TEST_AUDIO"
        exit 0
      fi
    fi
  done

rm -f "$TEST_AUDIO"
