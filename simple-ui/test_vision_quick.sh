#!/bin/bash
# Quick vision API test - just check for errors and results

echo "Testing Vision API (YOLO Detection)..."

curl -N -X POST http://localhost:3001/api/vision/inference \
  -F "modelPath=../quant/models/yolo-v8n/coreml/fp16/model.mlpackage" \
  -F "backend=coreml" \
  -F "taskType=detection" \
  -F "image=@/Users/robthelen/Downloads/test-images/horse.jpg" \
  2>&1 | while IFS= read -r line; do
    if [[ $line == data:* ]]; then
      json="${line#data: }"
      # Check for errors
      if echo "$json" | grep -q '"error"'; then
        echo "❌ ERROR: $json"
        exit 1
      fi
      # Check for result (without printing massive base64)
      if echo "$json" | grep -q '"result"' && echo "$json" | grep -q '"detections"'; then
        # Extract just the detection count
        detections=$(echo "$json" | python3 -c "import json,sys; data=json.load(sys.stdin); print(len(data.get('result', {}).get('detections', [])))" 2>/dev/null || echo "?")
        echo "✅ SUCCESS: Found $detections detections"
        exit 0
      fi
      # Show progress
      if echo "$json" | grep -q '"progress"'; then
        progress=$(echo "$json" | python3 -c "import json,sys; print(json.load(sys.stdin).get('progress', ''))" 2>/dev/null)
        if [[ ! "$progress" =~ ^[A-Za-z0-9+/=]{100,} ]]; then  # Skip base64 noise
          echo "  $progress"
        fi
      fi
      if echo "$json" | grep -q '"status":"complete"'; then
        echo "✅ Test completed"
        exit 0
      fi
    fi
  done
