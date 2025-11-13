#!/bin/bash
# Test vision API with horse image

set -e

echo "======================================================================"
echo "Testing Vision Inference API"
echo "======================================================================"

# Test image
IMAGE_PATH="/Users/robthelen/Downloads/test-images/horse.jpg"

# Test 1: YOLO Detection (CoreML)
echo ""
echo "Test 1: YOLO-v8n Detection (CoreML)"
echo "----------------------------------------------------------------------"

curl -N -X POST http://localhost:3001/api/vision/inference \
  -F "modelPath=../quant/models/yolo-v8n/coreml/fp16/model.mlpackage" \
  -F "backend=coreml" \
  -F "taskType=detection" \
  -F "image=@${IMAGE_PATH}" \
  2>&1 | while IFS= read -r line; do
    if [[ $line == data:* ]]; then
      # Remove "data: " prefix and parse JSON
      json="${line#data: }"
      echo "$json" | python3 -m json.tool 2>/dev/null || echo "$json"
    fi
  done

# Test 2: ViT Classification (CoreML)
echo ""
echo "Test 2: ViT-base Classification (CoreML)"
echo "----------------------------------------------------------------------"

curl -N -X POST http://localhost:3001/api/vision/inference \
  -F "modelPath=../quant/models/vit-base/coreml/fp16/model.mlpackage" \
  -F "backend=coreml" \
  -F "taskType=classification" \
  -F "image=@${IMAGE_PATH}" \
  2>&1 | while IFS= read -r line; do
    if [[ $line == data:* ]]; then
      json="${line#data: }"
      echo "$json" | python3 -m json.tool 2>/dev/null || echo "$json"
    fi
  done

# Test 3: CLIP Localization (CoreML)
echo ""
echo "Test 3: CLIP-ViT Localization (CoreML)"
echo "----------------------------------------------------------------------"

curl -N -X POST http://localhost:3001/api/vision/inference \
  -F "modelPath=../quant/models/clip-vit-base/coreml/fp16/model.mlpackage" \
  -F "backend=coreml" \
  -F "taskType=localization" \
  -F "image=@${IMAGE_PATH}" \
  2>&1 | while IFS= read -r line; do
    if [[ $line == data:* ]]; then
      json="${line#data: }"
      echo "$json" | python3 -m json.tool 2>/dev/null || echo "$json"
    fi
  done

echo ""
echo "======================================================================"
echo "Vision API Tests Complete"
echo "======================================================================"
