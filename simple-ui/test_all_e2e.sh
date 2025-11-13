#!/bin/bash
# Comprehensive end-to-end test suite for all inference modes

set -e

echo "========================================================================"
echo "END-TO-END TEST SUITE - LLama Pajamas Simple UI"
echo "========================================================================"
echo ""
echo "Testing all three inference modes:"
echo "  1. Speech Transcription (CoreML Whisper)"
echo "  2. Vision Inference (CoreML YOLO, ViT, CLIP)"
echo "  3. LLM Chat (GGUF Qwen)"
echo ""
echo "========================================================================"

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASSED=0
FAILED=0

# Test 1: Speech Transcription
echo ""
echo "========================================================================"
echo "TEST 1: Speech Transcription API"
echo "========================================================================"

TEST_AUDIO="/tmp/test_e2e_speech.wav"
ffmpeg -f lavfi -i "sine=frequency=440:duration=1" -ar 16000 -ac 1 "$TEST_AUDIO" -y 2>/dev/null

SPEECH_RESULT=$(curl -N -X POST http://localhost:3001/api/speech/transcribe \
  -F "modelPath=../quant/models/whisper-tiny/coreml/int8/encoder.mlpackage" \
  -F "backend=coreml" \
  -F "audio=@${TEST_AUDIO}" \
  2>&1 | grep -o '"result"' | head -1)

rm -f "$TEST_AUDIO"

if [ ! -z "$SPEECH_RESULT" ]; then
  echo -e "${GREEN}✅ PASS${NC} - Speech transcription completed"
  PASSED=$((PASSED + 1))
else
  echo -e "${RED}❌ FAIL${NC} - Speech transcription failed"
  FAILED=$((FAILED + 1))
fi

# Test 2: Vision Inference - YOLO Detection
echo ""
echo "========================================================================"
echo "TEST 2: Vision Inference API - YOLO Detection"
echo "========================================================================"

VISION_RESULT=$(curl -N -X POST http://localhost:3001/api/vision/inference \
  -F "modelPath=../quant/models/yolo-v8n/coreml/fp16/model.mlpackage" \
  -F "backend=coreml" \
  -F "taskType=detection" \
  -F "image=@/Users/robthelen/Downloads/test-images/horse.jpg" \
  2>&1 | grep -o '"status":"complete"' | head -1)

if [ ! -z "$VISION_RESULT" ]; then
  echo -e "${GREEN}✅ PASS${NC} - Vision detection completed"
  PASSED=$((PASSED + 1))
else
  echo -e "${RED}❌ FAIL${NC} - Vision detection failed"
  FAILED=$((FAILED + 1))
fi

# Test 3: Vision Inference - ViT Classification
echo ""
echo "========================================================================"
echo "TEST 3: Vision Inference API - ViT Classification"
echo "========================================================================"

VIT_RESULT=$(curl -N -X POST http://localhost:3001/api/vision/inference \
  -F "modelPath=../quant/models/vit-base/coreml/fp16/model.mlpackage" \
  -F "backend=coreml" \
  -F "taskType=classification" \
  -F "image=@/Users/robthelen/Downloads/test-images/horse.jpg" \
  2>&1 | grep -o '"status":"complete"' | head -1)

if [ ! -z "$VIT_RESULT" ]; then
  echo -e "${GREEN}✅ PASS${NC} - ViT classification completed"
  PASSED=$((PASSED + 1))
else
  echo -e "${RED}❌ FAIL${NC} - ViT classification failed"
  FAILED=$((FAILED + 1))
fi

# Test 4: LLM Chat Inference
echo ""
echo "========================================================================"
echo "TEST 4: LLM Chat API - GGUF Model"
echo "========================================================================"

# Check if qwen model exists
if [ -f "../quant/models/qwen/gguf/q4/qwen-0_5b-instruct-q4_k_m.gguf" ]; then
  CHAT_RESULT=$(curl -N -X POST http://localhost:3001/api/chat \
    -H "Content-Type: application/json" \
    -d '{
      "modelPath": "../quant/models/qwen/gguf/q4/qwen-0_5b-instruct-q4_k_m.gguf",
      "backend": "gguf",
      "prompt": "Say hello in one word.",
      "systemPrompt": "You are a helpful assistant.",
      "maxTokens": 10,
      "temperature": 0.7
    }' \
    2>&1 | grep -o '"status":"complete"' | head -1)

  if [ ! -z "$CHAT_RESULT" ]; then
    echo -e "${GREEN}✅ PASS${NC} - LLM chat completed"
    PASSED=$((PASSED + 1))
  else
    echo -e "${RED}❌ FAIL${NC} - LLM chat failed"
    FAILED=$((FAILED + 1))
  fi
else
  echo -e "${YELLOW}⊘ SKIP${NC} - Qwen GGUF model not found"
fi

# Summary
echo ""
echo "========================================================================"
echo "TEST SUMMARY"
echo "========================================================================"
echo -e "Passed: ${GREEN}${PASSED}${NC}"
echo -e "Failed: ${RED}${FAILED}${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
  echo -e "${GREEN}✅ ALL TESTS PASSED${NC}"
  echo "========================================================================"
  exit 0
else
  echo -e "${RED}❌ SOME TESTS FAILED${NC}"
  echo "========================================================================"
  exit 1
fi
