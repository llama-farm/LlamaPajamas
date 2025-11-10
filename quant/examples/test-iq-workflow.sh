#!/bin/bash
# Test IQ quantization workflow end-to-end
# Uses small model (Qwen3-0.6B) for quick testing

set -e

echo "=========================================="
echo "IQ Quantization Workflow Test"
echo "=========================================="
echo ""

# Configuration
MODEL="Qwen/Qwen3-0.6B"
TEST_DIR="./test-iq-workflow"
CALIBRATION_FILE="$TEST_DIR/calibration.txt"
BASE_MODEL_DIR="$TEST_DIR/qwen3-0.6b"
IQ_OUTPUT_DIR="$TEST_DIR/qwen3-0.6b/gguf/IQ2_XS"
IMATRIX_FILE="$TEST_DIR/qwen3-0.6b.imatrix"

# Cleanup previous test
if [ -d "$TEST_DIR" ]; then
    echo "Removing previous test directory..."
    rm -rf "$TEST_DIR"
fi

mkdir -p "$TEST_DIR"

# Step 1: Generate calibration data
echo "Step 1/4: Generating calibration data..."
uv run python -m llama_pajamas_quant.cli.main iq generate-calibration \
    --output "$CALIBRATION_FILE" \
    --num-samples 100

echo "✅ Calibration data generated: $CALIBRATION_FILE"
echo ""

# Step 2: Quantize base model to Q4_K_M
echo "Step 2/4: Quantizing base model to Q4_K_M..."
uv run python -m llama_pajamas_quant.cli.main quantize llm \
    --model "$MODEL" \
    --formats gguf \
    --gguf-precision Q4_K_M \
    --output "$BASE_MODEL_DIR" \
    --no-benchmark

echo "✅ Base model quantized"
echo ""

# Find the GGUF file
BASE_GGUF=$(find "$BASE_MODEL_DIR/gguf/Q4_K_M" -name "*.gguf" | head -1)
echo "Base GGUF: $BASE_GGUF"
echo ""

# Step 3: Generate importance matrix
echo "Step 3/4: Generating importance matrix..."
uv run python -m llama_pajamas_quant.cli.main iq generate-matrix \
    --model "$BASE_GGUF" \
    --calibration "$CALIBRATION_FILE" \
    --output "$IMATRIX_FILE" \
    --n-ctx 2048 \
    --n-chunks 50

echo "✅ Importance matrix generated: $IMATRIX_FILE"
echo ""

# Step 4: Quantize to IQ2_XS
echo "Step 4/4: Quantizing to IQ2_XS with importance matrix..."
uv run python -m llama_pajamas_quant.cli.main iq quantize \
    --model "$BASE_GGUF" \
    --imatrix "$IMATRIX_FILE" \
    --precision IQ2_XS \
    --output "$IQ_OUTPUT_DIR"

echo "✅ IQ2_XS quantization complete"
echo ""

# Summary
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""
echo "Files created:"
echo "  Calibration: $CALIBRATION_FILE"
echo "  Base model: $BASE_GGUF"
echo "  IMatrix: $IMATRIX_FILE"
echo "  IQ2_XS model: $IQ_OUTPUT_DIR/model-iq2_xs.gguf"
echo ""

# Show sizes
echo "File sizes:"
ls -lh "$CALIBRATION_FILE"
ls -lh "$BASE_GGUF"
ls -lh "$IMATRIX_FILE"
ls -lh "$IQ_OUTPUT_DIR"/*.gguf

echo ""
echo "Test directory: $TEST_DIR"
echo "To clean up: rm -rf $TEST_DIR"
