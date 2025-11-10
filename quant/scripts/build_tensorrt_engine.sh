#!/bin/bash
#
# Build TensorRT engine from ONNX using Docker
#
# This script uses NVIDIA's TensorRT Docker image to build engines
# even on non-NVIDIA hardware (like Apple Silicon).
#
# Usage:
#   ./build_tensorrt_engine.sh model.onnx output.engine fp16
#   ./build_tensorrt_engine.sh model.onnx output.engine int8

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input.onnx> <output.engine> [fp16|int8]"
    echo ""
    echo "Examples:"
    echo "  $0 yolov8n.onnx yolov8n-fp16.engine fp16"
    echo "  $0 yolov8n.onnx yolov8n-int8.engine int8"
    exit 1
fi

INPUT_ONNX="$1"
OUTPUT_ENGINE="$2"
PRECISION="${3:-fp16}"

echo "============================================================"
echo "Building TensorRT Engine"
echo "============================================================"
echo "Input:     $INPUT_ONNX"
echo "Output:    $OUTPUT_ENGINE"
echo "Precision: $PRECISION"
echo "============================================================"
echo ""

# Check if input exists
if [ ! -f "$INPUT_ONNX" ]; then
    echo "❌ Input ONNX file not found: $INPUT_ONNX"
    exit 1
fi

# Get absolute paths
INPUT_ABS=$(realpath "$INPUT_ONNX")
OUTPUT_ABS=$(realpath "$OUTPUT_ENGINE" 2>/dev/null || echo "$(pwd)/$OUTPUT_ENGINE")
OUTPUT_DIR=$(dirname "$OUTPUT_ABS")
OUTPUT_FILE=$(basename "$OUTPUT_ABS")

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build trtexec command
TRTEXEC_CMD="trtexec --onnx=/workspace/input.onnx --saveEngine=/workspace/output.engine"

# Add precision flags
case "$PRECISION" in
    fp16)
        TRTEXEC_CMD="$TRTEXEC_CMD --fp16"
        ;;
    int8)
        TRTEXEC_CMD="$TRTEXEC_CMD --int8 --best"
        ;;
    fp32)
        # No flags needed for FP32
        ;;
    *)
        echo "❌ Unknown precision: $PRECISION (use fp16, int8, or fp32)"
        exit 1
        ;;
esac

echo "🐳 Starting TensorRT Docker container..."
echo ""

# Run TensorRT in Docker
docker run --rm \
    -v "$INPUT_ABS:/workspace/input.onnx:ro" \
    -v "$OUTPUT_DIR:/workspace" \
    nvcr.io/nvidia/tensorrt:23.12-py3 \
    bash -c "$TRTEXEC_CMD && mv /workspace/output.engine /workspace/$OUTPUT_FILE"

# Check if engine was created
if [ -f "$OUTPUT_ABS" ]; then
    ENGINE_SIZE=$(du -h "$OUTPUT_ABS" | cut -f1)
    echo ""
    echo "============================================================"
    echo "✅ TensorRT Engine Built Successfully!"
    echo "============================================================"
    echo "Output: $OUTPUT_ABS"
    echo "Size:   $ENGINE_SIZE"
    echo ""
    echo "💡 Deploy this engine to NVIDIA GPU with TensorRT runtime"
    echo "============================================================"
else
    echo ""
    echo "❌ Failed to build TensorRT engine"
    exit 1
fi
