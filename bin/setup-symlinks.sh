#!/bin/bash
# Setup symlinks for commonly used llama.cpp tools

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BIN_DIR="$SCRIPT_DIR"
LLAMA_BIN="$SCRIPT_DIR/../libs/llama.cpp/build/bin"

echo "Setting up symlinks in $BIN_DIR"

# Check if llama.cpp binaries exist
if [ ! -d "$LLAMA_BIN" ]; then
    echo "❌ Error: llama.cpp binaries not found at $LLAMA_BIN"
    echo "Please build llama.cpp first:"
    echo "  cd quant"
    echo "  uv run python scripts/build_llama_cpp.py"
    exit 1
fi

# Core quantization tools
echo "Creating symlinks for core quantization tools..."
ln -sf "$LLAMA_BIN/llama-imatrix" "$BIN_DIR/llama-imatrix"
ln -sf "$LLAMA_BIN/llama-quantize" "$BIN_DIR/llama-quantize"

# Inference/testing tools
echo "Creating symlinks for inference tools..."
ln -sf "$LLAMA_BIN/llama-cli" "$BIN_DIR/llama-cli"
ln -sf "$LLAMA_BIN/llama-server" "$BIN_DIR/llama-server"
ln -sf "$LLAMA_BIN/llama-perplexity" "$BIN_DIR/llama-perplexity"
ln -sf "$LLAMA_BIN/llama-bench" "$BIN_DIR/llama-bench"

# Utilities
echo "Creating symlinks for utilities..."
ln -sf "$LLAMA_BIN/llama-gguf" "$BIN_DIR/llama-gguf"
ln -sf "$LLAMA_BIN/llama-gguf-split" "$BIN_DIR/llama-gguf-split"

echo ""
echo "✅ Successfully created symlinks in bin/"
echo ""
echo "You can now run:"
echo "  ./bin/llama-imatrix"
echo "  ./bin/llama-quantize"
echo "  ./bin/llama-cli"
echo ""
echo "Or add to PATH:"
echo "  source bin/setup-env.sh"
echo "  llama-imatrix --help"
