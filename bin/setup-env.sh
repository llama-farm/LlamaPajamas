#!/bin/bash
# Add llama-pajamas bin/ to PATH for easy access to tools

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Add to PATH
export PATH="$SCRIPT_DIR:$PATH"

echo "✅ Added $SCRIPT_DIR to PATH"
echo ""
echo "You can now run:"
echo "  llama-imatrix --help"
echo "  llama-quantize --help"
echo "  llama-cli --help"
echo ""
echo "To make this permanent, add this to your ~/.bashrc or ~/.zshrc:"
echo "  export PATH=\"$SCRIPT_DIR:\$PATH\""
