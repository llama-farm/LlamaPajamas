#!/bin/bash
# Convert Mermaid diagrams to PNG images

# Install mermaid-cli if not installed
if ! command -v mmdc &> /dev/null; then
    echo "Installing mermaid-cli..."
    npm install -g @mermaid-js/mermaid-cli
fi

# Create images directory
mkdir -p docs/images

# Extract and convert each mermaid diagram
echo "Converting Mermaid diagrams to PNG..."

# Diagram 1: Complete Pipeline
cat > /tmp/diagram1.mmd << 'EOF'
flowchart TB
    subgraph Input["📥 Input Models"]
        HF[Hugging Face Models<br/>Qwen, Llama, Whisper, YOLO]
    end

    subgraph Quant["🔧 QUANTIZATION PIPELINE<br/>(Offline, Heavy)"]
        direction TB
        Q1[Model Detection<br/>Auto-detect architecture]
        Q2[Format Conversion<br/>GGUF • MLX • CoreML<br/>ONNX • TensorRT]
        Q3[Compression<br/>4-bit • 3-bit • 2-bit<br/>INT8 • FP16]
        Q1 --> Q2 --> Q3
    end

    subgraph Output["💾 Quantized Models"]
        direction LR
        O1[GGUF<br/>Q4_K_M: 4.7GB<br/>IQ2_XS: 2.4GB]
        O2[MLX<br/>4-bit: 4.3GB<br/>2-bit: 2.4GB]
        O3[CoreML<br/>INT8: 3.1MB<br/>FP16: 6.2MB]
        O4[ONNX<br/>INT8: 3.3MB<br/>FP32: 12.2MB]
    end

    subgraph Runtime["⚡ RUNTIME SYSTEM<br/>(Online, Fast)"]
        direction TB
        R1[Load Model<br/>Select backend]
        R2[Inference<br/>Generate/Detect/Transcribe]
        R3[Hardware Acceleration<br/>GPU • ANE • CPU]
        R1 --> R2 --> R3
    end

    subgraph Deploy["🚀 DEPLOYMENT"]
        direction LR
        D1[Local<br/>Mac • Windows<br/>Linux]
        D2[Cloud<br/>AWS • GCP<br/>Azure]
        D3[Edge<br/>Raspberry Pi<br/>Jetson • Mobile]
    end

    HF --> Quant
    Quant --> Output
    Output --> O1 & O2 & O3 & O4
    O1 & O2 & O3 & O4 --> Runtime
    Runtime --> Deploy
    Deploy --> D1 & D2 & D3

    style Quant fill:#e1f5ff
    style Runtime fill:#fff3e0
    style Deploy fill:#f1f8e9
EOF

mmdc -i /tmp/diagram1.mmd -o docs/images/pipeline-overview.png -b transparent

echo "✅ Converted: pipeline-overview.png"

# You can add more conversions here...

echo ""
echo "All diagrams converted to docs/images/"
echo "Update OVERVIEW.md to use: ![Pipeline](docs/images/pipeline-overview.png)"
