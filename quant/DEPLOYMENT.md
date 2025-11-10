# Model Packaging & Deployment Guide

Complete guide for packaging quantized models with runtimes and deploying across different environments.

## Table of Contents

1. [Packaging Models](#packaging-models)
2. [Cross-Platform Deployment](#cross-platform-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Edge Deployment](#edge-deployment)

---

## Packaging Models

### Directory Structure

After quantization, your model directory looks like:

```
models/qwen3-8b/
├── gguf/
│   ├── Q4_K_M/
│   │   └── model.gguf          # 4.68 GB - Universal (CPU/GPU)
│   ├── IQ2_XS/
│   │   └── model.gguf          # 2.40 GB - Ultra-compressed
│   └── Q3_K_M/
│       └── model.gguf          # 3.84 GB - Balanced
├── mlx/
│   ├── 4bit-mixed/             # 4.31 GB - Apple Silicon only
│   │   ├── config.json
│   │   ├── tokenizer.json
│   │   └── weights.npz
│   └── 2bit-mixed/             # 2.40 GB - Extreme compression
└── manifest.json               # Metadata
```

### Package for Target Platform

#### 1. Universal GGUF Package (CPU/GPU - Any Platform)

**Best for**: Cross-platform deployment, servers, cloud

```bash
# Create deployment package
cd quant
mkdir -p deploy/qwen3-8b-gguf

# Copy model and metadata
cp models/qwen3-8b/gguf/Q4_K_M/*.gguf deploy/qwen3-8b-gguf/
cp models/qwen3-8b/manifest.json deploy/qwen3-8b-gguf/

# Create deployment script
cat > deploy/qwen3-8b-gguf/run.py << 'EOF'
from llama_pajamas_run import RuntimeConfig, ModelLoader

config = RuntimeConfig(
    backend="gguf",
    model_path="model.gguf",
    n_gpu_layers=-1,  # Auto-detect GPU
    n_ctx=4096,
)

with ModelLoader(config) as loader:
    response = loader.generate("Hello, world!")
    print(response)
EOF

# Create requirements
cat > deploy/qwen3-8b-gguf/requirements.txt << 'EOF'
llama-cpp-python>=0.2.0
numpy>=1.24.0
EOF

# Package
cd deploy
tar -czf qwen3-8b-gguf.tar.gz qwen3-8b-gguf/
```

**Deployment** (on any Linux/Mac/Windows server):
```bash
tar -xzf qwen3-8b-gguf.tar.gz
cd qwen3-8b-gguf
pip install -r requirements.txt
python run.py
```

#### 2. MLX Package (Apple Silicon Only)

**Best for**: Mac deployment, iOS apps

```bash
# Package MLX model
mkdir -p deploy/qwen3-8b-mlx
cp -r models/qwen3-8b/mlx/4bit-mixed/ deploy/qwen3-8b-mlx/model/
cp models/qwen3-8b/manifest.json deploy/qwen3-8b-mlx/

cat > deploy/qwen3-8b-mlx/requirements.txt << 'EOF'
mlx>=0.12.0
mlx-lm>=0.12.0
EOF

tar -czf qwen3-8b-mlx.tar.gz qwen3-8b-mlx/
```

#### 3. Vision Models (CoreML + ONNX)

```bash
# Package for multiple platforms
mkdir -p deploy/yolo-v8n

# CoreML (Apple)
cp -r models/yolo-v8n/coreml/ deploy/yolo-v8n/

# ONNX (Universal)
cp -r models/yolo-v8n/onnx/ deploy/yolo-v8n/

# Metadata
cp models/yolo-v8n/manifest.json deploy/yolo-v8n/

# Create runtime-agnostic inference script
cat > deploy/yolo-v8n/detect.py << 'EOF'
import sys
import platform
from pathlib import Path

# Auto-detect platform and use appropriate backend
if platform.system() == "Darwin" and platform.machine() == "arm64":
    # Apple Silicon - use CoreML
    from llama_pajamas_run_coreml.backends.vision import CoreMLVisionBackend
    backend = CoreMLVisionBackend()
    backend.load_model("coreml/fp16/model.mlpackage", model_type="detection")
else:
    # Other platforms - use ONNX
    from llama_pajamas_run_onnx.backends.vision import ONNXVisionBackend
    backend = ONNXVisionBackend()
    backend.load_model("onnx/yolov8n.onnx", model_type="detection")

from PIL import Image
image = Image.open(sys.argv[1])
detections = backend.detect(image)
print(f"Found {len(detections)} objects")
EOF

tar -czf yolo-v8n-universal.tar.gz yolo-v8n/
```

---

## Cross-Platform Deployment

### Scenario 1: Mac to Linux Server (LLM)

**On Mac** (quantization):
```bash
cd quant

# Quantize to GGUF (universal format)
uv run python scripts/quantize_llm.py \
  --model Qwen/Qwen3-8B \
  --output ./models/qwen3-8b \
  --formats gguf \
  --gguf-precision Q4_K_M

# Package
tar -czf qwen3-8b-gguf.tar.gz models/qwen3-8b/gguf/Q4_K_M/
```

**On Linux Server** (deployment):
```bash
# Extract
tar -xzf qwen3-8b-gguf.tar.gz

# Install runtime
pip install llama-cpp-python

# Run with GPU acceleration
python3 << 'EOF'
from llama_cpp import Llama

model = Llama(
    model_path="models/qwen3-8b/gguf/Q4_K_M/model.gguf",
    n_gpu_layers=-1,  # Use NVIDIA GPU
    n_ctx=4096,
)

response = model("Explain Docker in simple terms:", max_tokens=200)
print(response["choices"][0]["text"])
EOF
```

### Scenario 2: Mac to NVIDIA Cloud (Vision - TensorRT)

**Problem**: TensorRT engines must be built on target GPU architecture

**Solution**: Build TensorRT on target, ship ONNX from Mac

**On Mac** (export to ONNX):
```bash
cd quant

# Export to ONNX (universal intermediate format)
uv run python scripts/export_model.py \
  --model yolov8n \
  --backend onnx \
  --precision fp16 \
  --output models/yolo-v8n/

# Package ONNX
tar -czf yolo-v8n-onnx.tar.gz models/yolo-v8n/onnx/
```

**On NVIDIA Cloud** (build TensorRT engine with Docker):

```bash
# Extract ONNX
tar -xzf yolo-v8n-onnx.tar.gz

# Method 1: Build with Docker (RECOMMENDED)
docker run --gpus all -v $(pwd):/workspace \
  nvcr.io/nvidia/tensorrt:23.12-py3 \
  trtexec \
    --onnx=/workspace/models/yolo-v8n/onnx/yolov8n.onnx \
    --saveEngine=/workspace/yolov8n_fp16.engine \
    --fp16 \
    --workspace=4096 \
    --verbose

# Method 2: Build with docker-compose (for automation)
cat > docker-compose.build.yml << 'EOF'
version: '3.8'
services:
  build-tensorrt:
    image: nvcr.io/nvidia/tensorrt:23.12-py3
    volumes:
      - ./models:/workspace/models
      - ./engines:/workspace/engines
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: >
      trtexec
      --onnx=/workspace/models/yolo-v8n/onnx/yolov8n.onnx
      --saveEngine=/workspace/engines/yolov8n_fp16.engine
      --fp16
      --workspace=4096
      --verbose
EOF

docker-compose -f docker-compose.build.yml up

# Method 3: Interactive container for testing
docker run --gpus all -it -v $(pwd):/workspace \
  nvcr.io/nvidia/tensorrt:23.12-py3 bash

# Inside container:
cd /workspace
trtexec --onnx=models/yolo-v8n/onnx/yolov8n.onnx \
        --saveEngine=yolov8n_fp16.engine \
        --fp16

# View engine info
trtexec --loadEngine=yolov8n_fp16.engine --verbose
```

**Automated Build Script** (`build_tensorrt_docker.sh`):
```bash
#!/bin/bash
# Build TensorRT engine using Docker (works on any NVIDIA GPU)

ONNX_FILE=$1
OUTPUT_ENGINE=$2
PRECISION=${3:-fp16}

if [ -z "$ONNX_FILE" ] || [ -z "$OUTPUT_ENGINE" ]; then
    echo "Usage: $0 <onnx_file> <output_engine> [fp16|int8|fp32]"
    exit 1
fi

echo "🔨 Building TensorRT engine..."
echo "   ONNX: $ONNX_FILE"
echo "   Output: $OUTPUT_ENGINE"
echo "   Precision: $PRECISION"
echo ""

# Build with Docker
docker run --gpus all --rm \
  -v $(pwd):/workspace \
  nvcr.io/nvidia/tensorrt:23.12-py3 \
  trtexec \
    --onnx=/workspace/$ONNX_FILE \
    --saveEngine=/workspace/$OUTPUT_ENGINE \
    --$PRECISION \
    --workspace=4096 \
    --verbose

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ TensorRT engine built successfully!"
    echo "   File: $OUTPUT_ENGINE"
    echo "   Size: $(du -h $OUTPUT_ENGINE | cut -f1)"
    echo "   GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
    echo "❌ Build failed"
    exit 1
fi
```

**Usage**:
```bash
chmod +x build_tensorrt_docker.sh

# Build FP16 engine
./build_tensorrt_docker.sh \
  models/yolo-v8n/onnx/yolov8n.onnx \
  yolov8n_fp16.engine \
  fp16

# Build INT8 engine (requires calibration)
./build_tensorrt_docker.sh \
  models/yolo-v8n/onnx/yolov8n.onnx \
  yolov8n_int8.engine \
  int8
```

**Run Inference** (using built engine):
```python
# run_tensorrt.py
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from PIL import Image

class TensorRTInference:
    def __init__(self, engine_path):
        # Load TensorRT engine
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Allocate memory
        self.inputs = []
        self.outputs = []
        self.bindings = []

        for i in range(self.engine.num_bindings):
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = self.engine.get_binding_shape(i)
            size = trt.volume(shape)

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(i):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def infer(self, image):
        # Preprocess image
        img = image.resize((640, 640))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
        img_array = np.expand_dims(img_array, axis=0)   # Add batch dim

        # Copy input to device
        np.copyto(self.inputs[0]['host'], img_array.ravel())
        cuda.memcpy_htod(self.inputs[0]['device'], self.inputs[0]['host'])

        # Run inference
        self.context.execute_v2(bindings=self.bindings)

        # Copy output to host
        cuda.memcpy_dtoh(self.outputs[0]['host'], self.outputs[0]['device'])

        return self.outputs[0]['host']

# Usage
if __name__ == "__main__":
    model = TensorRTInference("yolov8n_fp16.engine")
    image = Image.open("test.jpg")

    # Run inference
    result = model.infer(image)
    print(f"Inference complete. Output shape: {result.shape}")
```

**Docker inference container** (`Dockerfile.tensorrt-inference`):
```dockerfile
FROM nvcr.io/nvidia/tensorrt:23.12-py3

# Install additional dependencies
RUN pip install pillow opencv-python numpy

# Copy engine and inference script
WORKDIR /app
COPY yolov8n_fp16.engine /app/
COPY run_tensorrt.py /app/

# Expose port for API
EXPOSE 8001

CMD ["python3", "/app/run_tensorrt.py"]
```

**Build and run**:
```bash
# Build inference container
docker build -f Dockerfile.tensorrt-inference -t yolo-tensorrt:fp16 .

# Run
docker run --gpus all -p 8001:8001 yolo-tensorrt:fp16
```

**Complete Build Pipeline** (`Makefile`):
```makefile
# Makefile for TensorRT engine building

DOCKER_IMAGE := nvcr.io/nvidia/tensorrt:23.12-py3
MODEL_DIR := models
ENGINES_DIR := engines

.PHONY: all build-fp16 build-int8 clean

all: build-fp16

build-fp16:
	@echo "Building FP16 TensorRT engine..."
	docker run --gpus all --rm \
		-v $(PWD)/$(MODEL_DIR):/workspace/models \
		-v $(PWD)/$(ENGINES_DIR):/workspace/engines \
		$(DOCKER_IMAGE) \
		trtexec \
			--onnx=/workspace/models/yolo-v8n/onnx/yolov8n.onnx \
			--saveEngine=/workspace/engines/yolov8n_fp16.engine \
			--fp16 \
			--workspace=4096

build-int8:
	@echo "Building INT8 TensorRT engine..."
	docker run --gpus all --rm \
		-v $(PWD)/$(MODEL_DIR):/workspace/models \
		-v $(PWD)/$(ENGINES_DIR):/workspace/engines \
		$(DOCKER_IMAGE) \
		trtexec \
			--onnx=/workspace/models/yolo-v8n/onnx/yolov8n.onnx \
			--saveEngine=/workspace/engines/yolov8n_int8.engine \
			--int8 \
			--workspace=4096

clean:
	rm -rf $(ENGINES_DIR)/*.engine

info:
	docker run --gpus all --rm \
		-v $(PWD)/$(ENGINES_DIR):/workspace/engines \
		$(DOCKER_IMAGE) \
		trtexec --loadEngine=/workspace/engines/yolov8n_fp16.engine --verbose
```

**Usage**:
```bash
# Build FP16 engine
make build-fp16

# Build INT8 engine
make build-int8

# View engine info
make info

# Clean engines
make clean
```

### Scenario 3: Windows to Linux (Speech)

**On Windows** (export to ONNX):
```bash
cd quant
uv run python scripts/export_model.py \
  --model whisper-tiny \
  --backend onnx \
  --precision fp32 \
  --output models/whisper-tiny/

# Package
tar -czf whisper-tiny-onnx.tar.gz models/whisper-tiny/onnx/
```

**On Linux** (run with ONNX Runtime):
```bash
tar -xzf whisper-tiny-onnx.tar.gz

pip install onnxruntime-gpu  # or onnxruntime for CPU

python3 << 'EOF'
from llama_pajamas_run_onnx.backends.speech import ONNXSpeechBackend
from llama_pajamas_run_core.utils.audio_utils import load_audio

backend = ONNXSpeechBackend()
backend.load_model(
    encoder_path="models/whisper-tiny/onnx/tiny_encoder.onnx",
    model_name="whisper-tiny",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

audio = load_audio("audio.wav", sample_rate=16000)
result = backend.transcribe(audio)
print(result.text)
EOF
```

---

## Docker Deployment

### LLM with GGUF (Universal)

```dockerfile
# Dockerfile.llm
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install llama-cpp-python with CUDA support
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" \
    pip install llama-cpp-python

# Copy model
WORKDIR /app
COPY models/qwen3-8b/gguf/Q4_K_M/model.gguf /app/model.gguf

# Create inference script
RUN echo 'from llama_cpp import Llama\n\
model = Llama(\n\
    model_path="/app/model.gguf",\n\
    n_gpu_layers=-1,\n\
    n_ctx=4096,\n\
)\n\
\n\
import sys\n\
prompt = sys.argv[1] if len(sys.argv) > 1 else "Hello!"\n\
response = model(prompt, max_tokens=200)\n\
print(response["choices"][0]["text"])' > /app/run.py

CMD ["python3", "/app/run.py"]
```

**Build and Run**:
```bash
# Build
docker build -f Dockerfile.llm -t qwen3-8b:gguf .

# Run with GPU
docker run --gpus all qwen3-8b:gguf python3 /app/run.py "Explain Docker"

# Run as API server
docker run --gpus all -p 8000:8000 qwen3-8b:gguf \
  python3 -m llama_cpp.server \
  --model /app/model.gguf \
  --host 0.0.0.0 \
  --port 8000
```

### Vision with ONNX (TensorRT)

```dockerfile
# Dockerfile.vision
FROM nvcr.io/nvidia/tensorrt:23.12-py3

# Install dependencies
RUN pip install onnxruntime-gpu pillow opencv-python

# Copy model
WORKDIR /app
COPY models/yolo-v8n/onnx/ /app/models/

# Copy runtime
COPY run-onnx/ /app/run-onnx/
RUN pip install -e /app/run-onnx/

# Inference script
COPY docker/vision_detect.py /app/

EXPOSE 8001
CMD ["python3", "/app/vision_detect.py"]
```

### Speech with ONNX

```dockerfile
# Dockerfile.speech
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install onnxruntime numpy soundfile

WORKDIR /app
COPY models/whisper-tiny/onnx/ /app/models/
COPY run-onnx/ /app/run-onnx/
RUN pip install -e /app/run-onnx/

COPY docker/speech_transcribe.py /app/

CMD ["python3", "/app/speech_transcribe.py"]
```

### Docker Compose (Multi-Modal Stack)

```yaml
# docker-compose.yml
version: '3.8'

services:
  llm:
    build:
      context: .
      dockerfile: Dockerfile.llm
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "8000:8000"
    volumes:
      - ./models/qwen3-8b:/app/models

  vision:
    build:
      context: .
      dockerfile: Dockerfile.vision
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "8001:8001"
    volumes:
      - ./models/yolo-v8n:/app/models

  speech:
    build:
      context: .
      dockerfile: Dockerfile.speech
    ports:
      - "8002:8002"
    volumes:
      - ./models/whisper-tiny:/app/models
```

**Run the stack**:
```bash
docker-compose up -d
```

---

## Cloud Deployment

### AWS EC2 (NVIDIA GPU)

**1. Launch EC2 instance** (e.g., g5.2xlarge with NVIDIA A10G)

**2. Setup script** (`setup_ec2.sh`):
```bash
#!/bin/bash
# Install NVIDIA drivers and CUDA
sudo apt-get update
sudo apt-get install -y nvidia-driver-535 cuda-toolkit-12-1

# Install Docker with NVIDIA support
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Pull and run your container
docker pull your-registry/qwen3-8b:gguf
docker run --gpus all -p 8000:8000 your-registry/qwen3-8b:gguf
```

**3. Deploy models**:
```bash
# Upload model package
scp qwen3-8b-gguf.tar.gz ec2-user@your-instance:/home/ec2-user/

# SSH and extract
ssh ec2-user@your-instance
tar -xzf qwen3-8b-gguf.tar.gz
cd qwen3-8b-gguf

# Run
pip install -r requirements.txt
python run.py
```

### Google Cloud (Vertex AI)

```python
# vertex_deploy.py
from google.cloud import aiplatform

aiplatform.init(project="your-project", location="us-central1")

# Upload model
model = aiplatform.Model.upload(
    display_name="qwen3-8b-gguf",
    artifact_uri="gs://your-bucket/models/qwen3-8b/",
    serving_container_image_uri="gcr.io/your-project/llama-gguf:latest",
)

# Deploy endpoint
endpoint = model.deploy(
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
)
```

### Azure (Container Instances)

```bash
# Deploy to Azure Container Instances
az container create \
  --resource-group myResourceGroup \
  --name qwen3-8b \
  --image your-registry.azurecr.io/qwen3-8b:gguf \
  --gpu-count 1 \
  --gpu-sku V100 \
  --cpu 4 \
  --memory 16 \
  --ports 8000 \
  --dns-name-label qwen3-8b-api
```

---

## Edge Deployment

### Raspberry Pi / Jetson Nano (ONNX CPU)

**1. Package minimal model**:
```bash
# Use IQ2_XS for smallest size
tar -czf qwen3-1.7b-iq2xs.tar.gz models/qwen3-1.7b/gguf/IQ2_XS/
```

**2. Deploy on device**:
```bash
# Install on Raspberry Pi 4/5
sudo apt-get install python3-pip
pip3 install llama-cpp-python

# Extract and run
tar -xzf qwen3-1.7b-iq2xs.tar.gz
python3 run.py
```

**Performance**: ~5 tok/s on Raspberry Pi 5

### NVIDIA Jetson (TensorRT)

```bash
# On Jetson device
# Install JetPack (includes TensorRT)
sudo apt-get install nvidia-jetpack

# Deploy ONNX and build TensorRT
tar -xzf yolo-v8n-onnx.tar.gz

# Build for Jetson
trtexec \
  --onnx=yolov8n.onnx \
  --saveEngine=yolov8n_jetson.engine \
  --fp16 \
  --workspace=2048  # Smaller for Jetson

# Run
python3 detect.py
```

**Performance**: ~30 FPS on Jetson Orin Nano

### iOS/Android (CoreML/ONNX)

**iOS** (CoreML):
```swift
// Swift example
import CoreML

let model = try YOLOv8n(configuration: MLModelConfiguration())
let input = try YOLOv8nInput(imageWith: cgImage)
let output = try model.prediction(input: input)

// Access detections
for detection in output.coordinates {
    print("Found: \\(detection.label)")
}
```

**Android** (ONNX):
```kotlin
// Kotlin example
val env = OrtEnvironment.getEnvironment()
val session = env.createSession("yolov8n.onnx")

// Run inference
val input = OnnxTensor.createTensor(env, inputArray)
val output = session.run(mapOf("input" to input))
```

---

## Best Practices

### 1. Model Selection by Platform

| Platform | LLM | Vision | Speech |
|----------|-----|--------|--------|
| **Apple Silicon** | MLX (4-bit) | CoreML (INT8) | CoreML (INT8) |
| **NVIDIA GPU** | GGUF (Q4_K_M) | TensorRT (FP16) | ONNX GPU |
| **CPU Server** | GGUF (IQ2_XS) | ONNX (INT8) | ONNX (FP32) |
| **Edge Device** | GGUF (IQ2_XS) | ONNX (INT8) | ONNX (FP32) |
| **Mobile** | N/A | CoreML/ONNX | CoreML/ONNX |

### 2. Packaging Checklist

- [ ] Include `manifest.json` with metadata
- [ ] Add `requirements.txt` or `environment.yml`
- [ ] Provide `README.md` with deployment instructions
- [ ] Include inference script example
- [ ] Document hardware requirements
- [ ] Add model license information
- [ ] Include performance benchmarks

### 3. Security

```bash
# Encrypt model files (optional)
openssl enc -aes-256-cbc -salt \
  -in model.gguf \
  -out model.gguf.enc \
  -k YOUR_PASSWORD

# Decrypt on deployment
openssl enc -d -aes-256-cbc \
  -in model.gguf.enc \
  -out model.gguf \
  -k YOUR_PASSWORD
```

### 4. Monitoring

```python
# Add telemetry to deployed models
import time
import logging

class MonitoredBackend:
    def __init__(self, backend):
        self.backend = backend
        self.logger = logging.getLogger(__name__)

    def generate(self, prompt):
        start = time.time()
        result = self.backend.generate(prompt)
        latency = time.time() - start

        self.logger.info(f"Generated {len(result)} tokens in {latency:.2f}s")
        return result
```

---

## Troubleshooting

### Issue: Model too large for deployment

**Solution**: Use extreme compression
```bash
# Re-quantize to IQ2_XS or 2-bit MLX
uv run python scripts/quantize_llm.py \
  --model Qwen/Qwen3-8B \
  --formats gguf \
  --gguf-precision IQ2_XS
```

### Issue: TensorRT engine crashes on different GPU

**Solution**: Always build TensorRT on target GPU
```bash
# Check GPU arch
nvidia-smi --query-gpu=name --format=csv,noheader

# Rebuild engine on target
trtexec --onnx=model.onnx --saveEngine=model_target.engine
```

### Issue: CoreML model not using ANE

**Solution**: Verify precision and shape
```bash
# INT8 models are required for ANE
uv run python scripts/quantize_coreml_vision.py --model yolo-v8n --precision int8
```

---

For more examples, see:
- `run/examples/` - LLM runtime examples
- `run-coreml/examples/` - Vision/Speech CoreML examples
- `run-onnx/examples/` - ONNX runtime examples
- `docker/` - Docker deployment examples
