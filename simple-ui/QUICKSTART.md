# Quick Start Guide

## Installation

```bash
cd simple-ui
npm install
```

## Running the UI

```bash
npm run dev
```

Open: **http://localhost:3001**

## Quick Tour

### 1. Models Tab (Browse Your Models)
- Default scans: `./quant/models`
- Shows all GGUF, MLX, CoreML, ONNX models
- Click **Copy** to get model path
- Quick actions: Evaluate, Start Server, Inference

### 2. Quantize Tab (Create Quantized Models)
**Example: Quantize Qwen3-1.7B to GGUF + MLX**
1. Select **LLM** type
2. Choose **Qwen3 1.7B**
3. Check both **GGUF** and **MLX**
4. Set GGUF precision: **Q4_K_M**
5. Set MLX bits: **4**
6. Output: `./models`
7. Click **Start Quantization**
8. Watch full CLI output in real-time!

**Example: IQ Quantization (Extreme Compression)**
1. Same as above, but check **Enable IQ Quantization**
2. Choose **IQ3_XS** (3-bit)
3. Creates tiny models with great accuracy

**Vision Models:**
- YOLOv8: Nano (3MB) → XLarge (68MB)
- ViT: Image classification
- CLIP: Text+Image embeddings

**Speech Models:**
- Whisper: Tiny (39MB) → Large (1.5GB)

### 3. Evaluate Tab (Test Model Quality)
**Run Evaluation:**
1. Enter model path (or copy from Models tab)
2. Select format: GGUF, MLX, CoreML, or ONNX
3. Choose type: LLM, Vision, or Speech
4. Set questions (10-140 for LLM)
5. Click **Run Evaluation**
6. Watch full output!

**Comparison View:**
- Automatic table of all evaluated models
- Color-coded accuracy (green >95%, yellow >90%)
- Best/Fastest/Average stats
- Category breakdown

### 4. Server Tab (Run Model Servers)
**Start GGUF Server:**
1. Enter model path
2. Select **GGUF (llama-server)**
3. Port: **8080** (or custom)
4. GPU Layers: **99** (use GPU)
5. Context: **2048**
6. Click **Start Server**
7. Click **Open →** to access server

**Multiple Servers:**
- Run GGUF on 8080
- Run MLX on 8081
- Both running simultaneously!
- Click **Stop All Servers** to stop all

### 5. Inference Tab (Chat with Models)
1. Enter server URL: `http://localhost:8080`
2. Select mode: Chat, Image, or Voice
3. Type message, press Send
4. View inference time per message
5. Check analytics at bottom

## Common Workflows

### Workflow 1: Quantize → Evaluate → Compare
```
1. Quantize Tab: Create Q4_K_M version
2. Quantize Tab: Create IQ3_XS version
3. Evaluate Tab: Test Q4_K_M
4. Evaluate Tab: Test IQ3_XS
5. View comparison table automatically!
```

### Workflow 2: Browse → Server → Chat
```
1. Models Tab: Scan directory
2. Models Tab: Find model, copy path
3. Server Tab: Start server with that model
4. Inference Tab: Chat with the model
```

### Workflow 3: Full Pipeline
```
1. Quantize: Qwen3-1.7B → GGUF Q4_K_M + MLX 4-bit + IQ3_XS
2. Evaluate: All 3 versions (140 questions each)
3. Compare: See which is best (accuracy vs speed vs size)
4. Server: Start the winner
5. Inference: Use it!
```

## Tips

### See Full CLI Output
- Quantize and Evaluate tabs show **full terminal output**
- Scrollable windows with mono-spaced font
- Easy to see downloads, progress bars, conversions

### Track Progress
- Green text = success
- Red text = error
- Progress bars show actual CLI output
- If stuck, check output window (e.g., "Fetching 12 files...")

### Model Paths
- Use **Models** tab to browse and copy paths
- Format: `./quant/models/model-name/format/precision/file.gguf`
- Examples:
  - `./quant/models/qwen3-1.7b/gguf/Q4_K_M/hf_model_q4_k_m.gguf`
  - `./quant/models/qwen3-1.7b/mlx/4bit-mixed/`

### Evaluation Results
- Stored automatically (last 50)
- Persistent across sessions
- File: `../quant/evaluation_results.json`

### Server Management
- Each server gets unique ID: `gguf-8080`, `mlx-8081`
- Green pulsing dot = running
- Shows PID (process ID) for debugging
- Can run multiple models simultaneously

## Troubleshooting

### "No models found"
- Check directory path in Models tab
- Default: `./quant/models`
- Try absolute path: `/Users/you/llama-pajamas/quant/models`

### "Server failed to start"
- Check port not already in use
- Try different port number
- Check model path is correct
- Look at terminal for error messages

### "Evaluation failed"
- Verify model path exists
- Check format matches (GGUF file = GGUF format)
- View full output window for errors

### "Quantization stuck"
- Check output window - may be downloading files
- Large models take time (e.g., "Fetching 12 files: 50%...")
- Output window shows actual progress

## What's Actually Available

### Working Server Types:
- ✅ GGUF (llama-server)
- ✅ MLX

### Planned (not yet implemented):
- ⏳ CoreML
- ⏳ ONNX Runtime
- ⏳ TensorRT

### Evaluation Formats:
- ✅ GGUF
- ✅ MLX
- ⏳ CoreML (UI ready, backend needs work)
- ⏳ ONNX (UI ready, backend needs work)

## Next Steps

1. Start the UI: `npm run dev`
2. Go to **Models** tab → Scan for existing models
3. Or go to **Quantize** tab → Create new models
4. Then **Evaluate** → Compare quality
5. Then **Server** → Run the best one
6. Finally **Inference** → Use it!

Enjoy! 🚀
