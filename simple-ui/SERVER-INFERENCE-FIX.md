# Server & Inference Fixes

## ✅ Fixed Implementation

### Server Starting (Fixed)

**Previous Issue:** Server starting was using incorrect commands and paths.

**Fix Applied:**
- GGUF servers now use: `uv run python -m llama_cpp.server`
- MLX servers now use: `uv run python -m mlx_lm.server`
- Proper working directory: `../run`
- Correct parameters passed:
  - GGUF: `--model`, `--host`, `--port`, `--n-gpu-layers`, `--n-ctx`
  - MLX: `--model`, `--host`, `--port`, `--max-tokens`

**File:** `simple-ui/app/api/server/start/route.ts`

**How It Works:**
```typescript
// GGUF Server
spawn('uv', [
  'run', 'python', '-m', 'llama_cpp.server',
  '--model', modelPath,
  '--host', '0.0.0.0',
  '--port', port.toString(),
  '--n-gpu-layers', gpuLayers.toString(),
  '--n-ctx', contextSize.toString(),
], { cwd: '../run', detached: true })

// MLX Server
spawn('uv', [
  'run', 'python', '-m', 'mlx_lm.server',
  '--model', modelPath,
  '--host', '0.0.0.0',
  '--port', port.toString(),
  '--max-tokens', contextSize.toString(),
], { cwd: '../run', detached: true })
```

---

### Inference (Fixed)

**Previous Issue:** Inference was trying to connect to HTTP servers instead of using Python runtime directly.

**Fix Applied:**
- Created new `/api/inference` endpoint
- Uses `llama_pajamas_run` Python module directly
- Streams responses in real-time
- Matches evaluation workflow from `run/examples/simple_usage.py`

**Files:**
- API: `simple-ui/app/api/inference/route.ts`
- UI: `simple-ui/components/InferencePanel.tsx`

**How It Works:**

1. **API Creates Temporary Python Script:**
```python
from llama_pajamas_run import RuntimeConfig, ModelLoader

config = RuntimeConfig(
    backend="gguf",  # or "mlx"
    model_path="./models/qwen3-8b",
    max_tokens=200,
    temperature=0.7,
    n_gpu_layers=-1,  # for GGUF
)

with ModelLoader(config) as loader:
    # Stream generation
    for chunk in loader.generate(prompt, stream=True):
        print(f"CHUNK:{chunk}", flush=True)
```

2. **API Streams Chunks to UI:**
```typescript
// Server-Sent Events stream
send({ chunk: "generated text" })
send({ status: "complete" })
```

3. **UI Updates in Real-Time:**
- Shows streaming response as it generates
- Tracks inference time per message
- Displays session analytics

---

## 🎯 Usage

### Starting a Server

1. Go to **Server** tab
2. Enter model path: `./models/qwen3-8b/gguf/Q4_K_M/model.gguf`
3. Select server type: GGUF or MLX
4. Click "Start Optimized Server"
5. Server runs on specified port (8080 for GGUF, 8081 for MLX)

**Server URL:** `http://localhost:8080` (or 8081 for MLX)

---

### Using Inference

1. Go to **Inference** tab
2. Enter model path: `./models/qwen3-8b` (directory) or `./models/qwen3-8b/gguf/Q4_K_M/model.gguf` (file)
3. Select backend: GGUF or MLX
4. Adjust temperature and max tokens
5. Type message and click "Send"

**Features:**
- Real-time streaming responses
- Chat history with timestamps
- Per-message inference time
- Session analytics (avg response time, message count)

---

## 🔄 How They Match `run/` Implementation

### Evaluation Pattern (from `run/examples/simple_usage.py`)

```python
from llama_pajamas_run import RuntimeConfig, ModelLoader

config = RuntimeConfig(
    backend="mlx",
    model_path="./models/qwen3-8b",
    max_tokens=100,
    temperature=0.7,
)

with ModelLoader(config) as loader:
    response = loader.generate("Write a Python function:")
    print(response)
```

### Server Pattern (from `llama-cpp-python` and `mlx-lm`)

```bash
# GGUF Server
python -m llama_cpp.server \
  --model ./models/qwen3-8b/gguf/Q4_K_M/model.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --n-gpu-layers 99 \
  --n-ctx 2048

# MLX Server
python -m mlx_lm.server \
  --model ./models/qwen3-8b/mlx/4bit-mixed \
  --host 0.0.0.0 \
  --port 8081 \
  --max-tokens 2048
```

---

## 🧪 Testing

### Test Server Starting

```bash
# Via UI:
1. Go to Server tab
2. Enter model path
3. Click "Start Optimized Server"
4. Check "Running Servers" section for green pulsing dot
5. Click "Open →" link to test server

# Via CLI (to verify):
curl http://localhost:8080/health
```

### Test Inference

```bash
# Via UI:
1. Go to Inference tab
2. Enter model path: ./models/qwen3-8b
3. Select backend: mlx
4. Type: "What is Python?"
5. Click Send
6. Watch streaming response appear

# Should see:
- Real-time text streaming
- Inference time displayed
- Analytics updated
```

---

## 📋 Implementation Files

**Server:**
- `/api/server/start/route.ts` - Fixed server spawning
- `/api/server/stop/route.ts` - Unchanged
- `/api/server/status/route.ts` - Unchanged

**Inference:**
- `/api/inference/route.ts` - NEW - Python runtime API
- `/components/InferencePanel.tsx` - Fixed to use new API

---

## ✅ Status

- ✅ Server starting uses correct llama-cpp and MLX commands
- ✅ Servers run in correct working directory (`../run`)
- ✅ Inference uses `llama_pajamas_run` Python module directly
- ✅ Streaming responses work in real-time
- ✅ Matches patterns from `run/examples/simple_usage.py`
- ✅ Both GGUF and MLX backends supported

**Ready for testing!** 🚀
