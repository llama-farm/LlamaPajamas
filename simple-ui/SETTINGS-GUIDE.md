# Settings Guide - Model Configuration Management

## ✅ New Feature: Settings Tab

The Settings tab provides a centralized location to configure and manage model paths, making them easily accessible across all tabs via dropdowns instead of manual path entry.

### 🎯 Key Features

1. **Model Configuration Management**
   - Add, edit, and delete model configurations
   - Save model paths with metadata (name, backend, type, description)
   - Persistent storage using browser localStorage

2. **Directory Scanning (NEW!)**
   - Automatically discover models in any directory
   - Recursively scans all subdirectories
   - Auto-detects backend (GGUF, MLX, CoreML, ONNX, TensorRT)
   - Auto-detects type (LLM, Vision, Speech)
   - Extracts precision/quantization from paths (Q4_K_M, IQ3_XS, FP16, etc.)
   - Preview discovered models before importing
   - Batch import selected models with one click

3. **Smart Dropdowns**
   - Select models from saved configurations in Inference and Server tabs
   - Auto-fills model path and backend based on selection
   - "Custom Path..." option for ad-hoc models

4. **Import/Export Settings**
   - Export all settings to JSON file
   - Import settings from JSON file
   - Share configurations across devices

5. **Model Statistics**
   - View count of configured models by type (LLM, Vision, Speech)
   - Track total configurations

---

## 📋 How to Use

### 1. Configure Models (Settings Tab)

**Add a New Model:**
1. Go to **⚙️ Settings** tab (first tab)
2. Click **+ Add Model**
3. Fill in the form:
   - **Name**: Friendly name (e.g., "Qwen3-8B Q4_K_M")
   - **Path**: Model file or directory path
   - **Backend**: GGUF, MLX, CoreML, ONNX, or TensorRT
   - **Type**: LLM, Vision, or Speech
   - **Description** (optional): Additional notes
4. Click **Add**

**Example Configuration:**
```
Name: Qwen3-8B Q4_K_M
Path: ./models/qwen3-8b/gguf/Q4_K_M/model.gguf
Backend: GGUF
Type: LLM
Description: Fast inference model for general chat
```

**Edit a Model:**
1. Click **Edit** on any model card
2. Modify the fields
3. Click **Update**

**Delete a Model:**
1. Click **Delete** on any model card
2. Confirm deletion

### 2. Discover Models Automatically (Directory Scanning)

**Quick Start:**
1. Go to **⚙️ Settings** tab
2. Find the **🔍 Discover Models in Directory** section (purple box)
3. Enter a directory path (e.g., `./models` or `./quant/models`)
4. Click **Scan Directory**
5. Review discovered models in the preview
6. Select/deselect models you want to import
7. Click **Import Selected (N)** to add them to your configurations

**What Gets Detected:**

The scanner automatically identifies:
- **File Formats**: `.gguf`, `.safetensors`, `.onnx`, `.mlpackage`, `.tflite`
- **Backends**: GGUF, MLX, CoreML, ONNX, TensorRT
- **Types**: LLM, Vision (YOLO, VIT, CLIP), Speech (Whisper, STT)
- **Precision**: Q4_K_M, IQ3_XS, FP16, INT8, etc. from directory structure

**Example Scan Results:**

After scanning `./quant/models`, you might see:
```
Found 27 model(s)

✓ qwen3-1.7b (Q4_K_M) - GGUF - LLM - 1.19 GB
✓ qwen3-8b (4bit-mixed) - MLX - LLM - 4.2 GB
✓ yolo-v8n (fp16) - ONNX - Vision - 24 MB
✓ whisper-tiny (int8) - ONNX - Speech - 38 MB
```

**Select and Import:**
- All models are auto-selected by default
- Uncheck models you don't want to import
- Click "Select All" / "Deselect All" to toggle all at once
- Click "Import Selected (N)" to batch import

**Benefits:**
- ✅ Discover all models at once (tested with 27 models!)
- ✅ No manual configuration needed
- ✅ Auto-detects backend, type, and precision
- ✅ Saves hours of manual entry
- ✅ Perfect for large model collections

**Tips:**
- Use relative paths like `./models` or `./quant/models`
- The scanner recursively searches all subdirectories
- MLX models are detected by `.safetensors` + `config.json` presence
- Precision is extracted from directory names (e.g., `gguf/Q4_K_M/model.gguf`)

### 3. Use Saved Models (Inference Tab)

**Select from Saved Models:**
1. Go to **💬 Inference** tab
2. Mode: **Chat**
3. Click the **Select Model** dropdown
4. Choose from your saved LLM models
5. Model path and backend are auto-filled!
6. Start chatting

**Use Custom Path:**
1. Select **"Custom Path..."** from dropdown
2. Manually enter model path and select backend
3. Use as before

### 4. Use Saved Models (Server Tab)

**Start Server with Saved Model:**
1. Go to **🚀 Server** tab
2. Click the **Select Model** dropdown
3. Choose from your saved LLM models
4. Model path is auto-filled
5. Click **Start Optimized Server**

**Use Custom Path:**
1. Select **"Custom Path..."** from dropdown
2. Manually enter model path
3. Start server as before

---

## 💾 Data Persistence

**Storage Location:** Browser localStorage
**Key:** `llamafarm-settings`

**Data Structure:**
```json
{
  "models": [
    {
      "id": "1699999999999",
      "name": "Qwen3-8B Q4_K_M",
      "path": "./models/qwen3-8b/gguf/Q4_K_M/model.gguf",
      "backend": "gguf",
      "type": "llm",
      "description": "Fast inference model"
    }
  ],
  "defaultMultimodalServer": "http://localhost:8000"
}
```

**Notes:**
- Settings persist across browser sessions
- Clearing browser data will remove settings
- Use Export/Import to backup configurations

---

## 📤 Import/Export Settings

### Export Settings

1. Go to **⚙️ Settings** tab
2. Click **Export Settings**
3. Downloads `llamafarm-settings.json`
4. Save for backup or sharing

### Import Settings

1. Go to **⚙️ Settings** tab
2. Click **Import Settings**
3. Select JSON file
4. Settings are loaded and saved
5. Existing settings are replaced

**Use Cases:**
- Backup configurations before clearing browser
- Share model configs with team members
- Sync settings across multiple machines
- Restore settings after browser reset

---

## 🎨 Model Configuration Examples

### LLM Models

**GGUF Model:**
```
Name: Qwen3-8B Q4_K_M
Path: ./models/qwen3-8b/gguf/Q4_K_M/model.gguf
Backend: GGUF
Type: LLM
Description: 4-bit quantized, balanced quality/speed
```

**MLX Model:**
```
Name: Qwen3-8B MLX 4-bit
Path: ./models/qwen3-8b/mlx/4bit-mixed
Backend: MLX
Type: LLM
Description: Apple Silicon optimized
```

### Vision Models

**YOLO Detection:**
```
Name: YOLO-v8n Detection
Path: ./models/yolo-v8n/coreml/fp16/model.mlpackage
Backend: CoreML
Type: Vision
Description: Object detection, FP16, ANE optimized
```

### Speech Models

**Whisper Transcription:**
```
Name: Whisper Tiny
Path: ./models/whisper-tiny/coreml/int8/encoder.mlpackage
Backend: CoreML
Type: Speech
Description: Fast transcription with ANE
```

---

## ⚙️ Server Settings

### Default Multimodal Server URL

**Purpose:** Default URL for Image and Voice modes in Inference tab

**How to Change:**
1. Go to **⚙️ Settings** tab
2. Find **Server Settings** section
3. Edit **Default Multimodal Server URL**
4. Changes save automatically
5. Inference tab will use this URL by default

**Default:** `http://localhost:8000`

---

## 📊 Benefits

### Before Settings Tab
- Manual path entry every time
- Prone to typos and errors
- No way to remember model paths
- Repetitive configuration
- Hard to discover what models you have

### After Settings Tab + Directory Scanning
- ✅ One-time model configuration
- ✅ Auto-discover ALL models at once (27 models in ~2 seconds!)
- ✅ Select from dropdown
- ✅ Auto-fill paths and backends
- ✅ Persistent across sessions
- ✅ Import/Export capabilities
- ✅ Team sharing enabled
- ✅ Reduced errors
- ✅ Faster workflow
- ✅ Know exactly what models you have and where they are

---

## 🚀 Workflow Examples

### Method 1: Directory Scanning (Fastest!)

**Initial Setup (Once):**
1. Go to **⚙️ Settings** tab
2. Enter your models directory: `./quant/models`
3. Click **Scan Directory** (wait ~2 seconds)
4. Review discovered models (e.g., 27 models found!)
5. Click **Import Selected (27)** to add all models
6. Export settings for backup
7. **Done! All models configured automatically!** 🚀

**Daily Usage:**
1. **Inference** → Select from 27+ models in dropdown → Chat!
2. **Server** → Select any LLM model → Start Server!
3. **Evaluate** → Select vision/speech models for testing!

### Method 2: Manual Configuration (For Custom Setups)

**Initial Setup (Once):**
1. Go to **⚙️ Settings** tab
2. Click **+ Add Model** for each model:
   - Qwen3-8B Q4_K_M (GGUF)
   - Qwen3-8B MLX 4-bit (MLX)
   - YOLO-v8n (CoreML)
   - Whisper Tiny (CoreML)
3. Export settings for backup

**Daily Usage:**
1. **Inference** → Select "Qwen3-8B Q4_K_M" from dropdown → Chat!
2. **Server** → Select "Qwen3-8B MLX 4-bit" → Start Server!
3. No more manual path entry! 🎉

---

## 🔧 Technical Details

**Component:** `components/SettingsPanel.tsx`
**Hook:** `useModelConfigs()` - Access model configs from other components
**Storage:** `localStorage.setItem('llamafarm-settings', JSON.stringify(settings))`

**Integration Points:**
- `InferencePanel.tsx` - Chat mode model selection
- `ServerPanel.tsx` - Server start model selection
- `app/api/models/discover/route.ts` - Directory scanning API

**Directory Scanning:**
- Recursively scans subdirectories using Node.js `fs/promises`
- Detects file types: `.gguf`, `.safetensors`, `.onnx`, `.mlpackage`, `.tflite`
- MLX detection: Directories with `.safetensors` + `config.json`
- Backend detection: Based on file extension and path patterns
- Type detection: Keywords in path (yolo, vit, clip, whisper, stt)
- Precision extraction: Regex pattern `/Q\d+_K_[MS]|IQ\d+_X?S|F(P)?(16|32)|INT\d+/i`
- Tested with 27 models across 3 backends and 3 types

**Future Enhancements:**
- Cloud sync
- Model validation (size, integrity checks)
- Model performance tracking and benchmarks
- Automatic model updates from Hugging Face
- Model tags and favorites

---

## ✅ Summary

The Settings tab provides:
- 🔍 **Directory Scanning** - Auto-discover models in any directory (NEW!)
- 📝 Centralized model configuration
- 💾 Persistent storage (localStorage)
- 📤 Import/Export functionality
- 🔄 Easy model selection via dropdowns
- 📊 Configuration statistics
- ⚡ Faster workflow (27 models configured in ~2 seconds!)

**Total Tabs: 8**
1. ⚙️ Settings (NEW!)
2. 📁 Models
3. ⚡ Quantize
4. 📤 Export
5. 📊 Evaluate
6. 🔄 Batch
7. 🚀 Server
8. 💬 Inference

**All features ready!** 🚀
