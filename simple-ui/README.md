# LlamaPajamas - Simple UI

A simple web interface for quantizing, evaluating, and running LLM models.

## Features

### 1. Quantize
- Select from pre-configured models (Qwen3 LLMs, YOLOv8 vision, Whisper speech)
- Choose formats: GGUF, MLX, or both
- Configure precision levels (Q4_K_M, Q5_K_M, Q6_K, Q8_0 for GGUF)
- Enable IQ quantization for extreme compression (IQ2_XS, IQ3_XS, etc.)
- Real-time progress tracking
- View results with model size and location

### 2. Evaluate
- Run evaluations on quantized models
- Support for both GGUF and MLX formats
- Configurable number of test questions (10-140)
- Live progress updates
- Detailed results with:
  - Overall accuracy percentage
  - Average inference time per question
  - Category-specific breakdown (knowledge, math, reasoning, etc.)

### 3. Server Management
- Start/stop model servers
- Support for GGUF (llama-server) and MLX servers
- Configure custom ports
- View running servers with:
  - Port numbers
  - Model paths
  - Process IDs (PIDs)
  - Clickable server URLs

### 4. Inference Interface
- Three modes: Chat, Image, Voice
- Real-time chat with models
- Upload images for vision models
- Voice input support (coming soon)
- Analytics:
  - Message count
  - Per-message inference time
  - Average speed across conversation
- Clean, simple UI

## Installation

```bash
cd simple-ui
npm install
```

## Running

```bash
npm run dev
```

The UI will be available at: http://localhost:3001

## Requirements

- Node.js 18+
- Running `llama-pajamas-quant` CLI (from parent `quant/` directory)
- Optional: Model servers running for inference

## Architecture

- **Frontend**: Next.js 14 + React 18 + TypeScript
- **Styling**: Tailwind CSS
- **API**: Next.js API routes calling CLI commands via child processes
- **Real-time Updates**: Server-Sent Events (SSE) for progress streaming

## API Endpoints

- `POST /api/quantize` - Run quantization
- `POST /api/evaluate` - Run evaluation
- `GET /api/server/status` - Check server status
- `POST /api/server/start` - Start model server
- `POST /api/server/stop` - Stop model server

## Usage

### 1. Quantize a Model

1. Go to the "Quantize" tab
2. Select model type (LLM, Vision, or Speech)
3. Choose a model from the dropdown
4. Select formats (GGUF, MLX, or both)
5. Configure precision settings
6. Optionally enable IQ quantization for extreme compression
7. Click "Start Quantization"
8. Watch real-time progress
9. View results with model location

### 2. Evaluate a Model

1. Go to the "Evaluate" tab
2. Enter the full path to your quantized model
3. Select format (GGUF or MLX)
4. Choose number of questions (slider: 10-140)
5. Click "Run Evaluation"
6. View results with accuracy and category breakdown

### 3. Start a Server

1. Go to the "Server" tab
2. Enter model path
3. Select server type (GGUF or MLX)
4. Choose port number
5. Click "Start Server"
6. Server will appear in "Running Servers" list
7. Click the URL to access the server

### 4. Chat with a Model

1. Go to the "Inference" tab
2. Enter server URL (e.g., http://localhost:8080)
3. Select mode (Chat, Image, or Voice)
4. Type your message and press Send
5. View response with inference time
6. Check analytics at the bottom

## File Structure

```
simple-ui/
├── app/
│   ├── api/
│   │   ├── quantize/route.ts
│   │   ├── evaluate/route.ts
│   │   └── server/
│   │       ├── status/route.ts
│   │       ├── start/route.ts
│   │       └── stop/route.ts
│   ├── globals.css
│   ├── layout.tsx
│   └── page.tsx
├── components/
│   ├── QuantizePanel.tsx
│   ├── EvaluatePanel.tsx
│   ├── ServerPanel.tsx
│   └── InferencePanel.tsx
├── package.json
└── README.md
```

## Notes

- The UI automatically streams progress from CLI commands
- Servers run as background processes
- Model paths are relative to the quant/ directory
- Dark mode is supported automatically
- All components are client-side rendered for interactivity
