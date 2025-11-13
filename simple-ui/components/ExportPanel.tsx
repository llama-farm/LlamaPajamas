'use client'

import { useState } from 'react'
import { useModelConfigs } from './SettingsPanel'

// LLM models can export to GGUF or MLX
const LLM_BACKENDS = [
  { id: 'gguf', name: 'GGUF', desc: 'Universal (CPU/GPU/Metal)' },
  { id: 'mlx', name: 'MLX', desc: 'Apple Silicon' },
]

// Vision/Speech models can export to ONNX, CoreML, or TensorRT
const MULTIMODAL_BACKENDS = [
  { id: 'onnx', name: 'ONNX', desc: 'Universal (CPU/GPU/Edge)' },
  { id: 'coreml', name: 'CoreML', desc: 'Apple Neural Engine' },
  { id: 'tensorrt', name: 'TensorRT', desc: 'NVIDIA GPU' },
]

const MODEL_TYPES = [
  { id: 'auto', name: 'Auto-detect' },
  { id: 'llm', name: 'LLM (Text)' },
  { id: 'vision', name: 'Vision' },
  { id: 'speech', name: 'Speech' },
]

export default function ExportPanel() {
  const modelConfigs = useModelConfigs()
  const [model, setModel] = useState('')
  const [useCustomModel, setUseCustomModel] = useState(false)
  const [backend, setBackend] = useState('gguf') // Default to GGUF for LLM
  const [precision, setPrecision] = useState('Q4_K_M') // Default GGUF precision
  const [modelType, setModelType] = useState('llm') // Default to LLM
  const [outputDir, setOutputDir] = useState('./models')
  const [isRunning, setIsRunning] = useState(false)
  const [progress, setProgress] = useState('')
  const [result, setResult] = useState<any>(null)

  // Get available backends based on model type
  const getAvailableBackends = () => {
    if (modelType === 'llm') {
      return LLM_BACKENDS
    } else if (modelType === 'vision' || modelType === 'speech') {
      return MULTIMODAL_BACKENDS
    }
    // For auto-detect, show all backends
    return [...LLM_BACKENDS, ...MULTIMODAL_BACKENDS]
  }

  const getPrecisionOptions = () => {
    switch (backend) {
      case 'gguf':
        return ['Q4_K_M', 'Q5_K_M', 'Q6_K', 'Q8_0', 'F16', 'F32']
      case 'onnx':
        return ['fp32', 'fp16', 'int8']
      case 'coreml':
        return ['fp32', 'fp16', 'int8', 'int4']
      case 'tensorrt':
        return ['fp32', 'fp16', 'int8']
      case 'mlx':
        return ['4bit', '8bit']
      default:
        return ['fp32', 'fp16', 'int8']
    }
  }

  const handleModelTypeChange = (newType: string) => {
    setModelType(newType)

    // Reset backend to first available for new model type
    const availableBackends = newType === 'llm' ? LLM_BACKENDS :
                              newType === 'vision' || newType === 'speech' ? MULTIMODAL_BACKENDS :
                              [...LLM_BACKENDS, ...MULTIMODAL_BACKENDS]

    // If current backend is not available for new type, switch to first available
    if (!availableBackends.find(b => b.id === backend)) {
      const newBackend = availableBackends[0].id
      setBackend(newBackend)

      // Also reset precision to first available option for new backend
      const precisionOptions = getPrecisionOptions()
      setPrecision(precisionOptions[0])
    }
  }

  const handleExport = async () => {
    setIsRunning(true)
    setProgress('Starting export...\n')
    setResult(null)

    try {
      const res = await fetch('/api/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          backend,
          precision,
          modelType,
          outputDir,
        }),
      })

      const reader = res.body?.getReader()
      const decoder = new TextDecoder()

      if (reader) {
        let fullOutput = ''
        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          const text = decoder.decode(value)
          fullOutput += text
          setProgress(fullOutput)

          const lines = text.split('\n').filter(l => l.trim())

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = JSON.parse(line.slice(6))
              if (data.result) {
                setResult(data.result)
              }
            }
          }
        }
      }
    } catch (error) {
      setProgress(prev => prev + `\nError: ${error}`)
    } finally {
      setIsRunning(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold mb-4">Export Model</h2>
        <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
          Unified model export to ONNX, CoreML, TensorRT, or MLX with automatic quantization
        </p>

        {/* Model Selection */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">
            Select Model
            {modelConfigs.filter(m => m.type === modelType).length === 0 && (
              <span className="ml-2 text-xs text-gray-500">(Configure models in Settings tab)</span>
            )}
          </label>
          <select
            value={useCustomModel ? 'custom' : model}
            onChange={(e) => {
              if (e.target.value === 'custom') {
                setUseCustomModel(true)
                setModel('')
              } else {
                setUseCustomModel(false)
                setModel(e.target.value)
              }
            }}
            className="w-full px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600 text-sm"
          >
            <option value="">Select a model...</option>

            {/* Saved Local Models */}
            {modelConfigs.filter(m => m.type === modelType).length > 0 && (
              <optgroup label="📁 Saved Local Models">
                {modelConfigs.filter(m => m.type === modelType).map((model) => (
                  <option key={model.id} value={model.path}>
                    {model.name} ({model.backend.toUpperCase()})
                  </option>
                ))}
              </optgroup>
            )}

            <option value="custom">Custom Model (HuggingFace/Path)...</option>
          </select>
        </div>

        {/* Custom Model Input */}
        {useCustomModel && (
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">Custom Model Name or Path</label>
            <input
              type="text"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="w-full px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600 text-sm"
              placeholder="e.g., yolov8n, Qwen/Qwen3-8B, whisper-tiny"
            />
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              HuggingFace model ID, local path, or model name
            </p>
          </div>
        )}

        {/* Model Type */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">Model Type</label>
          <div className="grid grid-cols-4 gap-2">
            {MODEL_TYPES.map((type) => (
              <button
                key={type.id}
                onClick={() => handleModelTypeChange(type.id)}
                className={`px-3 py-2 rounded text-sm ${
                  modelType === type.id
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-200 dark:bg-gray-700'
                }`}
              >
                {type.name}
              </button>
            ))}
          </div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
            {modelType === 'llm' && '📝 LLM models export to GGUF or MLX'}
            {(modelType === 'vision' || modelType === 'speech') && '🎯 Vision/Speech models export to ONNX, CoreML, or TensorRT'}
            {modelType === 'auto' && '🔍 Auto-detect will determine the best export format'}
          </p>
        </div>

        {/* Backend Selection */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">
            Export Backend
            {modelType !== 'auto' && (
              <span className="ml-2 text-xs text-gray-500">
                (Filtered by {modelType === 'llm' ? 'LLM' : 'Vision/Speech'} type)
              </span>
            )}
          </label>
          <div className="grid grid-cols-2 gap-2">
            {getAvailableBackends().map((b) => (
              <button
                key={b.id}
                onClick={() => {
                  setBackend(b.id)
                  // Reset precision to first available option
                  const options = getPrecisionOptions()
                  if (!options.includes(precision)) {
                    setPrecision(options[0])
                  }
                }}
                className={`px-3 py-2 rounded text-sm ${
                  backend === b.id
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-200 dark:bg-gray-700'
                }`}
                title={b.desc}
              >
                <div className="font-medium">{b.name}</div>
                <div className="text-xs opacity-75">{b.desc}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Precision Selection */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">
            Precision / Quantization
          </label>
          <select
            value={precision}
            onChange={(e) => setPrecision(e.target.value)}
            className="w-full px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600"
          >
            {getPrecisionOptions().map((p) => (
              <option key={p} value={p}>
                {p.toUpperCase()}
                {p === 'fp32' && ' (Full precision)'}
                {p === 'fp16' && ' (Half precision)'}
                {p === 'int8' && ' (8-bit quantized, 75% smaller)'}
                {p === 'int4' && ' (4-bit quantized, 87.5% smaller)'}
                {p === '4bit' && ' (4-bit quantized)'}
                {p === '8bit' && ' (8-bit quantized)'}
              </option>
            ))}
          </select>
        </div>

        {/* Output Directory */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">Output Directory</label>
          <input
            type="text"
            value={outputDir}
            onChange={(e) => setOutputDir(e.target.value)}
            className="w-full px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600"
            placeholder="./models"
          />
        </div>

        {/* Export Button */}
        <button
          onClick={handleExport}
          disabled={!model || isRunning}
          className="w-full px-4 py-3 bg-green-500 text-white rounded font-medium hover:bg-green-600 disabled:bg-gray-400 disabled:cursor-not-allowed"
        >
          {isRunning ? 'Exporting...' : 'Export Model'}
        </button>
      </div>

      {/* Progress Output */}
      {progress && (
        <div className="mt-6 p-4 bg-gray-100 dark:bg-gray-700 rounded max-h-96 overflow-y-auto">
          <h3 className="font-medium mb-2">Export Output:</h3>
          <pre className="text-xs font-mono whitespace-pre-wrap">{progress}</pre>
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="mt-6 p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded">
          <h3 className="font-medium mb-2 text-green-800 dark:text-green-300">
            Export Complete! ✅
          </h3>
          <div className="text-sm space-y-1">
            <p><strong>Output Path:</strong> {result.path}</p>
            <p><strong>Backend:</strong> {result.backend}</p>
            <p><strong>Precision:</strong> {result.precision}</p>
            {result.size && <p><strong>Size:</strong> {result.size}</p>}
          </div>
        </div>
      )}

      {/* Info Box */}
      <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded">
        <h3 className="font-medium mb-2">💡 Backend Guide</h3>
        <div className="text-sm space-y-2">
          <div><strong>ONNX:</strong> Universal - CPU, AMD GPU, Intel GPU, Edge devices</div>
          <div><strong>CoreML:</strong> Apple Silicon - ANE acceleration, mobile</div>
          <div><strong>TensorRT:</strong> NVIDIA GPU - Highest performance, data centers</div>
          <div><strong>MLX:</strong> Apple Silicon - LLM optimized, unified memory</div>
        </div>
      </div>
    </div>
  )
}
