'use client'

import { useState, useEffect } from 'react'
import { useModelConfigs, useDirectorySettings } from './SettingsPanel'

const MODELS = {
  llm: [
    { id: 'Qwen/Qwen3-1.7B', name: 'Qwen3 1.7B' },
    { id: 'Qwen/Qwen3-4B', name: 'Qwen3 4B' },
    { id: 'Qwen/Qwen3-8B', name: 'Qwen3 8B' },
    { id: 'Qwen/Qwen3-14B', name: 'Qwen3 14B' },
    { id: 'Qwen/Qwen3-32B', name: 'Qwen3 32B' },
  ],
  vision: [
    // YOLO Models
    { id: 'yolov8n', name: 'YOLOv8 Nano (3MB, fastest)' },
    { id: 'yolov8s', name: 'YOLOv8 Small (11MB)' },
    { id: 'yolov8m', name: 'YOLOv8 Medium (25MB)' },
    { id: 'yolov8l', name: 'YOLOv8 Large (43MB)' },
    { id: 'yolov8x', name: 'YOLOv8 XLarge (68MB, best accuracy)' },
    // Vision Transformers
    { id: 'google/vit-base-patch16-224', name: 'ViT Base (Image Classification)' },
    { id: 'google/vit-large-patch16-224', name: 'ViT Large' },
    // CLIP Models
    { id: 'openai/clip-vit-base-patch32', name: 'CLIP ViT-B/32 (Text+Image)' },
    { id: 'openai/clip-vit-large-patch14', name: 'CLIP ViT-L/14' },
  ],
  speech: [
    { id: 'whisper-tiny', name: 'Whisper Tiny (39MB)' },
    { id: 'whisper-base', name: 'Whisper Base (74MB)' },
    { id: 'whisper-small', name: 'Whisper Small (244MB)' },
    { id: 'whisper-medium', name: 'Whisper Medium (769MB)' },
    { id: 'whisper-large', name: 'Whisper Large (1.5GB)' },
  ],
}

type ModelType = keyof typeof MODELS

export default function QuantizePanel() {
  const modelConfigs = useModelConfigs()
  const directories = useDirectorySettings()
  const [modelType, setModelType] = useState<ModelType>('llm')
  const [selectedModel, setSelectedModel] = useState('')
  const [useLocalModel, setUseLocalModel] = useState(false)
  const [formats, setFormats] = useState<string[]>(['gguf'])
  const [ggufPrecision, setGgufPrecision] = useState('Q4_K_M')
  const [mlxBits, setMlxBits] = useState('4')
  const [enableIQ, setEnableIQ] = useState(false)
  const [iqPrecision, setIqPrecision] = useState('IQ2_XS')
  const [iqF16Model, setIqF16Model] = useState('')
  const [iqCalibrationFile, setIqCalibrationFile] = useState('')

  // Auto-discovered files
  const [f16Models, setF16Models] = useState<string[]>([])
  const [calibrationFiles, setCalibrationFiles] = useState<string[]>([])
  const [discoveringFiles, setDiscoveringFiles] = useState(false)
  const [outputDir, setOutputDir] = useState('./models')
  const [isRunning, setIsRunning] = useState(false)
  const [progress, setProgress] = useState('')
  const [result, setResult] = useState<any>(null)

  // Discover F16 models and calibration files on mount and when IQ is enabled
  useEffect(() => {
    if (enableIQ && !discoveringFiles) {
      discoverIQFiles()
    }
  }, [enableIQ])

  const discoverIQFiles = async () => {
    setDiscoveringFiles(true)
    try {
      const res = await fetch('/api/iq/discover', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ modelsDir: directories.modelsDirectory || '../quant/models' }),
      })
      const data = await res.json()
      setF16Models(data.f16Models || [])
      setCalibrationFiles(data.calibrationFiles || [])

      // Auto-select first options
      if (data.f16Models?.length > 0 && !iqF16Model) {
        setIqF16Model(data.f16Models[0])
      }
      if (data.calibrationFiles?.length > 0 && !iqCalibrationFile) {
        setIqCalibrationFile(data.calibrationFiles[0])
      }
    } catch (error) {
      console.error('Error discovering IQ files:', error)
    } finally {
      setDiscoveringFiles(false)
    }
  }

  // Simplified IQ quantization - single function, calls CLI once
  const handleIQQuantize = async () => {
    if (!iqF16Model || !iqCalibrationFile) {
      alert('Please select both F16 model and calibration file')
      return
    }

    setIsRunning(true)
    setProgress('')
    setResult(null)

    try {
      const res = await fetch('/api/iq', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sourceModel: iqF16Model,
          calibrationFile: iqCalibrationFile,
          precision: iqPrecision,
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

  const handleQuantize = async () => {
    setIsRunning(true)
    setProgress('Starting quantization...')
    setResult(null)

    try {
      const res = await fetch('/api/quantize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          modelType,
          model: selectedModel,
          formats,
          ggufPrecision,
          mlxBits: parseInt(mlxBits),
          outputDir,
          enableIQ,
          iqPrecision,
        }),
      })

      const reader = res.body?.getReader()
      const decoder = new TextDecoder()

      if (reader) {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          const text = decoder.decode(value)
          const lines = text.split('\n').filter(l => l.trim())

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = JSON.parse(line.slice(6))
              if (data.progress) {
                setProgress(data.progress)
              }
              if (data.result) {
                setResult(data.result)
              }
            }
          }
        }
      }
    } catch (error) {
      setProgress(`Error: ${error}`)
    } finally {
      setIsRunning(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold mb-4">Quantize Model</h2>

        {/* Model Type */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">Model Type</label>
          <div className="flex gap-2">
            {(['llm', 'vision', 'speech'] as ModelType[]).map((type) => (
              <button
                key={type}
                onClick={() => {
                  setModelType(type)
                  setSelectedModel('')
                }}
                className={`px-4 py-2 rounded capitalize ${
                  modelType === type
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                }`}
              >
                {type}
              </button>
            ))}
          </div>
        </div>

        {/* Model Selection */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">Model</label>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="w-full px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600"
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

            {/* HuggingFace Models */}
            <optgroup label="🤗 Download from HuggingFace">
              {MODELS[modelType].map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name}
                </option>
              ))}
            </optgroup>
          </select>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            Local models will be quantized, HuggingFace models will be downloaded first
          </p>
        </div>

        {/* LLM-specific options */}
        {modelType === 'llm' && (
          <>
            {/* Formats */}
            <div className="mb-4">
              <label className="block text-sm font-medium mb-2">Formats</label>
              <div className="flex gap-4">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={formats.includes('gguf')}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setFormats([...formats, 'gguf'])
                      } else {
                        setFormats(formats.filter(f => f !== 'gguf'))
                      }
                    }}
                    className="mr-2"
                  />
                  GGUF
                </label>
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={formats.includes('mlx')}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setFormats([...formats, 'mlx'])
                      } else {
                        setFormats(formats.filter(f => f !== 'mlx'))
                      }
                    }}
                    className="mr-2"
                  />
                  MLX
                </label>
              </div>
            </div>

            {/* GGUF Precision */}
            {formats.includes('gguf') && (
              <div className="mb-4">
                <label className="block text-sm font-medium mb-2">GGUF Precision</label>
                <select
                  value={ggufPrecision}
                  onChange={(e) => setGgufPrecision(e.target.value)}
                  className="w-full px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600"
                >
                  <option value="Q4_K_M">Q4_K_M (4-bit, balanced)</option>
                  <option value="Q5_K_M">Q5_K_M (5-bit, better quality)</option>
                  <option value="Q6_K">Q6_K (6-bit, high quality)</option>
                  <option value="Q8_0">Q8_0 (8-bit, very high quality)</option>
                </select>
              </div>
            )}

            {/* MLX Bits */}
            {formats.includes('mlx') && (
              <div className="mb-4">
                <label className="block text-sm font-medium mb-2">MLX Bits</label>
                <select
                  value={mlxBits}
                  onChange={(e) => setMlxBits(e.target.value)}
                  className="w-full px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600"
                >
                  <option value="4">4-bit</option>
                  <option value="8">8-bit</option>
                </select>
              </div>
            )}

            {/* IQ Quantization */}
            <div className="mb-4">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={enableIQ}
                  onChange={(e) => setEnableIQ(e.target.checked)}
                  className="mr-2"
                />
                <span className="font-medium">Enable IQ Quantization (Extreme Compression)</span>
              </label>
              {enableIQ && (
                <div className="mt-3 ml-6 p-4 bg-yellow-50 dark:bg-yellow-900/10 border border-yellow-200 dark:border-yellow-800 rounded">
                  <h4 className="font-medium text-sm mb-3">🎯 IQ Quantization (Ultra-Low Bit)</h4>

                  {discoveringFiles && (
                    <p className="text-xs text-gray-600 mb-2">🔍 Discovering F16 models and calibration files...</p>
                  )}

                  {/* F16 Model Dropdown (Auto-discovered) */}
                  <div className="mb-3">
                    <label className="block text-xs font-medium mb-1">
                      F16 Model (auto-discovered from quantization output)
                      <button
                        onClick={discoverIQFiles}
                        className="ml-2 text-xs text-blue-500 hover:underline"
                      >
                        Refresh
                      </button>
                    </label>
                    <select
                      value={iqF16Model}
                      onChange={(e) => setIqF16Model(e.target.value)}
                      className="w-full px-2 py-1 text-xs border rounded dark:bg-gray-700 dark:border-gray-600"
                    >
                      <option value="">Select F16 GGUF model...</option>
                      {f16Models.map((model) => (
                        <option key={model} value={model}>
                          {model.split('/').slice(-3).join('/')}
                        </option>
                      ))}
                    </select>
                    {f16Models.length === 0 && !discoveringFiles && (
                      <p className="text-xs text-gray-500 mt-1">
                        No F16 models found. F16 GGUF is created automatically during quantization.
                      </p>
                    )}
                  </div>

                  {/* Calibration File Dropdown (Auto-discovered) */}
                  <div className="mb-3">
                    <label className="block text-xs font-medium mb-1">
                      Calibration Data
                    </label>
                    <select
                      value={iqCalibrationFile}
                      onChange={(e) => setIqCalibrationFile(e.target.value)}
                      className="w-full px-2 py-1 text-xs border rounded dark:bg-gray-700 dark:border-gray-600"
                    >
                      <option value="">Select calibration file...</option>
                      {calibrationFiles.map((file) => (
                        <option key={file} value={file}>
                          {file.split('/').slice(-2).join('/')}
                        </option>
                      ))}
                    </select>
                    {calibrationFiles.length === 0 && !discoveringFiles && (
                      <p className="text-xs text-gray-500 mt-1">
                        No calibration files found. Add .txt files to quant/calibration/ or quant/datasets/
                      </p>
                    )}
                  </div>

                  {/* Precision Selection */}
                  <div className="mb-3">
                    <label className="block text-xs font-medium mb-1">Precision</label>
                    <select
                      value={iqPrecision}
                      onChange={(e) => setIqPrecision(e.target.value)}
                      className="w-full px-2 py-1 text-xs border rounded dark:bg-gray-700 dark:border-gray-600"
                    >
                      <option value="IQ2_XS">IQ2_XS (2-bit, 50% size reduction)</option>
                      <option value="IQ3_XS">IQ3_XS (3-bit)</option>
                      <option value="IQ3_S">IQ3_S (3-bit, higher quality)</option>
                      <option value="IQ4_XS">IQ4_XS (4-bit)</option>
                    </select>
                  </div>

                  {/* Single Quantize Button */}
                  <button
                    onClick={handleIQQuantize}
                    disabled={isRunning || !iqF16Model || !iqCalibrationFile}
                    className="w-full px-3 py-2 text-sm bg-purple-500 text-white rounded font-medium hover:bg-purple-600 disabled:bg-gray-400 disabled:cursor-not-allowed"
                  >
                    {isRunning ? 'Quantizing...' : 'Quantize with IQ'}
                  </button>

                  <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                    💡 IQ quantization automatically generates imatrix and quantizes in one step. Achieves better quality at ultra-low bit rates (2-4 bit).
                  </p>
                </div>
              )}
            </div>
          </>
        )}

        {/* Vision/Speech specific options */}
        {(modelType === 'vision' || modelType === 'speech') && (
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">Precision</label>
            <select className="w-full px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600">
              <option value="int8">INT8</option>
              <option value="int4">INT4</option>
              <option value="fp16">FP16</option>
            </select>
          </div>
        )}

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

        {/* Quantize Button */}
        <button
          onClick={handleQuantize}
          disabled={!selectedModel || isRunning}
          className="w-full px-4 py-3 bg-blue-500 text-white rounded font-medium hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed"
        >
          {isRunning ? 'Quantizing...' : 'Start Quantization'}
        </button>
      </div>

      {/* Progress - Full Output */}
      {progress && (
        <div className="mt-6 p-4 bg-gray-100 dark:bg-gray-700 rounded max-h-96 overflow-y-auto">
          <h3 className="font-medium mb-2">Output:</h3>
          <pre className="text-xs font-mono whitespace-pre-wrap">{progress}</pre>
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="mt-6 p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded">
          <h3 className="font-medium mb-2 text-green-800 dark:text-green-300">Quantization Complete!</h3>
          <div className="text-sm space-y-1">
            <p><strong>Model Path:</strong> {result.path}</p>
            <p><strong>Size:</strong> {result.size}</p>
            <p><strong>Format:</strong> {result.format}</p>
            {result.accuracy && <p><strong>Accuracy:</strong> {result.accuracy}%</p>}
          </div>
        </div>
      )}
    </div>
  )
}
