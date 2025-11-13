'use client'

import { useState } from 'react'

const EXAMPLE_CONFIG = `# Batch Processing Config (YAML)
models:
  - name: qwen3-8b
    model: Qwen/Qwen3-8B
    formats: [gguf, mlx]
    gguf_precision: Q4_K_M
    mlx_bits: 4
    output: ./models/qwen3-8b

  - name: qwen3-4b
    model: Qwen/Qwen3-4B
    formats: [gguf]
    gguf_precision: Q4_K_M
    output: ./models/qwen3-4b

  - name: yolo-v8n
    model: yolov8n
    backend: onnx
    precision: int8
    output: ./models/yolo-v8n

parallel: 2  # Run 2 models in parallel
`

export default function BatchPanel() {
  const [configText, setConfigText] = useState(EXAMPLE_CONFIG)
  const [parallel, setParallel] = useState(2)
  const [dryRun, setDryRun] = useState(false)
  const [isRunning, setIsRunning] = useState(false)
  const [progress, setProgress] = useState('')
  const [result, setResult] = useState<any>(null)

  const handleBatchProcess = async () => {
    setIsRunning(true)
    setProgress('Starting batch processing...\n')
    setResult(null)

    try {
      const res = await fetch('/api/batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          config: configText,
          parallel,
          dryRun,
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

  const loadExample = () => {
    setConfigText(EXAMPLE_CONFIG)
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold mb-4">Batch Processing</h2>
        <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
          Process multiple models from a YAML/JSON configuration file with parallel execution support
        </p>

        {/* Config Editor */}
        <div className="mb-4">
          <div className="flex items-center justify-between mb-2">
            <label className="block text-sm font-medium">Batch Configuration (YAML)</label>
            <button
              onClick={loadExample}
              className="px-3 py-1 text-xs bg-gray-200 dark:bg-gray-700 rounded hover:bg-gray-300 dark:hover:bg-gray-600"
            >
              Load Example
            </button>
          </div>
          <textarea
            value={configText}
            onChange={(e) => setConfigText(e.target.value)}
            className="w-full px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600 font-mono text-sm"
            rows={16}
            placeholder="Enter YAML configuration..."
          />
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            Supports both YAML and JSON formats. Each model can specify formats, precisions, and output paths.
          </p>
        </div>

        {/* Options */}
        <div className="grid grid-cols-2 gap-4 mb-4">
          {/* Parallel Workers */}
          <div>
            <label className="block text-sm font-medium mb-2">Parallel Workers</label>
            <input
              type="number"
              value={parallel}
              onChange={(e) => setParallel(parseInt(e.target.value))}
              className="w-full px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600"
              min="1"
              max="8"
            />
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              Number of models to process in parallel (1-8)
            </p>
          </div>

          {/* Dry Run */}
          <div>
            <label className="block text-sm font-medium mb-2">Execution Mode</label>
            <label className="flex items-center p-3 border rounded dark:border-gray-700 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700">
              <input
                type="checkbox"
                checked={dryRun}
                onChange={(e) => setDryRun(e.target.checked)}
                className="mr-2"
              />
              <div>
                <div className="font-medium text-sm">Dry Run</div>
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  Preview commands without execution
                </div>
              </div>
            </label>
          </div>
        </div>

        {/* Process Button */}
        <button
          onClick={handleBatchProcess}
          disabled={!configText || isRunning}
          className="w-full px-4 py-3 bg-purple-500 text-white rounded font-medium hover:bg-purple-600 disabled:bg-gray-400 disabled:cursor-not-allowed"
        >
          {isRunning ? 'Processing Batch...' : dryRun ? 'Preview Batch (Dry Run)' : 'Start Batch Processing'}
        </button>
      </div>

      {/* Progress Output */}
      {progress && (
        <div className="mt-6 p-4 bg-gray-100 dark:bg-gray-700 rounded max-h-96 overflow-y-auto">
          <h3 className="font-medium mb-2">Batch Output:</h3>
          <pre className="text-xs font-mono whitespace-pre-wrap">{progress}</pre>
        </div>
      )}

      {/* Result Summary */}
      {result && (
        <div className="mt-6 p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded">
          <h3 className="font-medium mb-2 text-green-800 dark:text-green-300">
            Batch Processing Complete! ✅
          </h3>
          <div className="text-sm space-y-1">
            <p><strong>Total Models:</strong> {result.total || 'N/A'}</p>
            <p><strong>Successful:</strong> {result.successful || 'N/A'}</p>
            <p><strong>Failed:</strong> {result.failed || 'N/A'}</p>
            {result.duration && <p><strong>Duration:</strong> {result.duration}</p>}
          </div>
        </div>
      )}

      {/* Info Box */}
      <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded">
        <h3 className="font-medium mb-2">💡 Batch Processing Features</h3>
        <div className="text-sm space-y-2">
          <div><strong>Parallel Execution:</strong> Process multiple models simultaneously</div>
          <div><strong>Flexible Configuration:</strong> Mix LLMs, vision, and speech models in one batch</div>
          <div><strong>Dry Run Mode:</strong> Preview commands before execution</div>
          <div><strong>Error Handling:</strong> Continue processing even if individual models fail</div>
          <div><strong>Format Support:</strong> YAML (preferred) or JSON configuration files</div>
        </div>
      </div>

      {/* Example Configurations */}
      <div className="p-4 bg-gray-50 dark:bg-gray-700/50 rounded">
        <h3 className="font-medium mb-3">📝 Configuration Examples</h3>
        <div className="text-sm space-y-3">
          <div>
            <strong className="text-blue-600 dark:text-blue-400">LLM Quantization:</strong>
            <pre className="mt-1 text-xs font-mono bg-white dark:bg-gray-800 p-2 rounded overflow-x-auto">
{`- name: qwen3-8b
  model: Qwen/Qwen3-8B
  formats: [gguf, mlx]
  gguf_precision: Q4_K_M
  mlx_bits: 4
  output: ./models/qwen3-8b`}
            </pre>
          </div>

          <div>
            <strong className="text-green-600 dark:text-green-400">Vision Export:</strong>
            <pre className="mt-1 text-xs font-mono bg-white dark:bg-gray-800 p-2 rounded overflow-x-auto">
{`- name: yolo-v8n
  model: yolov8n
  backend: onnx
  precision: int8
  output: ./models/yolo-v8n`}
            </pre>
          </div>

          <div>
            <strong className="text-purple-600 dark:text-purple-400">Speech Export:</strong>
            <pre className="mt-1 text-xs font-mono bg-white dark:bg-gray-800 p-2 rounded overflow-x-auto">
{`- name: whisper-tiny
  model: whisper-tiny
  backend: coreml
  precision: fp16
  output: ./models/whisper-tiny`}
            </pre>
          </div>
        </div>
      </div>
    </div>
  )
}
