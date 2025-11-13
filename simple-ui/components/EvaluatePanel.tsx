'use client'

import { useState, useEffect } from 'react'
import { useModelConfigs } from './SettingsPanel'

interface EvaluationResult {
  modelPath: string
  modelName: string
  format: string
  precision?: string
  accuracy: number
  avgTime: number
  categories?: { [key: string]: number }
  timestamp: string
}

export default function EvaluatePanel() {
  const modelConfigs = useModelConfigs()
  const [selectedModelId, setSelectedModelId] = useState('')
  const [useCustomPath, setUseCustomPath] = useState(false)
  const [modelPath, setModelPath] = useState('')
  const [modelFormat, setModelFormat] = useState<'gguf' | 'mlx' | 'coreml' | 'onnx'>('gguf')
  const [evalType, setEvalType] = useState<'llm' | 'vision' | 'speech'>('llm')
  const [numQuestions, setNumQuestions] = useState(140)
  const [isRunning, setIsRunning] = useState(false)
  const [progress, setProgress] = useState('')
  const [allResults, setAllResults] = useState<EvaluationResult[]>([])
  const [showComparison, setShowComparison] = useState(true)

  useEffect(() => {
    // Load previous evaluation results
    loadPreviousResults()
  }, [])

  const loadPreviousResults = async () => {
    try {
      const res = await fetch('/api/evaluate/results')
      const data = await res.json()
      setAllResults(data.results || [])
    } catch (error) {
      console.error('Error loading results:', error)
    }
  }

  const handleEvaluate = async () => {
    setIsRunning(true)
    setProgress('Starting evaluation...\n')
    setShowComparison(false)

    try {
      const res = await fetch('/api/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          modelPath,
          format: modelFormat,
          evalType,
          numQuestions,
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
              if (data.results) {
                const newResult: EvaluationResult = {
                  modelPath,
                  modelName: modelPath.split('/').pop() || modelPath,
                  format: modelFormat,
                  ...data.results,
                  timestamp: new Date().toISOString(),
                }
                setAllResults(prev => [newResult, ...prev])
                setShowComparison(true)
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
        <h2 className="text-xl font-semibold mb-4">Evaluate Models</h2>

        <div className="grid grid-cols-2 gap-4">
          {/* Left: Run New Evaluation */}
          <div className="space-y-4">
            <h3 className="font-medium">Run New Evaluation</h3>

            {/* Evaluation Type */}
            <div>
              <label className="block text-sm font-medium mb-2">Type</label>
              <div className="flex gap-2">
                {(['llm', 'vision', 'speech'] as const).map((type) => (
                  <button
                    key={type}
                    onClick={() => setEvalType(type)}
                    className={`px-3 py-2 rounded capitalize text-sm ${
                      evalType === type
                        ? 'bg-blue-500 text-white'
                        : 'bg-gray-200 dark:bg-gray-700'
                    }`}
                  >
                    {type}
                  </button>
                ))}
              </div>
            </div>

            {/* Model Selection */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Select Model
                {modelConfigs.filter(m => m.type === evalType).length === 0 && (
                  <span className="ml-2 text-xs text-gray-500">(Configure models in Settings tab)</span>
                )}
              </label>
              <select
                value={selectedModelId || 'custom'}
                onChange={(e) => {
                  const modelId = e.target.value
                  setSelectedModelId(modelId)
                  if (modelId === 'custom') {
                    setUseCustomPath(true)
                    setModelPath('')
                  } else {
                    const model = modelConfigs.find(m => m.id === modelId)
                    if (model) {
                      setModelPath(model.path)
                      setModelFormat(model.backend as any)
                      setUseCustomPath(false)
                    }
                  }
                }}
                className="w-full px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600 text-sm"
              >
                <option value="">-- Select a model --</option>
                {modelConfigs.filter(m => m.type === evalType).map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name} ({model.backend.toUpperCase()})
                  </option>
                ))}
                <option value="custom">Custom Path...</option>
              </select>
            </div>

            {/* Custom Path (if needed) */}
            {(useCustomPath || !selectedModelId) && (
              <div>
                <label className="block text-sm font-medium mb-2">Custom Model Path</label>
                <input
                  type="text"
                  value={modelPath}
                  onChange={(e) => setModelPath(e.target.value)}
                  className="w-full px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600 text-sm font-mono"
                  placeholder="/absolute/path/to/model"
                />
              </div>
            )}

            {/* Format */}
            <div>
              <label className="block text-sm font-medium mb-2">Format</label>
              <div className="grid grid-cols-2 gap-2">
                {(['gguf', 'mlx', 'coreml', 'onnx'] as const).map((fmt) => (
                  <button
                    key={fmt}
                    onClick={() => setModelFormat(fmt)}
                    className={`px-3 py-2 rounded uppercase text-sm ${
                      modelFormat === fmt
                        ? 'bg-blue-500 text-white'
                        : 'bg-gray-200 dark:bg-gray-700'
                    }`}
                  >
                    {fmt}
                  </button>
                ))}
              </div>
            </div>

            {/* Questions Count (LLM only) */}
            {evalType === 'llm' && (
              <div>
                <label className="block text-sm font-medium mb-2">
                  Questions: {numQuestions}
                </label>
                <input
                  type="range"
                  min="10"
                  max="140"
                  step="10"
                  value={numQuestions}
                  onChange={(e) => setNumQuestions(parseInt(e.target.value))}
                  className="w-full"
                />
              </div>
            )}

            <button
              onClick={handleEvaluate}
              disabled={!modelPath || isRunning}
              className="w-full px-4 py-3 bg-blue-500 text-white rounded font-medium hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed"
            >
              {isRunning ? 'Evaluating...' : 'Run Evaluation'}
            </button>
          </div>

          {/* Right: Output */}
          <div>
            <h3 className="font-medium mb-3">Output</h3>
            {progress ? (
              <div className="p-4 bg-gray-100 dark:bg-gray-700 rounded h-96 overflow-y-auto">
                <pre className="text-xs font-mono whitespace-pre-wrap">{progress}</pre>
              </div>
            ) : (
              <div className="p-4 bg-gray-50 dark:bg-gray-700/50 rounded h-96 flex items-center justify-center text-gray-500">
                No evaluation running
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Comparison View */}
      {showComparison && allResults.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold mb-4">Evaluation Results - Comparison</h3>

          {/* Summary Table */}
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b dark:border-gray-700">
                  <th className="text-left p-2">Model</th>
                  <th className="text-left p-2">Format</th>
                  <th className="text-left p-2">Precision</th>
                  <th className="text-right p-2">Accuracy</th>
                  <th className="text-right p-2">Avg Time</th>
                  <th className="text-left p-2">Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {allResults.map((result, idx) => (
                  <tr
                    key={idx}
                    className="border-b dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700/50"
                  >
                    <td className="p-2 font-mono text-xs">{result.modelName}</td>
                    <td className="p-2">
                      <span className="px-2 py-1 bg-gray-200 dark:bg-gray-700 rounded text-xs">
                        {result.format.toUpperCase()}
                      </span>
                    </td>
                    <td className="p-2">
                      {result.precision && (
                        <span className="px-2 py-1 bg-gray-200 dark:bg-gray-700 rounded text-xs">
                          {result.precision}
                        </span>
                      )}
                    </td>
                    <td className="p-2 text-right">
                      <span
                        className={`font-bold ${
                          result.accuracy >= 95
                            ? 'text-green-600 dark:text-green-400'
                            : result.accuracy >= 90
                            ? 'text-yellow-600 dark:text-yellow-400'
                            : 'text-red-600 dark:text-red-400'
                        }`}
                      >
                        {result.accuracy.toFixed(1)}%
                      </span>
                    </td>
                    <td className="p-2 text-right">{result.avgTime.toFixed(2)}s</td>
                    <td className="p-2 text-xs text-gray-500">
                      {new Date(result.timestamp).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Category Breakdown for Best Model */}
          {allResults[0]?.categories && (
            <div className="mt-6">
              <h4 className="font-medium mb-3">
                Category Breakdown - {allResults[0].modelName}
              </h4>
              <div className="grid grid-cols-3 gap-4">
                {Object.entries(allResults[0].categories).map(([category, score]) => (
                  <div key={category} className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded">
                    <div className="text-sm capitalize mb-1">{category}</div>
                    <div className="text-2xl font-bold">{score}%</div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mt-2">
                      <div
                        className="bg-green-500 h-2 rounded-full"
                        style={{ width: `${score}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Best/Worst/Average Stats */}
          <div className="mt-6 grid grid-cols-3 gap-4">
            <div className="p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded">
              <div className="text-sm text-green-800 dark:text-green-300 mb-1">Best Accuracy</div>
              <div className="text-2xl font-bold">
                {Math.max(...allResults.map(r => r.accuracy)).toFixed(1)}%
              </div>
              <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                {allResults.find(r => r.accuracy === Math.max(...allResults.map(r => r.accuracy)))?.modelName}
              </div>
            </div>
            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded">
              <div className="text-sm text-blue-800 dark:text-blue-300 mb-1">Fastest</div>
              <div className="text-2xl font-bold">
                {Math.min(...allResults.map(r => r.avgTime)).toFixed(2)}s
              </div>
              <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                {allResults.find(r => r.avgTime === Math.min(...allResults.map(r => r.avgTime)))?.modelName}
              </div>
            </div>
            <div className="p-4 bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded">
              <div className="text-sm text-purple-800 dark:text-purple-300 mb-1">Average</div>
              <div className="text-2xl font-bold">
                {(allResults.reduce((acc, r) => acc + r.accuracy, 0) / allResults.length).toFixed(1)}%
              </div>
              <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                Across {allResults.length} models
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
