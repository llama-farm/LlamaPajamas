'use client'

import { useState, useEffect } from 'react'
import { useDirectorySettings } from './SettingsPanel'

interface Model {
  name: string
  path: string
  format: string
  size: string
  precision?: string
}

export default function ModelsPanel() {
  const directorySettings = useDirectorySettings()
  const [selectedDirectory, setSelectedDirectory] = useState<'models' | 'export' | 'custom'>('models')
  const [customDir, setCustomDir] = useState('')
  const [models, setModels] = useState<Model[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [selectedModel, setSelectedModel] = useState<Model | null>(null)

  // Get the actual directory to scan based on selection
  const getModelsDir = () => {
    if (selectedDirectory === 'models') return directorySettings.modelsDirectory
    if (selectedDirectory === 'export') return directorySettings.exportDirectory
    return customDir
  }

  const scanModels = async () => {
    const dir = getModelsDir()
    if (!dir) {
      alert('Please configure directory paths in Settings tab first')
      return
    }

    setIsLoading(true)
    try {
      const res = await fetch('/api/models/scan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ directory: dir }),
      })
      const data = await res.json()
      setModels(data.models || [])
    } catch (error) {
      console.error('Error scanning models:', error)
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    // Only auto-scan if directories are configured
    if (directorySettings.modelsDirectory || directorySettings.exportDirectory) {
      scanModels()
    }
  }, [directorySettings])

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`
  }

  const copyPath = (path: string) => {
    navigator.clipboard.writeText(path)
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold mb-4">Available Models</h2>

        {/* Directory Selection */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">
            Select Directory
            {(!directorySettings.modelsDirectory && !directorySettings.exportDirectory) && (
              <span className="ml-2 text-xs text-red-500">(Configure in Settings tab first)</span>
            )}
          </label>
          <div className="flex gap-2">
            <select
              value={selectedDirectory}
              onChange={(e) => setSelectedDirectory(e.target.value as 'models' | 'export' | 'custom')}
              className="flex-1 px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600"
              disabled={!directorySettings.modelsDirectory && !directorySettings.exportDirectory}
            >
              {directorySettings.modelsDirectory && (
                <option value="models">
                  Models Directory ({directorySettings.modelsDirectory})
                </option>
              )}
              {directorySettings.exportDirectory && (
                <option value="export">
                  Export Directory ({directorySettings.exportDirectory})
                </option>
              )}
              <option value="custom">Custom Path...</option>
            </select>
            <button
              onClick={scanModels}
              disabled={isLoading || !getModelsDir()}
              className="px-4 py-2 bg-blue-500 text-white rounded font-medium hover:bg-blue-600 disabled:bg-gray-400 whitespace-nowrap"
            >
              {isLoading ? 'Scanning...' : 'Scan Directory'}
            </button>
          </div>

          {/* Custom Path Input */}
          {selectedDirectory === 'custom' && (
            <input
              type="text"
              value={customDir}
              onChange={(e) => setCustomDir(e.target.value)}
              className="w-full mt-2 px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600 font-mono text-sm"
              placeholder="/absolute/path/to/models"
            />
          )}
        </div>

        {/* Models List */}
        {models.length === 0 ? (
          <div className="text-center text-gray-500 dark:text-gray-400 py-8">
            <p>No models found. Click "Scan Directory" to search for models.</p>
          </div>
        ) : (
          <div className="space-y-2">
            {models.map((model, idx) => (
              <div
                key={idx}
                onClick={() => setSelectedModel(model)}
                className={`p-4 border rounded cursor-pointer transition-colors ${
                  selectedModel === model
                    ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                    : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="font-medium mb-1">{model.name}</div>
                    <div className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                      <div className="flex items-center gap-2">
                        <span className="font-medium">Format:</span>
                        <span className="px-2 py-0.5 bg-gray-200 dark:bg-gray-700 rounded text-xs">
                          {model.format}
                        </span>
                        {model.precision && (
                          <span className="px-2 py-0.5 bg-gray-200 dark:bg-gray-700 rounded text-xs">
                            {model.precision}
                          </span>
                        )}
                      </div>
                      <div><span className="font-medium">Size:</span> {model.size}</div>
                      <div className="flex items-center gap-2">
                        <span className="font-medium">Path:</span>
                        <code className="text-xs bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded">
                          {model.path}
                        </code>
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            copyPath(model.path)
                          }}
                          className="text-xs px-2 py-1 bg-gray-200 dark:bg-gray-700 rounded hover:bg-gray-300 dark:hover:bg-gray-600"
                        >
                          Copy
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Selected Model Actions */}
        {selectedModel && (
          <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded">
            <h3 className="font-medium mb-3">Actions for: {selectedModel.name}</h3>
            <div className="flex gap-2">
              <button className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">
                Evaluate
              </button>
              <button className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600">
                Start Server
              </button>
              <button className="px-4 py-2 bg-purple-500 text-white rounded hover:bg-purple-600">
                Run Inference
              </button>
              <button
                onClick={() => copyPath(selectedModel.path)}
                className="px-4 py-2 bg-gray-200 dark:bg-gray-700 rounded hover:bg-gray-300 dark:hover:bg-gray-600"
              >
                Copy Path
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
