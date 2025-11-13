'use client'

import { useState, useEffect, useRef } from 'react'

interface ModelConfig {
  id: string
  name: string
  path: string
  backend: 'gguf' | 'mlx' | 'coreml' | 'onnx' | 'tensorrt'
  type: 'llm' | 'vision' | 'speech'
  description?: string
}

interface Settings {
  models: ModelConfig[]
  defaultMultimodalServer: string
  modelsDirectory: string  // Where to discover existing models
  exportDirectory: string   // Where to export new models
}

const DEFAULT_SETTINGS: Settings = {
  models: [],
  defaultMultimodalServer: 'http://localhost:8000',
  modelsDirectory: '/Users/robthelen/llama-pajamas/quant/models',
  exportDirectory: '/Users/robthelen/llama-pajamas/models',
}

export default function SettingsPanel() {
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS)
  const [editingModel, setEditingModel] = useState<ModelConfig | null>(null)
  const [isAdding, setIsAdding] = useState(false)

  // Form states
  const [formName, setFormName] = useState('')
  const [formPath, setFormPath] = useState('')
  const [formBackend, setFormBackend] = useState<'gguf' | 'mlx' | 'coreml' | 'onnx' | 'tensorrt'>('gguf')
  const [formType, setFormType] = useState<'llm' | 'vision' | 'speech'>('llm')
  const [formDescription, setFormDescription] = useState('')

  // Directory scanning states
  const [isScanning, setIsScanning] = useState(false)
  const [discoveredModels, setDiscoveredModels] = useState<ModelConfig[]>([])
  const [selectedModels, setSelectedModels] = useState<Set<number>>(new Set())

  useEffect(() => {
    // Load from localStorage
    loadSettings()

    // Migrate old relative paths to absolute paths
    migrateOldPaths()
  }, [])

  const migrateOldPaths = () => {
    try {
      const stored = localStorage.getItem('llamafarm-settings')
      if (!stored) return

      const settings: Settings = JSON.parse(stored)
      let needsSave = false

      // Check if any models have relative paths (starting with ./)
      const migratedModels = settings.models.map(model => {
        if (model.path.startsWith('./')) {
          needsSave = true
          // Try to convert to absolute path using modelsDirectory as base
          if (settings.modelsDirectory) {
            // Remove ./ and join with models directory
            const relativePart = model.path.substring(2)
            const absolutePath = settings.modelsDirectory.endsWith('/')
              ? settings.modelsDirectory + relativePart
              : settings.modelsDirectory + '/' + relativePart
            return {
              ...model,
              path: absolutePath
            }
          }
        }
        return model
      })

      if (needsSave) {
        const newSettings = { ...settings, models: migratedModels }
        localStorage.setItem('llamafarm-settings', JSON.stringify(newSettings))
        setSettings(newSettings)
        console.log('Migrated relative paths to absolute paths')
      }
    } catch (error) {
      console.error('Error migrating paths:', error)
    }
  }

  const loadSettings = () => {
    try {
      const stored = localStorage.getItem('llamafarm-settings')
      if (stored) {
        setSettings(JSON.parse(stored))
      }
    } catch (error) {
      console.error('Error loading settings:', error)
    }
  }

  const saveSettings = (newSettings: Settings) => {
    try {
      localStorage.setItem('llamafarm-settings', JSON.stringify(newSettings))
      setSettings(newSettings)
    } catch (error) {
      console.error('Error saving settings:', error)
    }
  }

  const handleAddModel = () => {
    if (!formName || !formPath) {
      alert('Name and path are required')
      return
    }

    const newModel: ModelConfig = {
      id: Date.now().toString(),
      name: formName,
      path: formPath,
      backend: formBackend,
      type: formType,
      description: formDescription,
    }

    const newSettings = {
      ...settings,
      models: [...settings.models, newModel],
    }

    saveSettings(newSettings)
    resetForm()
    setIsAdding(false)
  }

  const handleUpdateModel = () => {
    if (!editingModel || !formName || !formPath) {
      alert('Name and path are required')
      return
    }

    const updatedModel: ModelConfig = {
      ...editingModel,
      name: formName,
      path: formPath,
      backend: formBackend,
      type: formType,
      description: formDescription,
    }

    const newSettings = {
      ...settings,
      models: settings.models.map(m => m.id === editingModel.id ? updatedModel : m),
    }

    saveSettings(newSettings)
    setEditingModel(null)
    resetForm()
  }

  const handleDeleteModel = (id: string) => {
    if (!confirm('Are you sure you want to delete this model configuration?')) {
      return
    }

    const newSettings = {
      ...settings,
      models: settings.models.filter(m => m.id !== id),
    }

    saveSettings(newSettings)
  }

  const handleEditModel = (model: ModelConfig) => {
    setEditingModel(model)
    setFormName(model.name)
    setFormPath(model.path)
    setFormBackend(model.backend)
    setFormType(model.type)
    setFormDescription(model.description || '')
    setIsAdding(false)
  }

  const resetForm = () => {
    setFormName('')
    setFormPath('')
    setFormBackend('gguf')
    setFormType('llm')
    setFormDescription('')
  }

  const handleCancelEdit = () => {
    setEditingModel(null)
    setIsAdding(false)
    resetForm()
  }

  const handleExportSettings = () => {
    const dataStr = JSON.stringify(settings, null, 2)
    const blob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'llamafarm-settings.json'
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const handleImportSettings = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (e) => {
      try {
        const imported = JSON.parse(e.target?.result as string)
        saveSettings(imported)
        alert('Settings imported successfully!')
      } catch (error) {
        alert('Error importing settings: ' + error)
      }
    }
    reader.readAsText(file)
  }

  const handleScanDirectories = async () => {
    if (!settings.modelsDirectory && !settings.exportDirectory) {
      alert('Please configure at least one directory path in settings')
      return
    }

    setIsScanning(true)
    setDiscoveredModels([])
    setSelectedModels(new Set())

    try {
      const allModels: ModelConfig[] = []

      // Scan models directory
      if (settings.modelsDirectory) {
        const res1 = await fetch('/api/models/discover', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ directory: settings.modelsDirectory }),
        })
        const data1 = await res1.json()
        if (data1.success) {
          const models1 = data1.models.map((m: any) => ({
            id: `models-${Math.random().toString(36)}`,
            name: m.name,
            path: m.path,
            backend: m.backend,
            type: m.type,
            description: `${m.sizeFormatted}${m.precision ? ` | ${m.precision}` : ''} | From models dir`,
          }))
          allModels.push(...models1)
        }
      }

      // Scan export directory
      if (settings.exportDirectory && settings.exportDirectory !== settings.modelsDirectory) {
        const res2 = await fetch('/api/models/discover', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ directory: settings.exportDirectory }),
        })
        const data2 = await res2.json()
        if (data2.success) {
          const models2 = data2.models.map((m: any) => ({
            id: `export-${Math.random().toString(36)}`,
            name: m.name,
            path: m.path,
            backend: m.backend,
            type: m.type,
            description: `${m.sizeFormatted}${m.precision ? ` | ${m.precision}` : ''} | From export dir`,
          }))
          allModels.push(...models2)
        }
      }

      setDiscoveredModels(allModels)
      // Auto-select all by default
      setSelectedModels(new Set(allModels.map((_, idx) => idx)))
    } catch (error) {
      alert(`Error: ${error}`)
    } finally {
      setIsScanning(false)
    }
  }

  const handleToggleModel = (idx: number) => {
    const newSelected = new Set(selectedModels)
    if (newSelected.has(idx)) {
      newSelected.delete(idx)
    } else {
      newSelected.add(idx)
    }
    setSelectedModels(newSelected)
  }

  const handleToggleAll = () => {
    if (selectedModels.size === discoveredModels.length) {
      setSelectedModels(new Set())
    } else {
      setSelectedModels(new Set(discoveredModels.map((_, idx) => idx)))
    }
  }

  const handleImportSelected = () => {
    if (selectedModels.size === 0) {
      alert('No models selected')
      return
    }

    const modelsToImport = discoveredModels
      .filter((_, idx) => selectedModels.has(idx))
      .map(m => ({
        ...m,
        id: Date.now().toString() + Math.random().toString(36),
      }))

    const newSettings = {
      ...settings,
      models: [...settings.models, ...modelsToImport],
    }

    saveSettings(newSettings)
    setDiscoveredModels([])
    setSelectedModels(new Set())
    alert(`Successfully imported ${modelsToImport.length} model(s)!`)
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold mb-4">Settings</h2>
        <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
          Configure and save model paths for easy access across tabs
        </p>

        {/* Import/Export */}
        <div className="mb-6 flex gap-2">
          <button
            onClick={handleExportSettings}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Export Settings
          </button>
          <label className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 cursor-pointer">
            Import Settings
            <input
              type="file"
              accept=".json"
              onChange={handleImportSettings}
              className="hidden"
            />
          </label>
        </div>

        {/* Directory Settings */}
        <div className="mb-6 p-4 bg-blue-50 dark:bg-blue-900/10 border-2 border-blue-200 dark:border-blue-800 rounded">
          <h3 className="text-lg font-medium mb-3">📁 Directory Paths</h3>
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
            Configure where to discover existing models and where to export new models. Use absolute paths.
          </p>

          {/* Models Directory */}
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">
              Models Directory
              <span className="ml-2 text-xs text-gray-500">(Where to find existing models)</span>
            </label>
            <input
              type="text"
              value={settings.modelsDirectory}
              onChange={(e) => saveSettings({ ...settings, modelsDirectory: e.target.value })}
              className="w-full px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600 font-mono text-sm"
              placeholder="/Users/robthelen/llama-pajamas/quant/models"
            />
          </div>

          {/* Export Directory */}
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">
              Export Directory
              <span className="ml-2 text-xs text-gray-500">(Where to export new models)</span>
            </label>
            <input
              type="text"
              value={settings.exportDirectory}
              onChange={(e) => saveSettings({ ...settings, exportDirectory: e.target.value })}
              className="w-full px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600 font-mono text-sm"
              placeholder="/Users/robthelen/llama-pajamas/models"
            />
          </div>
        </div>

        {/* Directory Scanner */}
        <div className="mb-6 p-4 bg-purple-50 dark:bg-purple-900/10 border-2 border-purple-200 dark:border-purple-800 rounded">
          <h3 className="text-lg font-medium mb-3">🔍 Discover Models</h3>
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
            Scan both configured directories to automatically discover all model files.
          </p>

          <button
            onClick={handleScanDirectories}
            disabled={isScanning || (!settings.modelsDirectory && !settings.exportDirectory)}
            className="w-full px-6 py-3 bg-purple-500 text-white rounded hover:bg-purple-600 disabled:bg-gray-400 disabled:cursor-not-allowed font-medium"
          >
            {isScanning ? 'Scanning Both Directories...' : 'Scan for Models'}
          </button>

          {(!settings.modelsDirectory && !settings.exportDirectory) && (
            <p className="text-xs text-red-500 dark:text-red-400 mt-2">
              ⚠️ Please configure at least one directory path above first
            </p>
          )}

          {/* Discovered Models */}
          {discoveredModels.length > 0 && (
            <div className="p-4 bg-white dark:bg-gray-800 rounded border dark:border-gray-700">
              <div className="flex items-center justify-between mb-3">
                <h4 className="font-medium">Found {discoveredModels.length} model(s)</h4>
                <div className="flex gap-2">
                  <button
                    onClick={handleToggleAll}
                    className="px-3 py-1 text-sm bg-gray-200 dark:bg-gray-700 rounded hover:bg-gray-300 dark:hover:bg-gray-600"
                  >
                    {selectedModels.size === discoveredModels.length ? 'Deselect All' : 'Select All'}
                  </button>
                  <button
                    onClick={handleImportSelected}
                    disabled={selectedModels.size === 0}
                    className="px-3 py-1 text-sm bg-green-500 text-white rounded hover:bg-green-600 disabled:bg-gray-400 disabled:cursor-not-allowed"
                  >
                    Import Selected ({selectedModels.size})
                  </button>
                </div>
              </div>

              <div className="space-y-2 max-h-96 overflow-y-auto">
                {discoveredModels.map((model, idx) => (
                  <label
                    key={idx}
                    className={`flex items-start p-3 rounded border cursor-pointer ${
                      selectedModels.has(idx)
                        ? 'bg-blue-50 dark:bg-blue-900/20 border-blue-500'
                        : 'border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700/50'
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={selectedModels.has(idx)}
                      onChange={() => handleToggleModel(idx)}
                      className="mt-1 mr-3"
                    />
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-medium text-sm">{model.name}</span>
                        <span className="px-2 py-0.5 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 rounded text-xs uppercase">
                          {model.backend}
                        </span>
                        <span className="px-2 py-0.5 bg-purple-100 dark:bg-purple-900 text-purple-800 dark:text-purple-200 rounded text-xs capitalize">
                          {model.type}
                        </span>
                      </div>
                      <div className="text-xs font-mono text-gray-600 dark:text-gray-400 mb-1">
                        {model.path}
                      </div>
                      {model.description && (
                        <div className="text-xs text-gray-500 dark:text-gray-400">
                          {model.description}
                        </div>
                      )}
                    </div>
                  </label>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Model Configurations */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium">Model Configurations ({settings.models.length})</h3>
            <div className="flex gap-2">
              {settings.models.length > 0 && (
                <button
                  onClick={() => {
                    if (confirm(`Delete all ${settings.models.length} model configurations? This cannot be undone.`)) {
                      saveSettings({ ...settings, models: [] })
                      alert('All models cleared!')
                    }
                  }}
                  className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
                >
                  Clear All Models
                </button>
              )}
              <button
                onClick={() => {
                  setIsAdding(true)
                  setEditingModel(null)
                  resetForm()
                }}
                className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
              >
                + Add Model
              </button>
            </div>
          </div>

          {/* Model List */}
          <div className="space-y-3 mb-4">
            {settings.models.length === 0 && (
              <p className="text-gray-500 dark:text-gray-400 text-center py-8">
                No models configured. Click "Add Model" to get started.
              </p>
            )}

            {settings.models.map((model) => (
              <div
                key={model.id}
                className="p-4 border rounded dark:border-gray-700 flex items-start justify-between"
              >
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="font-medium">{model.name}</span>
                    <span className="px-2 py-0.5 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 rounded text-xs uppercase">
                      {model.backend}
                    </span>
                    <span className="px-2 py-0.5 bg-purple-100 dark:bg-purple-900 text-purple-800 dark:text-purple-200 rounded text-xs capitalize">
                      {model.type}
                    </span>
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">
                    <div className="font-mono text-xs mb-1">{model.path}</div>
                    {model.description && (
                      <div className="text-xs">{model.description}</div>
                    )}
                  </div>
                </div>
                <div className="flex gap-2 ml-4">
                  <button
                    onClick={() => handleEditModel(model)}
                    className="px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600"
                  >
                    Edit
                  </button>
                  <button
                    onClick={() => handleDeleteModel(model.id)}
                    className="px-3 py-1 bg-red-500 text-white rounded text-sm hover:bg-red-600"
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))}
          </div>

          {/* Add/Edit Form */}
          {(isAdding || editingModel) && (
            <div className="p-4 bg-gray-50 dark:bg-gray-700/50 rounded border-2 border-blue-500">
              <h4 className="font-medium mb-4">
                {editingModel ? 'Edit Model' : 'Add New Model'}
              </h4>

              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Name *</label>
                  <input
                    type="text"
                    value={formName}
                    onChange={(e) => setFormName(e.target.value)}
                    className="w-full px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600"
                    placeholder="e.g., Qwen3-8B Q4_K_M"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">Type</label>
                  <select
                    value={formType}
                    onChange={(e) => setFormType(e.target.value as 'llm' | 'vision' | 'speech')}
                    className="w-full px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600"
                  >
                    <option value="llm">LLM</option>
                    <option value="vision">Vision</option>
                    <option value="speech">Speech</option>
                  </select>
                </div>
              </div>

              <div className="mb-4">
                <label className="block text-sm font-medium mb-2">Path *</label>
                <input
                  type="text"
                  value={formPath}
                  onChange={(e) => setFormPath(e.target.value)}
                  className="w-full px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600 font-mono text-sm"
                  placeholder="./models/qwen3-8b or ./models/qwen3-8b/gguf/Q4_K_M/model.gguf"
                />
              </div>

              <div className="mb-4">
                <label className="block text-sm font-medium mb-2">Backend</label>
                <div className="grid grid-cols-5 gap-2">
                  {(['gguf', 'mlx', 'coreml', 'onnx', 'tensorrt'] as const).map((b) => (
                    <button
                      key={b}
                      onClick={() => setFormBackend(b)}
                      className={`px-3 py-2 rounded text-sm capitalize ${
                        formBackend === b
                          ? 'bg-blue-500 text-white'
                          : 'bg-gray-200 dark:bg-gray-700'
                      }`}
                    >
                      {b}
                    </button>
                  ))}
                </div>
              </div>

              <div className="mb-4">
                <label className="block text-sm font-medium mb-2">Description (optional)</label>
                <input
                  type="text"
                  value={formDescription}
                  onChange={(e) => setFormDescription(e.target.value)}
                  className="w-full px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600 text-sm"
                  placeholder="e.g., Fast inference model for general chat"
                />
              </div>

              <div className="flex gap-2">
                <button
                  onClick={editingModel ? handleUpdateModel : handleAddModel}
                  className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
                >
                  {editingModel ? 'Update' : 'Add'}
                </button>
                <button
                  onClick={handleCancelEdit}
                  className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Other Settings */}
        <div className="mb-6">
          <h3 className="text-lg font-medium mb-4">Server Settings</h3>
          <div className="p-4 border rounded dark:border-gray-700">
            <label className="block text-sm font-medium mb-2">
              Default Multimodal Server URL
            </label>
            <input
              type="text"
              value={settings.defaultMultimodalServer}
              onChange={(e) => {
                const newSettings = {
                  ...settings,
                  defaultMultimodalServer: e.target.value,
                }
                saveSettings(newSettings)
              }}
              className="w-full px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600"
              placeholder="http://localhost:8000"
            />
          </div>
        </div>

        {/* Stats */}
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded">
          <h3 className="font-medium mb-2">📊 Configuration Stats</h3>
          <div className="grid grid-cols-4 gap-4 text-sm">
            <div>
              <div className="text-gray-600 dark:text-gray-400">Total Models</div>
              <div className="font-medium">{settings.models.length}</div>
            </div>
            <div>
              <div className="text-gray-600 dark:text-gray-400">LLM</div>
              <div className="font-medium">
                {settings.models.filter(m => m.type === 'llm').length}
              </div>
            </div>
            <div>
              <div className="text-gray-600 dark:text-gray-400">Vision</div>
              <div className="font-medium">
                {settings.models.filter(m => m.type === 'vision').length}
              </div>
            </div>
            <div>
              <div className="text-gray-600 dark:text-gray-400">Speech</div>
              <div className="font-medium">
                {settings.models.filter(m => m.type === 'speech').length}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// Export helper to use in other components
// Hook to access directory settings from other components
export function useDirectorySettings() {
  const [directories, setDirectories] = useState({
    modelsDirectory: '',
    exportDirectory: '',
  })

  useEffect(() => {
    const loadDirectories = () => {
      try {
        const stored = localStorage.getItem('llamafarm-settings')
        if (stored) {
          const settings: Settings = JSON.parse(stored)
          setDirectories({
            modelsDirectory: settings.modelsDirectory || '',
            exportDirectory: settings.exportDirectory || '',
          })
        }
      } catch (error) {
        console.error('Error loading directory settings:', error)
      }
    }

    loadDirectories()

    // Listen for storage changes
    const handleStorageChange = () => loadDirectories()
    window.addEventListener('storage', handleStorageChange)
    return () => window.removeEventListener('storage', handleStorageChange)
  }, [])

  return directories
}

export function useModelConfigs() {
  const [models, setModels] = useState<ModelConfig[]>([])

  useEffect(() => {
    const loadModels = () => {
      try {
        const stored = localStorage.getItem('llamafarm-settings')
        if (stored) {
          const settings: Settings = JSON.parse(stored)
          setModels(settings.models)
        }
      } catch (error) {
        console.error('Error loading model configs:', error)
      }
    }

    loadModels()

    // Listen for storage events
    const handleStorage = () => loadModels()
    window.addEventListener('storage', handleStorage)
    return () => window.removeEventListener('storage', handleStorage)
  }, [])

  return models
}
