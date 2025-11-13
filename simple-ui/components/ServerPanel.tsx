'use client'

import { useState, useEffect } from 'react'
import { useModelConfigs } from './SettingsPanel'

interface ServerStatus {
  running: boolean
  port?: number
  modelPath?: string
  pid?: number
  serverType?: string
}

const SERVER_TYPES = [
  { id: 'gguf', name: 'GGUF', port: 8080, desc: 'Universal (CPU/GPU/Metal)' },
  { id: 'mlx', name: 'MLX', port: 8081, desc: 'Apple Silicon' },
  { id: 'multimodal', name: 'Multimodal', port: 8000, desc: 'Vision + Speech (CoreML)' },
  { id: 'coreml', name: 'CoreML', port: 8082, desc: 'Apple Neural Engine' },
  { id: 'onnx', name: 'ONNX', port: 8083, desc: 'Cross-platform' },
  { id: 'tensorrt', name: 'TensorRT', port: 8084, desc: 'NVIDIA GPU' },
]

interface HardwareInfo {
  platform: string
  cpu: string
  ram_gb: number
  gpu?: string
  recommended_backend: string
}

export default function ServerPanel() {
  const modelConfigs = useModelConfigs()
  const [servers, setServers] = useState<{ [key: string]: ServerStatus }>({})
  const [selectedModelId, setSelectedModelId] = useState('')
  const [modelPath, setModelPath] = useState('')
  const [useCustomPath, setUseCustomPath] = useState(false)
  const [serverType, setServerType] = useState('gguf')
  const [port, setPort] = useState(8080)
  const [gpuLayers, setGpuLayers] = useState(99)
  const [contextSize, setContextSize] = useState(2048)
  const [hardware, setHardware] = useState<HardwareInfo | null>(null)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [isStarting, setIsStarting] = useState(false)
  const [statusMessage, setStatusMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null)

  useEffect(() => {
    // Poll server status
    const interval = setInterval(checkServerStatus, 2000)
    checkServerStatus()
    detectHardware()
    return () => clearInterval(interval)
  }, [])

  const detectHardware = async () => {
    try {
      const res = await fetch('/api/hardware/detect')
      const data = await res.json()
      setHardware(data.hardware)

      // Auto-configure based on hardware
      if (data.hardware) {
        // Set optimal server type
        if (data.hardware.recommended_backend === 'mlx') {
          setServerType('mlx')
          setPort(8081)
        } else {
          setServerType('gguf')
          setPort(8080)
        }

        // Set optimal GPU layers based on RAM
        if (data.hardware.ram_gb >= 64) {
          setGpuLayers(99) // All layers
        } else if (data.hardware.ram_gb >= 32) {
          setGpuLayers(50) // Half layers
        } else if (data.hardware.ram_gb >= 16) {
          setGpuLayers(25) // Quarter layers
        } else {
          setGpuLayers(0) // CPU only
        }

        // Set optimal context size based on RAM
        if (data.hardware.ram_gb >= 64) {
          setContextSize(8192) // Large context
        } else if (data.hardware.ram_gb >= 32) {
          setContextSize(4096) // Medium context
        } else {
          setContextSize(2048) // Standard context
        }
      }
    } catch (error) {
      console.error('Error detecting hardware:', error)
    }
  }

  const generateHardwareConfig = async () => {
    try {
      const res = await fetch('/api/hardware/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          serverType,
          gpuLayers,
          contextSize,
          port,
        }),
      })
      const data = await res.json()

      if (data.config) {
        // Download config as file
        const blob = new Blob([data.config], { type: 'application/x-yaml' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = 'runtime-config.yaml'
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        URL.revokeObjectURL(url)

        alert('Runtime configuration file downloaded!')
      }
    } catch (error) {
      console.error('Error generating config:', error)
      alert('Failed to generate configuration')
    }
  }

  const checkServerStatus = async () => {
    try {
      const res = await fetch('/api/server/status')
      const data = await res.json()
      setServers(data.servers || {})
    } catch (error) {
      console.error('Error checking server status:', error)
    }
  }

  const startServer = async () => {
    if (!modelPath) {
      setStatusMessage({ type: 'error', text: 'Please select a model first' })
      setTimeout(() => setStatusMessage(null), 5000)
      return
    }

    setIsStarting(true)
    setStatusMessage(null)

    try {
      const res = await fetch('/api/server/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          modelPath,
          serverType,
          port,
          gpuLayers,
          contextSize,
        }),
      })
      const data = await res.json()

      if (data.success) {
        setStatusMessage({
          type: 'success',
          text: `✅ Server started successfully on port ${port}! It will appear below in a few seconds.`
        })
        setTimeout(() => setStatusMessage(null), 5000)

        // Wait a bit before checking status to give server time to start
        setTimeout(() => checkServerStatus(), 2000)

        // Reset form
        setSelectedModelId('')
        setModelPath('')
        setUseCustomPath(false)
      } else {
        setStatusMessage({
          type: 'error',
          text: `❌ Error: ${data.error}`
        })
        setTimeout(() => setStatusMessage(null), 8000)
      }
    } catch (error) {
      console.error('Error starting server:', error)
      setStatusMessage({
        type: 'error',
        text: `❌ Error: ${error}`
      })
      setTimeout(() => setStatusMessage(null), 8000)
    } finally {
      setIsStarting(false)
    }
  }

  const stopServer = async (serverId: string) => {
    try {
      const res = await fetch('/api/server/stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ serverId }),
      })
      const data = await res.json()
      if (data.success) {
        checkServerStatus()
      }
    } catch (error) {
      console.error('Error stopping server:', error)
    }
  }

  const stopAllServers = async () => {
    for (const serverId of Object.keys(servers)) {
      await stopServer(serverId)
    }
  }

  const handleModelSelect = (modelId: string) => {
    setSelectedModelId(modelId)
    if (modelId === 'custom') {
      setUseCustomPath(true)
      return
    }

    const model = modelConfigs.find(m => m.id === modelId)
    if (model) {
      setModelPath(model.path)
      setUseCustomPath(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Model Servers</h2>
        {Object.keys(servers).length > 0 && (
          <button
            onClick={stopAllServers}
            className="px-3 py-1 bg-red-500 text-white rounded text-sm hover:bg-red-600"
          >
            Stop All Servers
          </button>
        )}
      </div>

      {/* Hardware Info */}
      {hardware && (
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded">
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-medium">🖥️ Detected Hardware</h3>
            <div className="flex gap-2">
              <button
                onClick={detectHardware}
                className="text-xs px-2 py-1 bg-blue-100 dark:bg-blue-800 rounded hover:bg-blue-200 dark:hover:bg-blue-700"
              >
                Refresh
              </button>
              <button
                onClick={generateHardwareConfig}
                className="text-xs px-2 py-1 bg-green-500 text-white rounded hover:bg-green-600"
              >
                Download Config
              </button>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div><strong>Platform:</strong> {hardware.platform}</div>
            <div><strong>RAM:</strong> {hardware.ram_gb} GB</div>
            <div><strong>CPU:</strong> {hardware.cpu}</div>
            {hardware.gpu && <div><strong>GPU:</strong> {hardware.gpu}</div>}
          </div>
          <div className="mt-2 p-2 bg-green-50 dark:bg-green-900/20 rounded text-sm">
            <strong>✨ Recommended:</strong> {hardware.recommended_backend.toUpperCase()}
            {hardware.recommended_backend === 'mlx' && ' (Apple Silicon optimized)'}
            {hardware.recommended_backend === 'gguf' && ' (Universal compatibility)'}
          </div>
        </div>
      )}

      {/* Start New Server */}
      <div className="p-4 bg-gray-50 dark:bg-gray-700/50 rounded">
        <h3 className="font-medium mb-3">Start New Server (Optimized Settings)</h3>

        {/* Model Selection */}
        <div className="mb-3">
          <label className="block text-sm font-medium mb-2">
            Select Model
            {modelConfigs.filter(m => m.type === 'llm').length === 0 && (
              <span className="ml-2 text-xs text-gray-500">(Configure models in Settings tab)</span>
            )}
          </label>
          <select
            value={selectedModelId || 'custom'}
            onChange={(e) => handleModelSelect(e.target.value)}
            className="w-full px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600 text-sm"
          >
            <option value="">-- Select a model --</option>
            {modelConfigs.filter(m => m.type === 'llm').map((model) => (
              <option key={model.id} value={model.id}>
                {model.name} ({model.backend.toUpperCase()})
              </option>
            ))}
            <option value="custom">Custom Path...</option>
          </select>
        </div>

        {/* Custom Path (if needed) */}
        {(useCustomPath || !selectedModelId) && (
          <div className="mb-3">
            <label className="block text-sm font-medium mb-2">Custom Model Path</label>
            <input
              type="text"
              value={modelPath}
              onChange={(e) => setModelPath(e.target.value)}
              className="w-full px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600 text-sm"
              placeholder="./quant/models/qwen3-1.7b/gguf/Q4_K_M/model.gguf"
            />
          </div>
        )}

        {/* Server Type - Auto-selected */}
        <div className="mb-3">
          <label className="block text-sm font-medium mb-2">
            Server Type {hardware && <span className="text-xs text-blue-500">(Auto-selected for your hardware)</span>}
          </label>
          <div className="grid grid-cols-3 gap-2">
            {SERVER_TYPES.map((type) => (
              <button
                key={type.id}
                onClick={() => {
                  setServerType(type.id)
                  setPort(type.port)
                }}
                className={`px-3 py-2 rounded text-sm ${
                  serverType === type.id
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-200 dark:bg-gray-700'
                }`}
                title={type.desc}
              >
                <div className="font-medium">
                  {type.name}
                  {hardware?.recommended_backend === type.id && ' ⭐'}
                </div>
                <div className="text-xs opacity-75">{type.desc}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Quick Settings Summary */}
        <div className="mb-3 p-3 bg-white dark:bg-gray-800 rounded border dark:border-gray-700">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">Optimized Settings</span>
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="text-xs px-2 py-1 bg-gray-200 dark:bg-gray-700 rounded hover:bg-gray-300 dark:hover:bg-gray-600"
            >
              {showAdvanced ? 'Hide' : 'Show'} Advanced
            </button>
          </div>
          <div className="text-xs space-y-1 text-gray-600 dark:text-gray-400">
            <div>Port: <strong>{port}</strong></div>
            <div>GPU Layers: <strong>{gpuLayers}</strong> {hardware && `(Based on ${hardware.ram_gb}GB RAM)`}</div>
            <div>Context: <strong>{contextSize}</strong> {hardware && `(Optimized for ${hardware.ram_gb}GB RAM)`}</div>
          </div>
        </div>

        {/* Advanced Options - Collapsible */}
        {showAdvanced && (
          <div className="mb-3 p-3 bg-yellow-50 dark:bg-yellow-900/10 border border-yellow-200 dark:border-yellow-800 rounded">
            <p className="text-xs text-yellow-800 dark:text-yellow-300 mb-3">
              ⚠️ Advanced: These settings are auto-optimized for your hardware. Only change if needed.
            </p>
            <div className="grid grid-cols-3 gap-3">
              {/* Port */}
              <div>
                <label className="block text-sm font-medium mb-2">Port</label>
                <input
                  type="number"
                  value={port}
                  onChange={(e) => setPort(parseInt(e.target.value))}
                  className="w-full px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600 text-sm"
                  min="3000"
                  max="9999"
                />
              </div>

              {/* GPU Layers */}
              <div>
                <label className="block text-sm font-medium mb-2">GPU Layers</label>
                <input
                  type="number"
                  value={gpuLayers}
                  onChange={(e) => setGpuLayers(parseInt(e.target.value))}
                  className="w-full px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600 text-sm"
                  min="0"
                  max="99"
                />
              </div>

              {/* Context Size */}
              <div>
                <label className="block text-sm font-medium mb-2">Context Size</label>
                <input
                  type="number"
                  value={contextSize}
                  onChange={(e) => setContextSize(parseInt(e.target.value))}
                  className="w-full px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600 text-sm"
                  min="512"
                  max="8192"
                  step="512"
                />
              </div>
            </div>
          </div>
        )}

        <button
          onClick={startServer}
          disabled={!modelPath || isStarting}
          className="w-full px-4 py-2 bg-green-500 text-white rounded font-medium hover:bg-green-600 disabled:bg-gray-400 disabled:cursor-not-allowed"
        >
          {isStarting ? '⏳ Starting Server...' : 'Start Optimized Server'}
        </button>

        {/* Status Message */}
        {statusMessage && (
          <div className={`mt-3 p-3 rounded ${
            statusMessage.type === 'success'
              ? 'bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 text-green-800 dark:text-green-200'
              : 'bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-800 dark:text-red-200'
          }`}>
            <p className="text-sm font-medium">{statusMessage.text}</p>
          </div>
        )}
      </div>

      {/* Running Servers */}
      <div>
        <h3 className="font-medium mb-3">Running Servers ({Object.keys(servers).length})</h3>
        {Object.keys(servers).length === 0 ? (
          <p className="text-gray-500 dark:text-gray-400 text-sm text-center py-8">No servers running</p>
        ) : (
          <div className="space-y-3">
            {Object.entries(servers).map(([id, server]) => (
              <div
                key={id}
                className="p-4 border rounded dark:border-gray-700 flex items-start justify-between"
              >
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                    <span className="font-medium">{id}</span>
                    <span className="px-2 py-0.5 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 rounded text-xs">
                      {server.serverType?.toUpperCase()}
                    </span>
                  </div>
                  <div className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                    <div className="flex items-center gap-2">
                      <strong>Port:</strong>
                      <code className="px-2 py-0.5 bg-gray-100 dark:bg-gray-800 rounded text-xs">
                        {server.port}
                      </code>
                      <a
                        href={`http://localhost:${server.port}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-500 hover:underline text-xs"
                      >
                        Open →
                      </a>
                    </div>
                    <div><strong>Model:</strong> <span className="text-xs font-mono">{server.modelPath}</span></div>
                    <div><strong>PID:</strong> {server.pid}</div>
                  </div>
                </div>
                <button
                  onClick={() => stopServer(id)}
                  className="px-3 py-1 bg-red-500 text-white rounded text-sm hover:bg-red-600"
                >
                  Stop
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

    </div>
  )
}
