'use client'

import { useState, useRef } from 'react'
import { useModelConfigs } from './SettingsPanel'

type Mode = 'chat' | 'image' | 'voice'

export default function InferencePanel() {
  const [mode, setMode] = useState<Mode>('chat')
  const modelConfigs = useModelConfigs()

  // Chat mode states
  const [selectedModelId, setSelectedModelId] = useState('')
  const [modelPath, setModelPath] = useState('')
  const [backend, setBackend] = useState<'gguf' | 'mlx'>('gguf')
  const [useCustomPath, setUseCustomPath] = useState(false)
  const [prompt, setPrompt] = useState('')
  const [systemPrompt, setSystemPrompt] = useState('You are a helpful assistant.')
  const [showSystemPrompt, setShowSystemPrompt] = useState(false)
  const [temperature, setTemperature] = useState(0.7)
  const [maxTokens, setMaxTokens] = useState(200)
  const [messages, setMessages] = useState<{ role: string; content: string; time?: number; ttft?: number; startupTime?: number }[]>([])
  const [isGenerating, setIsGenerating] = useState(false)
  const [currentResponse, setCurrentResponse] = useState('')

  // Image mode states
  const [selectedImageModelId, setSelectedImageModelId] = useState('')
  const [imageModelPath, setImageModelPath] = useState('')
  const [imageBackend, setImageBackend] = useState<'coreml' | 'onnx' | 'tensorrt'>('coreml')
  const [useCustomImagePath, setUseCustomImagePath] = useState(false)
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [imageResults, setImageResults] = useState<any>(null)
  const [isProcessingImage, setIsProcessingImage] = useState(false)
  const [visionTaskType, setVisionTaskType] = useState<'classification' | 'localization' | 'detection' | 'segmentation'>('detection')
  const [annotatedImage, setAnnotatedImage] = useState<string | null>(null)
  const imageInputRef = useRef<HTMLInputElement>(null)

  // Voice mode states
  const [selectedVoiceModelId, setSelectedVoiceModelId] = useState('')
  const [voiceModelPath, setVoiceModelPath] = useState('')
  const [voiceBackend, setVoiceBackend] = useState<'coreml' | 'onnx'>('coreml')
  const [useCustomVoicePath, setUseCustomVoicePath] = useState(false)
  const [audioFile, setAudioFile] = useState<File | null>(null)
  const [transcription, setTranscription] = useState('')
  const [isProcessingAudio, setIsProcessingAudio] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null)
  const [recordedChunks, setRecordedChunks] = useState<Blob[]>([])
  const audioInputRef = useRef<HTMLInputElement>(null)

  // Chat mode handlers
  const handleSend = async () => {
    if (!prompt.trim() || !modelPath) return

    setIsGenerating(true)
    setCurrentResponse('')

    const userMessage = { role: 'user', content: prompt }
    setMessages(prev => [...prev, userMessage])

    const startTime = Date.now()
    let firstTokenTime: number | null = null

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          modelPath,
          backend,
          prompt,
          systemPrompt,  // Include system prompt
          maxTokens,
          temperature,
        }),
      })

      const reader = res.body?.getReader()
      const decoder = new TextDecoder()

      if (reader) {
        let fullResponse = ''
        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          const text = decoder.decode(value)
          const lines = text.split('\n').filter(l => l.trim())

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = JSON.parse(line.slice(6))

              if (data.chunk) {
                // Capture time to first token
                if (firstTokenTime === null) {
                  firstTokenTime = Date.now()
                }
                fullResponse += data.chunk
                setCurrentResponse(fullResponse)
              }

              if (data.status === 'complete') {
                const endTime = Date.now()
                const totalTime = (endTime - startTime) / 1000
                const ttft = firstTokenTime ? (firstTokenTime - startTime) / 1000 : 0
                const inferenceTime = totalTime - ttft

                setMessages(prev => [...prev, {
                  role: 'assistant',
                  content: fullResponse,
                  time: parseFloat(totalTime.toFixed(2)),
                  ttft: parseFloat(ttft.toFixed(3)),
                  startupTime: parseFloat(inferenceTime.toFixed(2)),
                }])
                setCurrentResponse('')
              }

              if (data.error) {
                setMessages(prev => [...prev, {
                  role: 'error',
                  content: `Error: ${data.error}`,
                }])
                setCurrentResponse('')
              }
            }
          }
        }
      }
    } catch (error) {
      setMessages(prev => [...prev, {
        role: 'error',
        content: `Error: ${error}`,
      }])
    } finally {
      setIsGenerating(false)
      setPrompt('')
    }
  }

  // Image mode handlers
  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setImageFile(file)
      const reader = new FileReader()
      reader.onloadend = () => {
        setImagePreview(reader.result as string)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleImageDetection = async () => {
    if (!imageFile || !imagePreview || !imageModelPath) {
      setImageResults({ error: 'Please select a vision model and upload an image' })
      return
    }

    setIsProcessingImage(true)
    setImageResults(null)
    setAnnotatedImage(null)

    try {
      // Create FormData with model path, backend, task type, and image
      const formData = new FormData()
      formData.append('modelPath', imageModelPath)
      formData.append('backend', imageBackend)
      formData.append('taskType', visionTaskType)
      formData.append('image', imageFile)

      const res = await fetch('/api/vision/inference', {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`)
      }

      if (!res.body) {
        throw new Error('No response body')
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()

      let buffer = ''
      let resultReceived = false

      while (true) {
        const { done, value } = await reader.read()
        if (done) {
          console.log('Vision inference stream ended')
          if (!resultReceived) {
            setImageResults({ error: 'No result received from inference' })
          }
          break
        }

        const text = decoder.decode(value, { stream: true })
        buffer += text

        // Split by double newline for complete SSE messages
        const messages = buffer.split('\n\n')
        buffer = messages.pop() || ''

        for (const message of messages) {
          if (!message.trim()) continue

          const lines = message.split('\n')
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6))

                if (data.progress) {
                  console.log('Progress:', data.progress)
                }
                if (data.result) {
                  resultReceived = true
                  // Handle different task types
                  if (data.result.detections) {
                    setImageResults({ detections: data.result.detections, count: data.result.detections.length })
                  } else if (data.result.predictions) {
                    setImageResults({ predictions: data.result.predictions, count: data.result.predictions.length })
                  } else if (data.result.segments) {
                    setImageResults({ segments: data.result.segments, count: data.result.segments.length })
                  }

                  // Display annotated image if provided
                  if (data.result.annotated_image) {
                    setAnnotatedImage(data.result.annotated_image)
                  }

                  console.log('Got vision result!')
                }
                if (data.error) {
                  setImageResults({ error: data.error })
                  console.error('Error:', data.error)
                }
              } catch (e) {
                console.error('Parse error:', e)
              }
            }
          }
        }
      }
    } catch (error: any) {
      setImageResults({ error: error.message })
      console.error('Vision inference error:', error)
    } finally {
      setIsProcessingImage(false)
    }
  }

  // Voice mode handlers
  const handleAudioSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setAudioFile(file)
    }
  }

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)
      const chunks: Blob[] = []

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunks.push(e.data)
        }
      }

      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'audio/webm' })
        const file = new File([blob], 'recording.webm', { type: 'audio/webm' })
        setAudioFile(file)
        setRecordedChunks([])

        // Stop all tracks
        stream.getTracks().forEach(track => track.stop())
      }

      recorder.start()
      setMediaRecorder(recorder)
      setIsRecording(true)
      setRecordedChunks(chunks)
    } catch (error) {
      console.error('Error starting recording:', error)
      alert('Error accessing microphone. Please check permissions.')
    }
  }

  const stopRecording = () => {
    if (mediaRecorder && isRecording) {
      mediaRecorder.stop()
      setIsRecording(false)
      setMediaRecorder(null)
    }
  }

  const handleTranscribe = async () => {
    if (!audioFile || !voiceModelPath) {
      setTranscription('Please select a speech model and upload/record audio')
      return
    }

    setIsProcessingAudio(true)
    setTranscription('')

    try {
      // Create FormData with model path, backend, and audio
      const formData = new FormData()
      formData.append('modelPath', voiceModelPath)
      formData.append('backend', voiceBackend)
      formData.append('audio', audioFile)

      const res = await fetch('/api/speech/transcribe', {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`)
      }

      if (!res.body) {
        throw new Error('No response body')
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()

      let buffer = ''
      let resultReceived = false

      while (true) {
        const { done, value } = await reader.read()
        if (done) {
          console.log('Stream ended')
          if (!resultReceived) {
            setTranscription('No transcription received')
          }
          break
        }

        const text = decoder.decode(value, { stream: true })
        buffer += text

        // Split by double newline for complete SSE messages
        const messages = buffer.split('\n\n')
        buffer = messages.pop() || ''

        for (const message of messages) {
          if (!message.trim()) continue

          const lines = message.split('\n')
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6))

                if (data.progress) {
                  console.log('Progress:', data.progress)
                }
                if (data.result) {
                  resultReceived = true
                  setTranscription(data.result.text || JSON.stringify(data.result))
                  console.log('Got transcription!')
                }
                if (data.error) {
                  setTranscription(`Error: ${data.error}`)
                  console.error('Error:', data.error)
                }
              } catch (e) {
                console.error('Parse error:', e)
              }
            }
          }
        }
      }
    } catch (error) {
      console.error('Transcription error:', error)
      setTranscription(`Error: ${error}`)
    } finally {
      setIsProcessingAudio(false)
    }
  }

  const clearHistory = () => {
    setMessages([])
    setCurrentResponse('')
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
      if (model.backend === 'gguf' || model.backend === 'mlx') {
        setBackend(model.backend)
      }
      setUseCustomPath(false)
    }
  }

  // Determine available vision tasks based on model
  const getAvailableVisionTasks = (modelPath: string) => {
    const modelName = modelPath.toLowerCase()
    const isYolo = modelName.includes('yolo')
    const isClassification = modelName.includes('vit') || modelName.includes('clip')

    if (isYolo) {
      // YOLO models support all tasks
      return ['classification', 'localization', 'detection', 'segmentation'] as const
    } else if (isClassification) {
      // Classification models only support classification and localization
      return ['classification', 'localization'] as const
    }

    // Default: detection models
    return ['detection', 'localization'] as const
  }

  const handleImageModelSelect = (modelId: string) => {
    setSelectedImageModelId(modelId)
    if (modelId === 'custom') {
      setUseCustomImagePath(true)
      return
    }

    const model = modelConfigs.find(m => m.id === modelId)
    if (model) {
      setImageModelPath(model.path)
      if (model.backend === 'coreml' || model.backend === 'onnx' || model.backend === 'tensorrt') {
        setImageBackend(model.backend)
      }
      setUseCustomImagePath(false)

      // Set default task type based on model capabilities
      const availableTasks = getAvailableVisionTasks(model.path)
      const modelName = model.path.toLowerCase()
      if (modelName.includes('yolo')) {
        setVisionTaskType('detection')
      } else if (modelName.includes('vit') || modelName.includes('clip')) {
        setVisionTaskType('classification')
      } else {
        setVisionTaskType('detection')
      }
    }
  }

  const handleVoiceModelSelect = (modelId: string) => {
    setSelectedVoiceModelId(modelId)
    if (modelId === 'custom') {
      setUseCustomVoicePath(true)
      return
    }

    const model = modelConfigs.find(m => m.id === modelId)
    if (model) {
      setVoiceModelPath(model.path)
      if (model.backend === 'coreml' || model.backend === 'onnx') {
        setVoiceBackend(model.backend)
      }
      setUseCustomVoicePath(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold mb-4">Inference</h2>

        {/* Mode Selection */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">Mode</label>
          <div className="flex gap-2">
            {(['chat', 'image', 'voice'] as Mode[]).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={`px-4 py-2 rounded capitalize ${
                  mode === m
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-200 dark:bg-gray-700'
                }`}
              >
                {m === 'chat' && '💬 '}
                {m === 'image' && '🖼️ '}
                {m === 'voice' && '🎤 '}
                {m}
              </button>
            ))}
          </div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
            {mode === 'chat' && 'Direct LLM inference using llama_pajamas_run'}
            {mode === 'image' && 'Vision inference via multimodal server (port 8000)'}
            {mode === 'voice' && 'Speech-to-text via multimodal server (port 8000)'}
          </p>
        </div>

        {/* Chat Mode */}
        {mode === 'chat' && (
          <>
            <div className="mb-4">
              <label className="block text-sm font-medium mb-2">
                Select Model
                {modelConfigs.filter(m => m.type === 'llm').length === 0 && (
                  <span className="ml-2 text-xs text-gray-500">(Configure models in Settings tab)</span>
                )}
              </label>
              <select
                value={selectedModelId || 'custom'}
                onChange={(e) => handleModelSelect(e.target.value)}
                className="w-full px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600"
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

            {(useCustomPath || !selectedModelId) && (
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Custom Model Path</label>
                  <input
                    type="text"
                    value={modelPath}
                    onChange={(e) => setModelPath(e.target.value)}
                    className="w-full px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600 text-sm"
                    placeholder="./models/qwen3-8b or ./models/qwen3-8b/gguf/Q4_K_M/model.gguf"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">Backend</label>
                  <select
                    value={backend}
                    onChange={(e) => setBackend(e.target.value as 'gguf' | 'mlx')}
                    className="w-full px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600"
                  >
                    <option value="gguf">GGUF (Universal)</option>
                    <option value="mlx">MLX (Apple Silicon)</option>
                  </select>
                </div>
              </div>
            )}

            {/* System Prompt Section */}
            <div className="mb-4">
              <div className="flex items-center justify-between mb-2">
                <label className="block text-sm font-medium">System Prompt</label>
                <button
                  onClick={() => setShowSystemPrompt(!showSystemPrompt)}
                  className="text-xs text-blue-500 hover:text-blue-600"
                >
                  {showSystemPrompt ? 'Hide' : 'Show'}
                </button>
              </div>
              {showSystemPrompt && (
                <textarea
                  value={systemPrompt}
                  onChange={(e) => setSystemPrompt(e.target.value)}
                  className="w-full px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600 text-sm"
                  rows={3}
                  placeholder="You are a helpful assistant."
                />
              )}
              {!showSystemPrompt && (
                <div className="text-xs text-gray-500 dark:text-gray-400 italic">
                  "{systemPrompt.substring(0, 50)}{systemPrompt.length > 50 ? '...' : ''}"
                </div>
              )}
            </div>

            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <label className="block text-sm font-medium mb-2">
                  Temperature: {temperature}
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">
                  Max Tokens: {maxTokens}
                </label>
                <input
                  type="range"
                  min="50"
                  max="2000"
                  step="50"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                  className="w-full"
                />
              </div>
            </div>

            <div className="mb-4 p-4 bg-gray-50 dark:bg-gray-700/50 rounded h-96 overflow-y-auto">
              {messages.length === 0 && (
                <p className="text-gray-500 dark:text-gray-400 text-center py-8">
                  No messages yet. Enter a model path and start chatting!
                </p>
              )}

              {messages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`mb-3 p-3 rounded ${
                    msg.role === 'user'
                      ? 'bg-blue-100 dark:bg-blue-900/30 ml-8'
                      : msg.role === 'error'
                      ? 'bg-red-100 dark:bg-red-900/30'
                      : 'bg-gray-100 dark:bg-gray-800 mr-8'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-medium capitalize">{msg.role}</span>
                    {msg.time && (
                      <span className="text-xs text-gray-500">
                        {msg.time}s
                      </span>
                    )}
                  </div>
                  <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                </div>
              ))}

              {currentResponse && (
                <div className="mb-3 p-3 rounded bg-gray-100 dark:bg-gray-800 mr-8">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-medium">Assistant</span>
                    <span className="text-xs text-blue-500 animate-pulse">Generating...</span>
                  </div>
                  <p className="text-sm whitespace-pre-wrap">{currentResponse}</p>
                </div>
              )}
            </div>

            <div className="flex gap-2">
              <input
                type="text"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && !isGenerating && handleSend()}
                className="flex-1 px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600"
                placeholder="Type your message..."
                disabled={isGenerating || !modelPath}
              />
              <button
                onClick={handleSend}
                disabled={!prompt.trim() || isGenerating || !modelPath}
                className="px-6 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed"
              >
                {isGenerating ? 'Generating...' : 'Send'}
              </button>
              <button
                onClick={clearHistory}
                className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
              >
                Clear
              </button>
            </div>

            {messages.length > 0 && (
              <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded">
                <h3 className="text-sm font-medium mb-2">📊 Session Analytics</h3>
                <div className="grid grid-cols-4 gap-4 text-sm">
                  <div>
                    <div className="text-gray-600 dark:text-gray-400">Messages</div>
                    <div className="font-medium">{messages.filter(m => m.role !== 'error').length}</div>
                  </div>
                  <div>
                    <div className="text-gray-600 dark:text-gray-400">Time to First Token</div>
                    <div className="font-medium">
                      {messages.filter(m => m.ttft).length > 0
                        ? (messages.filter(m => m.ttft).reduce((sum, m) => sum + (m.ttft || 0), 0) / messages.filter(m => m.ttft).length).toFixed(3)
                        : '0.000'}s
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-600 dark:text-gray-400">Avg Inference</div>
                    <div className="font-medium">
                      {messages.filter(m => m.startupTime).length > 0
                        ? (messages.filter(m => m.startupTime).reduce((sum, m) => sum + (m.startupTime || 0), 0) / messages.filter(m => m.startupTime).length).toFixed(2)
                        : '0.00'}s
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-600 dark:text-gray-400">Backend</div>
                    <div className="font-medium uppercase">{backend}</div>
                  </div>
                </div>
              </div>
            )}
          </>
        )}

        {/* Image Mode */}
        {mode === 'image' && (
          <>
            {/* Vision Model Selection */}
            <div className="mb-4">
              <label className="block text-sm font-medium mb-2">
                Select Vision Model
                {modelConfigs.filter(m => m.type === 'vision').length === 0 && (
                  <span className="ml-2 text-xs text-gray-500">(Configure vision models in Settings tab)</span>
                )}
              </label>
              <select
                value={selectedImageModelId || 'custom'}
                onChange={(e) => handleImageModelSelect(e.target.value)}
                className="w-full px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600"
              >
                <option value="">-- Select a vision model --</option>
                {modelConfigs.filter(m => m.type === 'vision').map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name} ({model.backend.toUpperCase()})
                  </option>
                ))}
                <option value="custom">Custom Path...</option>
              </select>
            </div>

            {/* Custom Path Input */}
            {(useCustomImagePath || !selectedImageModelId) && (
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Custom Model Path</label>
                  <input
                    type="text"
                    value={imageModelPath}
                    onChange={(e) => setImageModelPath(e.target.value)}
                    className="w-full px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600 text-sm font-mono"
                    placeholder="/path/to/model.mlpackage"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">Backend</label>
                  <select
                    value={imageBackend}
                    onChange={(e) => setImageBackend(e.target.value as 'coreml' | 'onnx' | 'tensorrt')}
                    className="w-full px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600"
                  >
                    <option value="coreml">CoreML (Apple)</option>
                    <option value="onnx">ONNX (Universal)</option>
                    <option value="tensorrt">TensorRT (NVIDIA)</option>
                  </select>
                </div>
              </div>
            )}

            {/* Vision Task Type Selector - adapts per model */}
            {imageModelPath && (
              <div className="mb-4">
                <label className="block text-sm font-medium mb-2">Vision Task</label>
                <div className="grid grid-cols-2 gap-2">
                  {getAvailableVisionTasks(imageModelPath).map((task) => (
                    <button
                      key={task}
                      onClick={() => setVisionTaskType(task)}
                      className={`px-4 py-2 rounded text-sm ${
                        visionTaskType === task
                          ? 'bg-blue-500 text-white'
                          : 'bg-gray-200 dark:bg-gray-700'
                      }`}
                    >
                      {task === 'classification' && '🏷️ Classification'}
                      {task === 'localization' && '📍 Localization'}
                      {task === 'detection' && '🔍 Detection'}
                      {task === 'segmentation' && '🎨 Segmentation'}
                    </button>
                  ))}
                </div>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
                  {visionTaskType === 'classification' && 'What is in the image? (Top-K predictions)'}
                  {visionTaskType === 'localization' && 'Where is the main object? (Bounding box)'}
                  {visionTaskType === 'detection' && 'What and where? (All objects with boxes)'}
                  {visionTaskType === 'segmentation' && 'Instance segmentation (Masks per object)'}
                </p>
              </div>
            )}

            <div className="mb-4">
              <label className="block text-sm font-medium mb-2">Upload Image</label>
              <input
                ref={imageInputRef}
                type="file"
                accept="image/*"
                onChange={handleImageSelect}
                className="w-full px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600"
              />
            </div>

            {imagePreview && (
              <div className="mb-4">
                <img src={imagePreview} alt="Preview" className="max-w-full h-auto max-h-96 rounded border" />
              </div>
            )}

            <button
              onClick={handleImageDetection}
              disabled={!imageFile || !imageModelPath || isProcessingImage}
              className="w-full px-4 py-3 bg-green-500 text-white rounded hover:bg-green-600 disabled:bg-gray-400 disabled:cursor-not-allowed"
            >
              {isProcessingImage
                ? 'Processing...'
                : visionTaskType === 'classification'
                  ? 'Classify Image'
                  : visionTaskType === 'localization'
                    ? 'Localize Object'
                    : visionTaskType === 'detection'
                      ? 'Detect Objects'
                      : 'Segment Instances'
              }
            </button>

            {/* Annotated Image Display */}
            {annotatedImage && (
              <div className="mt-4">
                <h3 className="text-sm font-medium mb-2">Annotated Result:</h3>
                <img
                  src={`data:image/png;base64,${annotatedImage}`}
                  alt="Annotated"
                  className="max-w-full h-auto max-h-96 rounded border"
                />
              </div>
            )}

            {/* Results Display */}
            {imageResults && (
              <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-700/50 rounded">
                <h3 className="font-medium mb-2">
                  {imageResults.error ? 'Error:' :
                   imageResults.predictions ? `${visionTaskType === 'classification' ? 'Classification' : 'Localization'} Results:` :
                   imageResults.detections ? `${visionTaskType === 'detection' ? 'Detection' : 'Localization'} Results:` :
                   imageResults.segments ? 'Segmentation Results:' :
                   'Results:'}
                </h3>

                {imageResults.error ? (
                  <p className="text-red-500">{imageResults.error}</p>
                ) : imageResults.predictions ? (
                  <>
                    <p className="text-sm mb-2">Top {imageResults.predictions?.length || 0} predictions</p>
                    <div className="space-y-2 max-h-64 overflow-y-auto">
                      {imageResults.predictions?.map((pred: any, idx: number) => (
                        <div key={idx} className="p-2 bg-white dark:bg-gray-800 rounded text-sm">
                          <div className="font-medium">{pred.label}</div>
                          <div className="text-gray-600 dark:text-gray-400">
                            Confidence: {(pred.confidence * 100).toFixed(1)}%
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                ) : imageResults.detections ? (
                  <>
                    <p className="text-sm mb-2">Found {imageResults.detections?.length || 0} object{imageResults.detections?.length !== 1 ? 's' : ''}</p>
                    <div className="space-y-2 max-h-64 overflow-y-auto">
                      {imageResults.detections?.map((det: any, idx: number) => (
                        <div key={idx} className="p-2 bg-white dark:bg-gray-800 rounded text-sm">
                          <div className="font-medium">{det.label}</div>
                          <div className="text-gray-600 dark:text-gray-400">
                            Confidence: {(det.confidence * 100).toFixed(1)}%
                          </div>
                          {det.box && (
                            <div className="text-xs text-gray-500">
                              Box: [{det.box.map((v: number) => v.toFixed(2)).join(', ')}]
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </>
                ) : imageResults.segments ? (
                  <>
                    <p className="text-sm mb-2">Segmented {imageResults.segments?.length || 0} instance{imageResults.segments?.length !== 1 ? 's' : ''}</p>
                    <div className="space-y-2 max-h-64 overflow-y-auto">
                      {imageResults.segments?.map((seg: any, idx: number) => (
                        <div key={idx} className="p-2 bg-white dark:bg-gray-800 rounded text-sm">
                          <div className="font-medium">Instance {idx + 1}: {seg.label}</div>
                          <div className="text-gray-600 dark:text-gray-400">
                            Confidence: {(seg.confidence * 100).toFixed(1)}%
                          </div>
                          {seg.box && (
                            <div className="text-xs text-gray-500">
                              Box: [{seg.box.map((v: number) => v.toFixed(2)).join(', ')}]
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </>
                ) : null}
              </div>
            )}
          </>
        )}

        {/* Voice Mode */}
        {mode === 'voice' && (
          <>
            {/* Speech Model Selection */}
            <div className="mb-4">
              <label className="block text-sm font-medium mb-2">
                Select Speech Model
                {modelConfigs.filter(m => m.type === 'speech').length === 0 && (
                  <span className="ml-2 text-xs text-gray-500">(Configure speech models in Settings tab)</span>
                )}
              </label>
              <select
                value={selectedVoiceModelId || 'custom'}
                onChange={(e) => handleVoiceModelSelect(e.target.value)}
                className="w-full px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600"
              >
                <option value="">-- Select a speech model --</option>
                {modelConfigs.filter(m => m.type === 'speech').map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name} ({model.backend.toUpperCase()})
                  </option>
                ))}
                <option value="custom">Custom Path...</option>
              </select>
            </div>

            {/* Custom Path Input */}
            {(useCustomVoicePath || !selectedVoiceModelId) && (
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Custom Model Path</label>
                  <input
                    type="text"
                    value={voiceModelPath}
                    onChange={(e) => setVoiceModelPath(e.target.value)}
                    className="w-full px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600 text-sm font-mono"
                    placeholder="/path/to/encoder.mlpackage"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">Backend</label>
                  <select
                    value={voiceBackend}
                    onChange={(e) => setVoiceBackend(e.target.value as 'coreml' | 'onnx')}
                    className="w-full px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600"
                  >
                    <option value="coreml">CoreML (Apple)</option>
                    <option value="onnx">ONNX (Universal)</option>
                  </select>
                </div>
              </div>
            )}

            <div className="mb-4">
              <label className="block text-sm font-medium mb-2">Upload Audio File or Record</label>
              <div className="flex gap-2">
                <input
                  ref={audioInputRef}
                  type="file"
                  accept="audio/*"
                  onChange={handleAudioSelect}
                  className="flex-1 px-3 py-2 border rounded dark:bg-gray-700 dark:border-gray-600"
                />
                {!isRecording ? (
                  <button
                    onClick={startRecording}
                    className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600 whitespace-nowrap flex items-center gap-2"
                  >
                    <span>🎤</span> Record
                  </button>
                ) : (
                  <button
                    onClick={stopRecording}
                    className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 whitespace-nowrap animate-pulse flex items-center gap-2"
                  >
                    <span>⏹</span> Stop
                  </button>
                )}
              </div>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Upload audio file or click Record to capture from microphone
              </p>
            </div>

            {isRecording && (
              <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 rounded border-2 border-red-500 animate-pulse">
                <p className="text-sm font-medium text-red-600 dark:text-red-400">
                  🎤 Recording in progress... Click Stop when finished
                </p>
              </div>
            )}

            {audioFile && !isRecording && (
              <div className="mb-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                <p className="text-sm">
                  <strong>Selected:</strong> {audioFile.name} ({(audioFile.size / 1024).toFixed(1)} KB)
                </p>
              </div>
            )}

            <button
              onClick={handleTranscribe}
              disabled={!audioFile || !voiceModelPath || isProcessingAudio}
              className="w-full px-4 py-3 bg-purple-500 text-white rounded hover:bg-purple-600 disabled:bg-gray-400 disabled:cursor-not-allowed"
            >
              {isProcessingAudio ? 'Transcribing...' : 'Transcribe Audio'}
            </button>

            {transcription && (
              <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-700/50 rounded">
                <h3 className="font-medium mb-2">Transcription:</h3>
                <p className="text-sm whitespace-pre-wrap">{transcription}</p>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
