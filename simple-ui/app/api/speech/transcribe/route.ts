import { NextRequest } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'
import { writeFile, unlink } from 'fs/promises'
import { randomUUID } from 'crypto'

export async function POST(request: NextRequest) {
  const formData = await request.formData()
  const modelPath = formData.get('modelPath') as string
  const backend = formData.get('backend') as string || 'coreml'
  const audioFile = formData.get('audio') as File

  if (!modelPath || !audioFile) {
    return new Response(
      JSON.stringify({ error: 'Missing modelPath or audio' }),
      { status: 400, headers: { 'Content-Type': 'application/json' } }
    )
  }

  // Determine which runtime directory to use based on backend
  const runtimeMap: Record<string, string> = {
    'onnx': 'run-onnx',
    'coreml': 'run-coreml',
  }

  const runtimeDir = path.join(process.cwd(), '..', runtimeMap[backend] || 'run-coreml')
  const tempDir = path.join(process.cwd(), '..', 'run')

  // Save as .wav extension regardless of input format - librosa/ffmpeg will handle conversion
  const tempAudioPath = path.join(tempDir, `audio-${randomUUID()}.webm`)
  const tempWavPath = path.join(tempDir, `audio-${randomUUID()}.wav`)
  const tempScript = path.join(tempDir, `speech-${randomUUID()}.py`)

  const stream = new ReadableStream({
    async start(controller) {
      const send = (data: any) => {
        controller.enqueue(`data: ${JSON.stringify(data)}\n\n`)
      }

      try {
        // Save uploaded audio to temp file
        const audioBuffer = Buffer.from(await audioFile.arrayBuffer())
        await writeFile(tempAudioPath, audioBuffer)

        send({ progress: 'Audio saved, converting format...' })

        // Convert to WAV using ffmpeg (librosa needs this for webm)
        const { spawn: spawnConvert } = require('child_process')
        await new Promise<void>((resolve, reject) => {
          const ffmpeg = spawnConvert('ffmpeg', [
            '-i', tempAudioPath,
            '-ar', '16000',  // 16kHz for Whisper
            '-ac', '1',      // Mono
            '-f', 'wav',     // WAV format
            '-y',            // Overwrite
            tempWavPath
          ])

          ffmpeg.on('close', (code: number) => {
            if (code === 0) resolve()
            else reject(new Error(`ffmpeg failed with code ${code}`))
          })

          ffmpeg.on('error', reject)
        })

        send({ progress: 'Audio converted, loading model...' })

        // Create Python script for speech inference using EXACT README pattern
        const pythonScript = backend === 'onnx' ? `
import sys
import json

from llama_pajamas_run_onnx.backends.speech import ONNXSpeechBackend
from llama_pajamas_run_core.utils.audio_utils import load_audio

print("Loading ONNX speech model...", flush=True)

try:
    backend = ONNXSpeechBackend()

    # Load model (README lines 929-940)
    backend.load_model(
        encoder_path="${modelPath.replace(/\\/g, '/')}",
        model_name="whisper-tiny",
        providers=["CPUExecutionProvider"],
    )

    print("Model loaded, transcribing audio...", flush=True)

    # Load audio using proper utility (README line 803)
    audio = load_audio("${tempWavPath.replace(/\\/g, '/')}", sample_rate=16000)

    # Transcribe (README line 949)
    result = backend.transcribe(audio, sample_rate=16000)

    print(f"RESULT:{json.dumps({'text': result.text, 'language': getattr(result, 'language', 'en'), 'confidence': getattr(result, 'confidence', 1.0)})}", flush=True)

except Exception as e:
    print(f"ERROR:{str(e)}", flush=True, file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
` : `
import sys
import json

from llama_pajamas_run_coreml.backends.stt import CoreMLSTTBackend
from llama_pajamas_run_core.utils.audio_utils import load_audio

print("Loading CoreML speech model...", flush=True)

try:
    # Use proper CoreMLSTTBackend (README lines 788-819)
    backend = CoreMLSTTBackend()

    # Extract model name from path (e.g., "tiny" from "whisper-tiny")
    # Path format: .../whisper-tiny/coreml/int8/encoder.mlpackage
    # So [-4] is "whisper-tiny", then remove "whisper-" prefix
    model_name = "${modelPath.replace(/\\/g, '/')}".split('/')[-4].replace('whisper-', '')

    backend.load_model(
        model_path="${modelPath.replace(/\\/g, '/')}",
        model_name=model_name
    )

    print("Model loaded, transcribing audio...", flush=True)

    # Load audio using proper utility (README line 803)
    audio = load_audio("${tempWavPath.replace(/\\/g, '/')}", sample_rate=16000)

    # Transcribe (README line 814)
    result = backend.transcribe(audio, sample_rate=16000)

    # Calculate average confidence from segments if available
    confidence = 1.0
    if result.segments and len(result.segments) > 0:
        confidence = sum(seg.confidence for seg in result.segments) / len(result.segments)

    print(f"RESULT:{json.dumps({'text': result.text, 'language': result.language, 'confidence': confidence})}", flush=True)

except Exception as e:
    print(f"ERROR:{str(e)}", flush=True, file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
`

        await writeFile(tempScript, pythonScript, 'utf-8')

        const proc = spawn('uv', ['run', 'python', tempScript], {
          cwd: runtimeDir,
          stdio: 'pipe',
        })

        let output = ''

        proc.stdout?.on('data', (data) => {
          const text = data.toString()
          output += text

          const lines = text.split('\n')
          for (const line of lines) {
            if (line.startsWith('RESULT:')) {
              const resultData = JSON.parse(line.substring(7))
              send({ result: resultData })
            } else if (line.trim() && !line.startsWith('ERROR:')) {
              send({ progress: line })
            }
          }
        })

        proc.stderr?.on('data', (data) => {
          const text = data.toString()
          if (text.startsWith('ERROR:')) {
            send({ error: text.substring(6) })
          } else {
            // Send stderr as progress
            send({ progress: text })
          }
        })

        proc.on('close', async (code) => {
          // Clean up temp files
          try {
            await unlink(tempScript)
            await unlink(tempAudioPath)
            await unlink(tempWavPath)
          } catch (e) {
            // Ignore cleanup errors
          }

          if (code === 0) {
            send({ status: 'complete' })
          } else {
            send({ error: `Speech transcription failed with code ${code}` })
          }
          controller.close()
        })

        proc.on('error', async (error) => {
          try {
            await unlink(tempScript)
            await unlink(tempAudioPath)
            await unlink(tempWavPath)
          } catch (e) {
            // Ignore
          }
          send({ error: error.message })
          controller.close()
        })
      } catch (error: any) {
        send({ error: error.message })
        controller.close()
      }
    },
  })

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    },
  })
}
