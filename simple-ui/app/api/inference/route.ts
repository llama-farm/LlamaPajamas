import { NextRequest } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'
import { writeFile, unlink } from 'fs/promises'
import { randomUUID } from 'crypto'

export async function POST(request: NextRequest) {
  const { modelPath, backend, prompt, maxTokens, temperature } = await request.json()

  const runDir = path.join(process.cwd(), '../run')
  const tempScript = path.join(runDir, `inference-${randomUUID()}.py`)

  const stream = new ReadableStream({
    async start(controller) {
      const send = (data: any) => {
        controller.enqueue(`data: ${JSON.stringify(data)}\n\n`)
      }

      try {
        // Create temporary Python script using llama_pajamas_run
        const pythonScript = `
import sys
from llama_pajamas_run import RuntimeConfig, ModelLoader

config = RuntimeConfig(
    backend="${backend}",
    model_path="${modelPath}",
    max_tokens=${maxTokens || 200},
    temperature=${temperature || 0.7},
    ${backend === 'gguf' ? 'n_gpu_layers=-1,' : ''}
)

try:
    with ModelLoader(config) as loader:
        prompt = """${prompt.replace(/"/g, '\\"')}"""

        # Stream generation
        print("STARTING_GENERATION", flush=True)
        for chunk in loader.generate(prompt, stream=True):
            print(f"CHUNK:{chunk}", flush=True, end="")
        print("\\nGENERATION_COMPLETE", flush=True)
except Exception as e:
    print(f"ERROR:{str(e)}", flush=True, file=sys.stderr)
    sys.exit(1)
`

        await writeFile(tempScript, pythonScript, 'utf-8')

        const proc = spawn('uv', ['run', 'python', tempScript], {
          cwd: runDir,
          stdio: 'pipe',
        })

        let output = ''
        let isGenerating = false

        proc.stdout?.on('data', (data) => {
          const text = data.toString()
          output += text

          const lines = text.split('\n')
          for (const line of lines) {
            if (line === 'STARTING_GENERATION') {
              isGenerating = true
              send({ status: 'generating' })
            } else if (line === 'GENERATION_COMPLETE') {
              isGenerating = false
              send({ status: 'complete' })
            } else if (line.startsWith('CHUNK:')) {
              const chunk = line.substring(6)
              send({ chunk })
            } else if (line.trim()) {
              send({ progress: line })
            }
          }
        })

        proc.stderr?.on('data', (data) => {
          const text = data.toString()
          if (text.startsWith('ERROR:')) {
            send({ error: text.substring(6) })
          }
        })

        proc.on('close', async (code) => {
          // Clean up temp file
          try {
            await unlink(tempScript)
          } catch (e) {
            // Ignore errors when deleting temp file
          }

          if (code === 0) {
            send({ status: 'complete', output })
          } else {
            send({ error: `Inference failed with code ${code}` })
          }
          controller.close()
        })

        proc.on('error', async (error) => {
          try {
            await unlink(tempScript)
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
