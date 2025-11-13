import { NextRequest } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'

// Simplified IQ quantization - just call the CLI!
// CLI handles imatrix generation + quantization in one command
export async function POST(request: NextRequest) {
  const { sourceModel, calibrationFile, precision, outputDir } = await request.json()

  const cliPath = path.join(process.cwd(), '../quant')

  const stream = new ReadableStream({
    start(controller) {
      const send = (data: any) => {
        controller.enqueue(`data: ${JSON.stringify(data)}\n\n`)
      }

      // Single command - CLI handles everything!
      const args = [
        'run',
        'llama-pajamas-quant',
        'iq',
        'quantize',
        '--model', sourceModel,
        '--calibration', calibrationFile,
        '--precision', precision,
        '--output', outputDir || path.join(path.dirname(sourceModel), precision),
      ]

      send({ progress: `🚀 Starting IQ quantization to ${precision}...\n` })
      send({ progress: `📦 Source: ${path.basename(sourceModel)}\n` })
      send({ progress: `📊 Calibration: ${path.basename(calibrationFile)}\n\n` })

      const proc = spawn('uv', args, {
        cwd: cliPath,
        stdio: 'pipe',
      })

      let output = ''

      proc.stdout?.on('data', (data) => {
        const text = data.toString()
        output += text
        send({ progress: text })
      })

      proc.stderr?.on('data', (data) => {
        const text = data.toString()
        output += text
        send({ progress: text })
      })

      proc.on('close', (code) => {
        if (code === 0) {
          send({
            result: {
              path: outputDir || path.join(path.dirname(sourceModel), precision),
              precision,
            },
          })
        } else {
          send({ error: `IQ quantization failed with code ${code}` })
        }
        controller.close()
      })

      proc.on('error', (error) => {
        send({ error: error.message })
        controller.close()
      })
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
