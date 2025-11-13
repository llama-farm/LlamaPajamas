import { NextRequest } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'

export async function POST(request: NextRequest) {
  const { model, backend, precision, modelType, outputDir } = await request.json()

  const cliPath = path.join(process.cwd(), '../quant')

  const stream = new ReadableStream({
    start(controller) {
      const send = (data: any) => {
        controller.enqueue(`data: ${JSON.stringify(data)}\n\n`)
      }

      const args = [
        'run',
        'llama-pajamas-quant',
        'export',
        '--model', model,
        '--backend', backend,
        '--precision', precision,
        '--output', outputDir,
      ]

      if (modelType !== 'auto') {
        args.push('--model-type', modelType)
      }

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
              path: outputDir,
              backend,
              precision,
              size: 'See output for details',
            },
          })
        } else {
          send({ error: `Export failed with code ${code}` })
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
