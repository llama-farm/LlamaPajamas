import { NextRequest } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'

export async function POST(request: NextRequest) {
  const body = await request.json()
  const {
    modelType,
    model,
    formats,
    ggufPrecision,
    mlxBits,
    outputDir,
    enableIQ,
    iqPrecision,
  } = body

  const encoder = new TextEncoder()
  const stream = new ReadableStream({
    start(controller) {
      // Build CLI command
      const cliPath = path.join(process.cwd(), '../quant')
      const args = ['run', 'llama-pajamas-quant', 'quantize', modelType]

      args.push('--model', model)
      args.push('--output', outputDir)

      if (modelType === 'llm') {
        args.push('--formats', formats.join(','))
        if (formats.includes('gguf')) {
          args.push('--gguf-precision', ggufPrecision)
        }
        if (formats.includes('mlx')) {
          args.push('--mlx-bits', mlxBits.toString())
        }
      }

      const send = (data: any) => {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`))
      }

      send({ progress: `Starting quantization for ${model}...` })

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

      proc.on('close', async (code) => {
        if (code === 0) {
          send({
            progress: 'Quantization complete!',
            result: {
              path: outputDir,
              format: formats.join(', '),
              size: 'See output directory',
            },
          })

          // Run IQ quantization if enabled
          if (enableIQ && modelType === 'llm') {
            send({ progress: '\nStarting IQ quantization...' })

            // Find the F16 model
            const f16Path = `${outputDir}/gguf/F16/hf_model_f16.gguf`

            // Generate calibration
            const calibPath = `${outputDir}/calibration.txt`
            const imatrixPath = `${outputDir}/model.imatrix`

            const iqArgs = [
              'run',
              'llama-pajamas-quant',
              'iq',
              'quantize',
              '--model', f16Path,
              '--calibration', calibPath,
              '--precision', iqPrecision,
              '--output', `${outputDir}/gguf/${iqPrecision}/`,
            ]

            const iqProc = spawn('uv', iqArgs, { cwd: cliPath })

            iqProc.stdout?.on('data', (data) => {
              send({ progress: data.toString() })
            })

            iqProc.on('close', (iqCode) => {
              if (iqCode === 0) {
                send({ progress: 'IQ quantization complete!' })
              }
              controller.close()
            })
          } else {
            controller.close()
          }
        } else {
          send({ progress: `Error: Process exited with code ${code}` })
          controller.close()
        }
      })
    },
  })

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
    },
  })
}
