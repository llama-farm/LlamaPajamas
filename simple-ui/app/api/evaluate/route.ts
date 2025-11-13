import { NextRequest } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'

export async function POST(request: NextRequest) {
  const body = await request.json()
  const { modelPath, format, evalType, numQuestions } = body

  const encoder = new TextEncoder()
  const stream = new ReadableStream({
    start(controller) {
      const cliPath = path.join(process.cwd(), '../quant')
      let args: string[] = []

      const send = (data: any) => {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`))
      }

      // Build args based on evaluation type
      if (evalType === 'llm') {
        args = [
          'run',
          'llama-pajamas-quant',
          'evaluate',
          'llm',
          '--model-dir', modelPath,
          '--num-questions', numQuestions.toString(),
        ]
      } else if (evalType === 'vision') {
        // Vision evaluation - use run-coreml with UV
        const runCoreMLDir = path.join(process.cwd(), '../run-coreml')
        const evalScript = path.join(process.cwd(), '../quant/evaluation/vision/run_eval.py')
        const imagesDir = path.join(process.cwd(), '../quant/evaluation/vision/images')
        const modelsDir = path.dirname(path.dirname(modelPath))
        const modelName = path.basename(modelPath)

        send({ progress: `Starting vision evaluation of ${modelName}...\n` })

        const proc = spawn('uv', ['run', 'python', evalScript, '--model', modelName, '--models-dir', modelsDir, '--images', imagesDir], {
          cwd: runCoreMLDir,
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
            // Parse vision results (FPS, latency, etc.)
            const fpsMatch = output.match(/Avg FPS:\s+([\d.]+)/)
            const latencyMatch = output.match(/Avg Latency:\s+([\d.]+)ms/)
            const accuracyMatch = output.match(/Accuracy:\s+([\d.]+)%/)

            if (fpsMatch && latencyMatch) {
              send({
                results: {
                  fps: parseFloat(fpsMatch[1]),
                  latency: parseFloat(latencyMatch[1]),
                  avgTime: parseFloat(latencyMatch[1]) / 1000,
                  accuracy: accuracyMatch ? parseFloat(accuracyMatch[1]) : 100,
                },
              })
            }
          } else {
            send({ error: `Evaluation failed with code ${code}` })
          }
          controller.close()
        })

        proc.on('error', (error) => {
          send({ error: error.message })
          controller.close()
        })

        return
      } else if (evalType === 'speech') {
        // Speech evaluation - use run-coreml evaluation script
        const runCoreMLDir = path.join(process.cwd(), '../run-coreml')
        const evalScript = path.join(process.cwd(), '../quant/evaluation/stt/run_eval.py')
        const modelsDir = path.dirname(path.dirname(modelPath))
        const modelName = path.basename(path.dirname(path.dirname(modelPath)))

        send({ progress: `Starting speech evaluation of ${modelName}...\n` })

        const proc = spawn('uv', ['run', 'python', evalScript, '--models-dir', modelsDir, '--model', modelName], {
          cwd: runCoreMLDir,
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
            // Parse WER and latency from output
            const werMatch = output.match(/Avg WER:\s+([\d.]+)/)
            const latencyMatch = output.match(/Avg Latency:\s+([\d.]+)ms/)
            const rtfMatch = output.match(/RTF:\s+([\d.]+)/)

            if (werMatch && latencyMatch) {
              const wer = parseFloat(werMatch[1])
              const avgLatency = parseFloat(latencyMatch[1])
              const rtf = rtfMatch ? parseFloat(rtfMatch[1]) : 0

              send({
                results: {
                  wer: wer,
                  accuracy: (1 - wer) * 100, // Convert WER to accuracy percentage
                  avgTime: avgLatency / 1000, // Convert to seconds
                  rtf: rtf,
                },
              })
            }
          } else {
            send({ error: `Evaluation failed with code ${code}` })
          }
          controller.close()
        })

        proc.on('error', (error) => {
          send({ error: error.message })
          controller.close()
        })

        return
      }

      send({ progress: `Starting ${evalType} evaluation of ${path.basename(modelPath)}...\n` })

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
          // Parse final results from output
          if (evalType === 'llm') {
            const resultsMatch = output.match(/(\d+)\/(\d+) correct \(([\d.]+)%\)/)
            const timeMatch = output.match(/Average time: ([\d.]+)s/)

            if (resultsMatch && timeMatch) {
              const correct = parseInt(resultsMatch[1])
              const total = parseInt(resultsMatch[2])
              const accuracy = parseFloat(resultsMatch[3])
              const avgTime = parseFloat(timeMatch[1])

              // Parse category breakdown
              const categories: { [key: string]: number } = {}
              const categoryRegex = /(\w+)\s+([\d.]+)%/g
              let match
              while ((match = categoryRegex.exec(output)) !== null) {
                categories[match[1]] = parseFloat(match[2])
              }

              send({
                results: {
                  correct,
                  total,
                  accuracy,
                  avgTime,
                  categories: Object.keys(categories).length > 0 ? categories : undefined,
                },
              })
            }
          } else if (evalType === 'vision') {
            // Parse vision results (FPS, latency, etc.)
            const fpsMatch = output.match(/FPS:\s+([\d.]+)/)
            const latencyMatch = output.match(/Latency:\s+([\d.]+)ms/)

            if (fpsMatch && latencyMatch) {
              send({
                results: {
                  fps: parseFloat(fpsMatch[1]),
                  latency: parseFloat(latencyMatch[1]),
                  avgTime: parseFloat(latencyMatch[1]) / 1000, // Convert to seconds
                  accuracy: 100, // Vision doesn't have accuracy, use 100%
                },
              })
            }
          }
        } else {
          send({ error: `Evaluation failed with code ${code}` })
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
      Connection: 'keep-alive',
    },
  })
}
