import { NextRequest } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'
import { writeFile, unlink } from 'fs/promises'
import { randomUUID } from 'crypto'

export async function POST(request: NextRequest) {
  const { config, parallel, dryRun } = await request.json()

  const cliPath = path.join(process.cwd(), '../quant')
  const tempConfigFile = path.join(cliPath, `batch-config-${randomUUID()}.yaml`)

  const stream = new ReadableStream({
    async start(controller) {
      const send = (data: any) => {
        controller.enqueue(`data: ${JSON.stringify(data)}\n\n`)
      }

      try {
        // Write config to temporary file
        await writeFile(tempConfigFile, config, 'utf-8')

        const args = [
          'run',
          'llama-pajamas-quant',
          'batch',
          '--config', tempConfigFile,
          '--parallel', parallel.toString(),
        ]

        if (dryRun) {
          args.push('--dry-run')
        }

        const proc = spawn('uv', args, {
          cwd: cliPath,
          stdio: 'pipe',
        })

        let output = ''
        const startTime = Date.now()

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
          // Clean up temp file
          try {
            await unlink(tempConfigFile)
          } catch (e) {
            // Ignore errors when deleting temp file
          }

          if (code === 0) {
            const duration = ((Date.now() - startTime) / 1000).toFixed(1)

            // Parse output to extract stats
            let total = 0
            let successful = 0
            let failed = 0

            const lines = output.split('\n')
            for (const line of lines) {
              if (line.includes('Processing model')) total++
              if (line.includes('✓') || line.includes('Success')) successful++
              if (line.includes('✗') || line.includes('Failed')) failed++
            }

            send({
              result: {
                total,
                successful,
                failed,
                duration: `${duration}s`,
              },
            })
          } else {
            send({ error: `Batch processing failed with code ${code}` })
          }
          controller.close()
        })

        proc.on('error', async (error) => {
          try {
            await unlink(tempConfigFile)
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
