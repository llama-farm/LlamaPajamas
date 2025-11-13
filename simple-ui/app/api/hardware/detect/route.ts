import { NextResponse } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'

export async function GET() {
  try {
    const cliPath = path.join(process.cwd(), '../quant')

    // Run hardware detection CLI
    const proc = spawn('uv', ['run', 'llama-pajamas-quant', 'hardware', 'detect'], {
      cwd: cliPath,
      stdio: 'pipe',
    })

    let output = ''

    return new Promise<Response>((resolve) => {
      proc.stdout?.on('data', (data) => {
        output += data.toString()
      })

      proc.on('close', (code) => {
        if (code === 0) {
          // Parse output to extract hardware info
          const lines = output.split('\n')

          let platform = 'unknown'
          let cpu = 'unknown'
          let ram_gb = 16 // default
          let gpu = undefined
          let recommended_backend = 'gguf'

          for (const line of lines) {
            if (line.includes('Platform:')) {
              platform = line.split('Platform:')[1]?.trim() || 'unknown'
            }
            if (line.includes('CPU:')) {
              cpu = line.split('CPU:')[1]?.trim() || 'unknown'
            }
            if (line.includes('Memory:')) {
              const match = line.match(/([\d.]+)\s*GB/)
              if (match) {
                ram_gb = parseFloat(match[1])
              }
            }
            if (line.includes('GPU:')) {
              gpu = line.split('GPU:')[1]?.trim()
            }
            if (line.includes('Recommended backend:')) {
              recommended_backend = line.split('Recommended backend:')[1]?.trim() || 'gguf'
            }
          }

          resolve(
            NextResponse.json({
              hardware: {
                platform,
                cpu,
                ram_gb,
                gpu,
                recommended_backend,
              },
            })
          )
        } else {
          // Fallback to basic detection
          resolve(
            NextResponse.json({
              hardware: {
                platform: process.platform,
                cpu: 'Unknown',
                ram_gb: 16,
                recommended_backend: 'gguf',
              },
            })
          )
        }
      })
    })
  } catch (error) {
    return NextResponse.json({
      hardware: {
        platform: process.platform,
        cpu: 'Unknown',
        ram_gb: 16,
        recommended_backend: 'gguf',
      },
    })
  }
}
