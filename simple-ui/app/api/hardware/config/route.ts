import { NextRequest, NextResponse } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'

export async function POST(request: NextRequest) {
  try {
    const { serverType, gpuLayers, contextSize, port } = await request.json()

    const cliPath = path.join(process.cwd(), '../quant')

    return new Promise<Response>((resolve) => {
      const proc = spawn(
        'uv',
        ['run', 'llama-pajamas-quant', 'hardware', 'config'],
        {
          cwd: cliPath,
          stdio: 'pipe',
        }
      )

      let output = ''

      proc.stdout?.on('data', (data) => {
        output += data.toString()
      })

      proc.on('close', (code) => {
        if (code === 0) {
          // Generate YAML config based on current settings
          const config = `# LlamaPajamas Runtime Configuration
# Generated based on detected hardware

runtime:
  backend: ${serverType}
  port: ${port}

  # Model settings
  gpu_layers: ${gpuLayers}
  context_size: ${contextSize}

  # Performance settings
  num_threads: ${serverType === 'gguf' ? '8' : 'auto'}
  batch_size: ${contextSize >= 8192 ? '512' : '256'}

  # Memory settings
  use_mlock: ${gpuLayers > 0 ? 'true' : 'false'}
  use_mmap: true

# Recommended models for this configuration
models:
  - path: ./models/qwen3-8b/${serverType === 'mlx' ? 'mlx/4bit-mixed' : 'gguf/Q4_K_M/model.gguf'}
    format: ${serverType}
    ${gpuLayers > 0 ? 'gpu_enabled: true' : '# GPU disabled'}

# Hardware info from CLI output:
${output.split('\n').map(line => `# ${line}`).join('\n')}
`

          resolve(
            NextResponse.json({
              config,
              success: true,
            })
          )
        } else {
          // Fallback config if CLI fails
          const fallbackConfig = `# LlamaPajamas Runtime Configuration
# Fallback configuration (hardware detection failed)

runtime:
  backend: ${serverType}
  port: ${port}
  gpu_layers: ${gpuLayers}
  context_size: ${contextSize}
`

          resolve(
            NextResponse.json({
              config: fallbackConfig,
              success: true,
              warning: 'Hardware detection failed, using fallback config',
            })
          )
        }
      })

      proc.on('error', (error) => {
        // Error fallback
        const fallbackConfig = `# LlamaPajamas Runtime Configuration
# Error generating config: ${error.message}

runtime:
  backend: ${serverType}
  port: ${port}
  gpu_layers: ${gpuLayers}
  context_size: ${contextSize}
`

        resolve(
          NextResponse.json({
            config: fallbackConfig,
            success: true,
            warning: 'Error in hardware detection',
          })
        )
      })
    })
  } catch (error: any) {
    return NextResponse.json({
      error: error.message,
      success: false,
    }, { status: 500 })
  }
}
