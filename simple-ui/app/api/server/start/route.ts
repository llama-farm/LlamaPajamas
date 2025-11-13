import { NextRequest, NextResponse } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'

// In-memory store for running servers
const runningServers: { [key: string]: any } = {}

export async function POST(request: NextRequest) {
  try {
    const { modelPath, serverType, port, gpuLayers, contextSize } = await request.json()

    const serverId = `${serverType}-${port}`

    // Get the appropriate runtime directory
    const runDir = path.join(process.cwd(), '../run')

    // Spawn server process based on type
    let proc
    if (serverType === 'gguf') {
      // Use llama-cpp-python server (via llama.cpp)
      proc = spawn('uv', [
        'run',
        'python',
        '-m',
        'llama_cpp.server',
        '--model', modelPath,
        '--host', '0.0.0.0',
        '--port', port.toString(),
        '--n-gpu-layers', gpuLayers.toString(),
        '--n-ctx', contextSize.toString(),
      ], {
        cwd: runDir,
        detached: true,
        stdio: 'ignore',
      })
    } else if (serverType === 'mlx') {
      // Use MLX server
      proc = spawn('uv', [
        'run',
        'python',
        '-m',
        'mlx_lm.server',
        '--model', modelPath,
        '--host', '0.0.0.0',
        '--port', port.toString(),
        '--max-tokens', contextSize.toString(),
      ], {
        cwd: runDir,
        detached: true,
        stdio: 'ignore',
      })
    } else {
      return NextResponse.json({ error: `Server type ${serverType} not yet implemented` }, { status: 400 })
    }

    proc.unref()

    // Store server info
    runningServers[serverId] = {
      running: true,
      port,
      modelPath,
      pid: proc.pid,
      serverType,
    }

    return NextResponse.json({
      success: true,
      serverId,
      port,
      pid: proc.pid,
    })
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 })
  }
}
