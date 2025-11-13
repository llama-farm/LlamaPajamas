import { NextRequest, NextResponse } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'

// In-memory store for running servers (shared)
const runningServers: { [key: string]: any } = {}

export async function POST(request: NextRequest) {
  try {
    const { visionModel, sttModel, port } = await request.json()

    const serverId = `multimodal-${port}`

    // Get the CoreML runtime directory
    const runCoreMLDir = path.join(process.cwd(), '../run-coreml')

    // Build Python script to start multimodal server
    const args = [
      'run',
      'python',
      'examples/multimodal_server_demo.py',
    ]

    // Note: The demo script has hardcoded paths, but we can override via environment
    // or modify the script. For now, we'll use the demo as-is.

    const proc = spawn('uv', args, {
      cwd: runCoreMLDir,
      detached: true,
      stdio: 'ignore',
      env: {
        ...process.env,
        VISION_MODEL: visionModel || '',
        STT_MODEL: sttModel || '',
        SERVER_PORT: port.toString(),
      },
    })

    proc.unref()

    // Store server info
    runningServers[serverId] = {
      running: true,
      port,
      visionModel,
      sttModel,
      pid: proc.pid,
      serverType: 'multimodal',
    }

    return NextResponse.json({
      success: true,
      serverId,
      port,
      pid: proc.pid,
      message: 'Multimodal server started with Vision + STT endpoints',
    })
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 })
  }
}
