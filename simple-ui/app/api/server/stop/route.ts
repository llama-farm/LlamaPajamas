import { NextRequest, NextResponse } from 'next/server'
import { exec } from 'child_process'
import { promisify } from 'util'

const execAsync = promisify(exec)

// In-memory store for running servers (shared with other routes)
const runningServers: { [key: string]: any } = {}

export async function POST(request: NextRequest) {
  const body = await request.json()
  const { serverId } = body

  const server = runningServers[serverId]
  if (!server) {
    return NextResponse.json({
      success: false,
      error: 'Server not found',
    })
  }

  try {
    await execAsync(`kill ${server.pid}`)
    delete runningServers[serverId]

    return NextResponse.json({
      success: true,
    })
  } catch (error) {
    return NextResponse.json({
      success: false,
      error: String(error),
    })
  }
}
