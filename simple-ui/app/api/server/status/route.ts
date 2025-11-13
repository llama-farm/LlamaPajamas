import { NextResponse } from 'next/server'
import { exec } from 'child_process'
import { promisify } from 'util'

const execAsync = promisify(exec)

// In-memory store for running servers
const runningServers: { [key: string]: any } = {}

export async function GET() {
  // Check which servers are actually still running
  for (const [id, server] of Object.entries(runningServers)) {
    try {
      const { stdout } = await execAsync(`ps -p ${server.pid}`)
      if (!stdout.includes(server.pid.toString())) {
        delete runningServers[id]
      }
    } catch {
      delete runningServers[id]
    }
  }

  return NextResponse.json({ servers: runningServers })
}
