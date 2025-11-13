import { NextRequest, NextResponse } from 'next/server'
import { readdir } from 'fs/promises'
import path from 'path'
import { stat } from 'fs/promises'

export async function POST(request: NextRequest) {
  try {
    const { modelsDir } = await request.json()
    const baseDir = modelsDir || path.join(process.cwd(), '../quant/models')

    // Find F16 GGUF files (created during quantization)
    const f16Models: string[] = []

    // Find calibration text files
    const calibrationFiles: string[] = []

    const scanDir = async (dir: string): Promise<void> => {
      try {
        const entries = await readdir(dir, { withFileTypes: true })

        for (const entry of entries) {
          const fullPath = path.join(dir, entry.name)

          if (entry.isDirectory()) {
            // Recurse into subdirectories
            await scanDir(fullPath)
          } else if (entry.isFile()) {
            // Check for F16 GGUF files
            if (entry.name.includes('f16') && entry.name.endsWith('.gguf')) {
              f16Models.push(fullPath)
            }
            // Check for calibration files
            if (entry.name.endsWith('.txt') &&
                (entry.name.includes('calib') || entry.name.includes('train') || entry.name.includes('dataset'))) {
              calibrationFiles.push(fullPath)
            }
          }
        }
      } catch (error) {
        // Skip directories we can't read
      }
    }

    await scanDir(baseDir)

    // Also check common calibration locations
    const commonCalibPaths = [
      path.join(process.cwd(), '../quant/calibration'),
      path.join(process.cwd(), '../quant/datasets'),
      path.join(process.cwd(), '../datasets'),
    ]

    for (const calibPath of commonCalibPaths) {
      try {
        const entries = await readdir(calibPath, { withFileTypes: true })
        for (const entry of entries) {
          if (entry.isFile() && entry.name.endsWith('.txt')) {
            const fullPath = path.join(calibPath, entry.name)
            if (!calibrationFiles.includes(fullPath)) {
              calibrationFiles.push(fullPath)
            }
          }
        }
      } catch {
        // Directory doesn't exist, skip
      }
    }

    return NextResponse.json({
      f16Models: f16Models.sort(),
      calibrationFiles: calibrationFiles.sort(),
    })
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 })
  }
}
