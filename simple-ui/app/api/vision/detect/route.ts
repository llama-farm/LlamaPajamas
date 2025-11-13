import { NextRequest } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'
import { writeFile, unlink } from 'fs/promises'
import { randomUUID } from 'crypto'

export async function POST(request: NextRequest) {
  const formData = await request.formData()
  const modelPath = formData.get('modelPath') as string
  const backend = formData.get('backend') as string || 'onnx'
  const imageFile = formData.get('image') as File

  if (!modelPath || !imageFile) {
    return new Response(
      JSON.stringify({ error: 'Missing modelPath or image' }),
      { status: 400, headers: { 'Content-Type': 'application/json' } }
    )
  }

  // Determine which runtime directory to use based on backend
  const runtimeMap: Record<string, string> = {
    'onnx': 'run-onnx',
    'coreml': 'run-coreml',
    'tensorrt': 'run-tensorrt',
  }

  const runtimeDir = path.join(process.cwd(), '..', runtimeMap[backend] || 'run-onnx')
  const tempDir = path.join(process.cwd(), '..', 'run')
  const tempImagePath = path.join(tempDir, `image-${randomUUID()}.png`)
  const tempScript = path.join(tempDir, `vision-${randomUUID()}.py`)

  const stream = new ReadableStream({
    async start(controller) {
      const send = (data: any) => {
        controller.enqueue(`data: ${JSON.stringify(data)}\n\n`)
      }

      try {
        // Save uploaded image to temp file
        const imageBuffer = Buffer.from(await imageFile.arrayBuffer())
        await writeFile(tempImagePath, imageBuffer)

        send({ progress: 'Image saved, loading model...' })

        // Detect model type from path/name
        const modelName = modelPath.toLowerCase()
        const isDetection = modelName.includes('yolo')
        const isClassification = modelName.includes('vit') || modelName.includes('clip')
        const modelType = isDetection ? 'detection' : isClassification ? 'classification' : 'detection'

        // Create Python script for vision inference using EXACT README pattern
        const pythonScript = backend === 'onnx' ? `
import sys
import json
from PIL import Image

from llama_pajamas_run_onnx.backends.vision import ONNXVisionBackend

print("Loading ONNX vision model...", flush=True)

try:
    backend = ONNXVisionBackend()
    backend.load_model(
        "${modelPath.replace(/\\/g, '/')}",
        model_type="${modelType}",
        providers=["CPUExecutionProvider"],
        num_threads=4
    )

    print("Model loaded, running inference...", flush=True)

    # Load image
    image = Image.open("${tempImagePath.replace(/\\/g, '/')}")

    # Run appropriate method based on model type
    if "${modelType}" == "detection":
        # Object detection (README line 643-645)
        detections = backend.detect(
            image,
            confidence_threshold=0.5,
            iou_threshold=0.45,
            max_detections=100
        )
        # Convert DetectionResult objects to dicts
        detections_dict = [{'label': d.class_name, 'confidence': d.confidence, 'box': [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2]} for d in detections]
        print(f"RESULT:{json.dumps({'detections': detections_dict})}", flush=True)
    else:
        # Classification (README line 578-582)
        predictions = backend.classify(image, top_k=5)
        # Convert ClassificationResult objects to dicts
        predictions_dict = [{'label': p.class_name, 'confidence': p.confidence} for p in predictions]
        print(f"RESULT:{json.dumps({'predictions': predictions_dict})}", flush=True)

except Exception as e:
    print(f"ERROR:{str(e)}", flush=True, file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
` : `
import sys
import json
from PIL import Image

from llama_pajamas_run_coreml.backends.vision import CoreMLVisionBackend

print("Loading CoreML vision model...", flush=True)

try:
    # Use proper CoreMLVisionBackend (README lines 519-563)
    backend = CoreMLVisionBackend()
    backend.load_model(
        model_path="${modelPath.replace(/\\/g, '/')}",
        model_type="${modelType}"
    )

    print("Model loaded, running inference...", flush=True)

    # Load image
    image = Image.open("${tempImagePath.replace(/\\/g, '/')}")

    # Run appropriate method based on model type
    if "${modelType}" == "detection":
        # Object detection (README line 549-563)
        detections = backend.detect(
            image,
            confidence_threshold=0.5,
            iou_threshold=0.45,
            max_detections=100
        )
        # Convert DetectionResult objects to dicts
        detections_dict = [{'label': d.class_name, 'confidence': d.confidence, 'box': [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2]} for d in detections]
        print(f"RESULT:{json.dumps({'detections': detections_dict})}", flush=True)
    else:
        # Classification (README line 569-582)
        predictions = backend.classify(image, top_k=5)
        # Convert ClassificationResult objects to dicts
        predictions_dict = [{'label': p.class_name, 'confidence': p.confidence} for p in predictions]
        print(f"RESULT:{json.dumps({'predictions': predictions_dict})}", flush=True)

except Exception as e:
    print(f"ERROR:{str(e)}", flush=True, file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
`

        await writeFile(tempScript, pythonScript, 'utf-8')

        const proc = spawn('uv', ['run', 'python', tempScript], {
          cwd: runtimeDir,
          stdio: 'pipe',
        })

        let output = ''

        proc.stdout?.on('data', (data) => {
          const text = data.toString()
          output += text

          const lines = text.split('\n')
          for (const line of lines) {
            if (line.startsWith('RESULT:')) {
              const resultData = JSON.parse(line.substring(7))
              send({ result: resultData })
            } else if (line.trim() && !line.startsWith('ERROR:')) {
              send({ progress: line })
            }
          }
        })

        proc.stderr?.on('data', (data) => {
          const text = data.toString()
          if (text.startsWith('ERROR:')) {
            send({ error: text.substring(6) })
          } else {
            // Also send stderr as progress (might contain useful info)
            send({ progress: text })
          }
        })

        proc.on('close', async (code) => {
          // Clean up temp files
          try {
            await unlink(tempScript)
            await unlink(tempImagePath)
          } catch (e) {
            // Ignore cleanup errors
          }

          if (code === 0) {
            send({ status: 'complete' })
          } else {
            send({ error: `Vision inference failed with code ${code}` })
          }
          controller.close()
        })

        proc.on('error', async (error) => {
          try {
            await unlink(tempScript)
            await unlink(tempImagePath)
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
