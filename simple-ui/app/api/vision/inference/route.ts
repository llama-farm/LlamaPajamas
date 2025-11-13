import { NextRequest } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'
import { writeFile, unlink } from 'fs/promises'
import { randomUUID } from 'crypto'

export async function POST(request: NextRequest) {
  const formData = await request.formData()
  const modelPath = formData.get('modelPath') as string
  const backend = formData.get('backend') as string || 'coreml'
  const taskType = formData.get('taskType') as string || 'auto' // 'classification', 'detection', 'segmentation', 'localization'
  const imageFile = formData.get('image') as File

  if (!modelPath || !imageFile) {
    return new Response(
      JSON.stringify({ error: 'Missing modelPath or image' }),
      { status: 400, headers: { 'Content-Type': 'application/json' } }
    )
  }

  const runtimeMap: Record<string, string> = {
    'onnx': 'run-onnx',
    'coreml': 'run-coreml',
    'tensorrt': 'run-tensorrt',
  }

  const runtimeDir = path.join(process.cwd(), '..', runtimeMap[backend] || 'run-coreml')
  const tempDir = path.join(process.cwd(), '..', 'run')
  const tempImagePath = path.join(tempDir, `image-${randomUUID()}.png`)
  const tempScript = path.join(tempDir, `vision-${randomUUID()}.py`)
  const tempOutputImage = path.join(tempDir, `annotated-${randomUUID()}.png`)

  const stream = new ReadableStream({
    async start(controller) {
      const send = (data: any) => {
        controller.enqueue(`data: ${JSON.stringify(data)}\n\n`)
      }

      try {
        // Save uploaded image
        const imageBuffer = Buffer.from(await imageFile.arrayBuffer())
        await writeFile(tempImagePath, imageBuffer)

        send({ progress: 'Image saved, loading model...' })

        // Auto-detect model type if task is 'auto'
        const modelName = modelPath.toLowerCase()
        let detectedTask = taskType
        if (taskType === 'auto') {
          if (modelName.includes('yolo')) {
            detectedTask = 'detection'
          } else if (modelName.includes('mask') || modelName.includes('segment')) {
            detectedTask = 'segmentation'
          } else if (modelName.includes('vit') || modelName.includes('clip')) {
            detectedTask = 'classification'
          } else {
            detectedTask = 'detection'
          }
        }

        // Create comprehensive Python script
        const pythonScript = `
import sys
import json
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from llama_pajamas_run_${backend}.backends.vision import ${backend === 'coreml' ? 'CoreML' : backend === 'onnx' ? 'ONNX' : 'TensorRT'}VisionBackend
from llama_pajamas_run_core.utils import get_coco_class_names, get_imagenet_class_names

print("Loading vision model...", flush=True)

try:
    backend = ${backend === 'coreml' ? 'CoreML' : backend === 'onnx' ? 'ONNX' : 'TensorRT'}VisionBackend()

    # Determine model type for loading
    model_type = "${detectedTask}"
    if model_type == "localization":
        model_type = "detection"  # Localization uses detection backend
    elif model_type == "segmentation":
        model_type = "detection"  # For now, use detection (can be enhanced)

    backend.load_model(
        model_path="${modelPath.replace(/\\/g, '/')}",
        model_type=model_type,
        ${backend === 'onnx' ? 'providers=["CPUExecutionProvider"],' : ''}
    )

    print("Model loaded, running inference...", flush=True)

    # Load image
    image = Image.open("${tempImagePath.replace(/\\/g, '/')}")
    original_size = image.size

    # Run inference based on task type
    task = "${detectedTask}"
    results = {}
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        font = ImageFont.load_default()

    if task == "classification":
        # Pure classification - top K predictions
        predictions = backend.classify(image, top_k=5)

        # Get ImageNet class names if available
        try:
            from llama_pajamas_run_core.utils import get_imagenet_class_names
            imagenet_names = get_imagenet_class_names()
            predictions_with_names = []
            for p in predictions:
                class_id = p.class_id if hasattr(p, 'class_id') else int(p.class_name.replace('class_', ''))
                label = imagenet_names[class_id] if class_id < len(imagenet_names) else p.class_name
                predictions_with_names.append({
                    'label': label,
                    'confidence': p.confidence,
                    'class_id': class_id
                })
            results['predictions'] = predictions_with_names
        except:
            results['predictions'] = [{'label': p.class_name, 'confidence': p.confidence} for p in predictions]

        # Annotate image with top prediction
        if results['predictions']:
            top_pred = results['predictions'][0]
            text = f"{top_pred['label']}: {top_pred['confidence']:.1%}"
            draw.text((10, 10), text, fill='red', font=font)

    elif task == "localization":
        # Classification + Localization - find object and draw main bounding box
        detections = backend.detect(image, confidence_threshold=0.3)

        if detections:
            # Get highest confidence detection
            top_detection = max(detections, key=lambda d: d.confidence)

            # Scale bbox to image size
            bbox = top_detection.bbox
            x1 = int(bbox.x1 * original_size[0])
            y1 = int(bbox.y1 * original_size[1])
            x2 = int(bbox.x2 * original_size[0])
            y2 = int(bbox.y2 * original_size[1])

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline='green', width=3)

            # Draw label
            label = f"{top_detection.class_name}: {top_detection.confidence:.1%}"
            text_bbox = draw.textbbox((x1, y1-20), label, font=font)
            draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill='green')
            draw.text((x1, y1-20), label, fill='white', font=font)

            # Return as single-item detection list for UI compatibility
            results['detections'] = [{
                'label': top_detection.class_name,
                'confidence': top_detection.confidence,
                'box': [x1, y1, x2, y2]
            }]
        else:
            results['detections'] = []

    elif task == "detection":
        # Object detection - find all objects with bounding boxes
        detections = backend.detect(image, confidence_threshold=0.5, iou_threshold=0.45)

        colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'cyan', 'magenta']
        detections_list = []

        for idx, det in enumerate(detections):
            # Scale bbox
            bbox = det.bbox
            x1 = int(bbox.x1 * original_size[0])
            y1 = int(bbox.y1 * original_size[1])
            x2 = int(bbox.x2 * original_size[0])
            y2 = int(bbox.y2 * original_size[1])

            # Draw box
            color = colors[idx % len(colors)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            # Draw label
            label = f"{det.class_name}: {det.confidence:.1%}"
            text_bbox = draw.textbbox((x1, y1-20), label, font=font)
            draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill=color)
            draw.text((x1, y1-20), label, fill='white', font=font)

            detections_list.append({
                'label': det.class_name,
                'confidence': det.confidence,
                'box': [x1, y1, x2, y2]
            })

        results['detections'] = detections_list

    elif task == "segmentation":
        # Instance segmentation - detect + segment individual instances
        # For now, use detection with visual indication
        # TODO: Add proper mask-rcnn support when models available
        detections = backend.detect(image, confidence_threshold=0.5)

        segmentations = []
        for idx, det in enumerate(detections):
            bbox = det.bbox
            x1 = int(bbox.x1 * original_size[0])
            y1 = int(bbox.y1 * original_size[1])
            x2 = int(bbox.x2 * original_size[0])
            y2 = int(bbox.y2 * original_size[1])

            # Draw filled semi-transparent rectangle as "mask"
            overlay = Image.new('RGBA', original_size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            color = (255, idx*30 % 255, (255-idx*30) % 255, 100)  # Semi-transparent
            overlay_draw.rectangle([x1, y1, x2, y2], fill=color)
            annotated_image = Image.alpha_composite(annotated_image.convert('RGBA'), overlay).convert('RGB')

            # Redraw to get the draw object
            draw = ImageDraw.Draw(annotated_image)

            # Draw label
            label = f"{det.class_name}: {det.confidence:.1%}"
            draw.text((x1, y1-20), label, fill='white', font=font)

            segmentations.append({
                'label': det.class_name,
                'confidence': det.confidence,
                'box': [x1, y1, x2, y2],
                'has_mask': True  # Indicate it's segmented
            })

        results['segments'] = segmentations

    # Save annotated image
    annotated_image.save("${tempOutputImage.replace(/\\/g, '/')}")

    # Convert to base64 for JSON response
    buffered = BytesIO()
    annotated_image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    results['annotated_image'] = img_base64
    results['task_type'] = task

    print(f"RESULT:{json.dumps(results)}", flush=True)

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
        let buffer = ''

        proc.stdout?.on('data', (data) => {
          const text = data.toString()
          output += text
          buffer += text

          // Process complete lines only
          const lines = buffer.split('\n')
          // Keep the last incomplete line in the buffer
          buffer = lines.pop() || ''

          for (const line of lines) {
            if (line.startsWith('RESULT:')) {
              try {
                const resultData = JSON.parse(line.substring(7))
                send({ result: resultData })
              } catch (e) {
                // JSON might be incomplete, add back to buffer
                buffer = line + '\n' + buffer
              }
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
            send({ progress: text })
          }
        })

        proc.on('close', async (code) => {
          // Clean up temp files
          try {
            await unlink(tempScript)
            await unlink(tempImagePath)
            await unlink(tempOutputImage)
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
            await unlink(tempOutputImage)
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
