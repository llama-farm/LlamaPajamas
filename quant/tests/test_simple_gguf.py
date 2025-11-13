#!/usr/bin/env python3
"""
Simplified E2E Test for LLM Runtime (Standalone)
Tests GGUF format only without requiring full LlamaPajamas installation
"""

import os
import sys
import json
import time
import tempfile
from pathlib import Path
from typing import Dict, Optional


def check_dependencies():
    """Check if required dependencies are available."""
    required = {
        'huggingface_hub': 'huggingface-hub',
        'llama_cpp': 'llama-cpp-python',
    }

    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        print("❌ Missing required packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\nInstall with:")
        print(f"pip install {' '.join(missing)}")
        return False

    return True


def test_gguf_runtime():
    """Test GGUF runtime with a simple model."""
    print("=" * 70)
    print("🧪 Simple GGUF Runtime Test")
    print("=" * 70)

    if not check_dependencies():
        return False

    try:
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        print("\n📦 Downloading test model...")
        # Use a very small pre-quantized GGUF model
        model_path = hf_hub_download(
            repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            cache_dir="./hf_cache"
        )
        print(f"✅ Model downloaded: {model_path}")

        print("\n🔧 Loading model...")
        llm = Llama(
            model_path=model_path,
            n_ctx=512,
            n_threads=4,
            n_gpu_layers=0,  # CPU only
            verbose=False
        )
        print("✅ Model loaded")

        print("\n🎯 Testing generation...")
        prompt = "Q: What is the capital of France?\nA:"

        start_time = time.time()
        output = llm(
            prompt,
            max_tokens=32,
            temperature=0.7,
            stop=["Q:", "\n"],
            echo=False
        )
        elapsed = time.time() - start_time

        response = output['choices'][0]['text'].strip()
        tokens = output['usage']['completion_tokens']
        tokens_per_sec = tokens / elapsed if elapsed > 0 else 0

        print(f"✅ Generation successful")
        print(f"   Prompt: {prompt}")
        print(f"   Response: {response}")
        print(f"   Tokens: {tokens}")
        print(f"   Speed: {tokens_per_sec:.2f} tokens/sec")
        print(f"   Time: {elapsed:.2f}s")

        # Save results
        results = {
            "test": "gguf_runtime",
            "model": "TinyLlama-1.1B-Q4_K_M",
            "success": True,
            "tokens": tokens,
            "tokens_per_sec": tokens_per_sec,
            "elapsed": elapsed,
            "response": response
        }

        with open("test_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print("\n💾 Results saved to test_results.json")
        print("\n✅ Test PASSED")

        return True

    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_gguf_runtime()
    sys.exit(0 if success else 1)
