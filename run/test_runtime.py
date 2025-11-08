"""Test script for Llama-Pajamas Runtime.

Tests both MLX and GGUF backends with the quantized Qwen3-8B model.
"""

import sys
from pathlib import Path

# Add to path for development
sys.path.insert(0, str(Path(__file__).parent))

from llama_pajamas_run import RuntimeConfig, ModelLoader


def test_mlx_backend():
    """Test MLX backend (Apple Silicon)."""
    print("\n" + "=" * 70)
    print("Testing MLX Backend (Apple Silicon)")
    print("=" * 70)

    try:
        config = RuntimeConfig(
            backend="mlx",
            model_path="../quant/models/qwen3-8b/mlx",
            max_tokens=50,
            temperature=0.7,
        )

        print(f"\nConfig:")
        print(f"  Backend: {config.backend}")
        print(f"  Model: {config.model_path}")
        print(f"  Max tokens: {config.max_tokens}")

        print(f"\nLoading model...")
        with ModelLoader(config) as loader:
            print(f"✅ Model loaded successfully!")

            # Test generation
            prompt = "Write a Python function to add two numbers:"
            print(f"\nPrompt: {prompt}")
            print(f"\nGenerating (max {config.max_tokens} tokens)...")
            print("-" * 70)

            response = loader.generate(prompt, max_tokens=config.max_tokens)
            print(response)
            print("-" * 70)

            # Count tokens
            tokens = loader.count_tokens(prompt + response)
            print(f"\nTotal tokens: {tokens}")

        print(f"\n✅ MLX backend test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ MLX backend test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gguf_backend():
    """Test GGUF backend (llama.cpp)."""
    print("\n" + "=" * 70)
    print("Testing GGUF Backend (llama.cpp)")
    print("=" * 70)

    try:
        # Get GGUF file path
        import glob
        gguf_files = glob.glob("../quant/models/qwen3-8b/gguf/*.gguf")
        if not gguf_files:
            print("❌ No GGUF files found")
            return False

        gguf_path = gguf_files[0]

        config = RuntimeConfig(
            backend="gguf",
            model_path=gguf_path,
            max_tokens=50,
            temperature=0.7,
            n_gpu_layers=0,  # Use CPU for testing (set to -1 for GPU)
            verbose=False,
        )

        print(f"\nConfig:")
        print(f"  Backend: {config.backend}")
        print(f"  Model: {Path(config.model_path).name}")
        print(f"  Max tokens: {config.max_tokens}")
        print(f"  GPU layers: {config.n_gpu_layers}")

        print(f"\nLoading model...")
        with ModelLoader(config) as loader:
            print(f"✅ Model loaded successfully!")

            # Test generation
            prompt = "Write a Python function to multiply two numbers:"
            print(f"\nPrompt: {prompt}")
            print(f"\nGenerating (max {config.max_tokens} tokens)...")
            print("-" * 70)

            response = loader.generate(prompt, max_tokens=config.max_tokens)
            print(response)
            print("-" * 70)

            # Count tokens
            tokens = loader.count_tokens(prompt + response)
            print(f"\nTotal tokens: {tokens}")

        print(f"\n✅ GGUF backend test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ GGUF backend test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chat_completion():
    """Test OpenAI-compatible chat completion."""
    print("\n" + "=" * 70)
    print("Testing OpenAI-Compatible Chat Completion (MLX)")
    print("=" * 70)

    try:
        config = RuntimeConfig(
            backend="mlx",
            model_path="../quant/models/qwen3-8b/mlx",
            max_tokens=100,
            temperature=0.8,
        )

        with ModelLoader(config) as loader:
            messages = [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "What is Python?"},
            ]

            print(f"\nMessages:")
            for msg in messages:
                print(f"  {msg['role']}: {msg['content']}")

            print(f"\nGenerating...")
            print("-" * 70)

            response = loader.chat(messages, max_tokens=100)

            # Extract and print response
            assistant_message = response["choices"][0]["message"]["content"]
            print(assistant_message)
            print("-" * 70)

            # Print usage stats
            usage = response.get("usage", {})
            print(f"\nUsage:")
            print(f"  Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
            print(f"  Completion tokens: {usage.get('completion_tokens', 'N/A')}")
            print(f"  Total tokens: {usage.get('total_tokens', 'N/A')}")

        print(f"\n✅ Chat completion test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Chat completion test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Llama-Pajamas Runtime Test Suite")
    print("=" * 70)

    results = []

    # Test MLX backend
    results.append(("MLX Backend", test_mlx_backend()))

    # Test GGUF backend
    # results.append(("GGUF Backend", test_gguf_backend()))

    # Test chat completion
    results.append(("Chat Completion", test_chat_completion()))

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:30s} {status}")

    all_passed = all(passed for _, passed in results)
    print("=" * 70)

    if all_passed:
        print("\n🎉 All tests PASSED!")
        return 0
    else:
        print("\n❌ Some tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
