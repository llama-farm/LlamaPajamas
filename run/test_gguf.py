"""Quick test for GGUF backend."""

import sys
from pathlib import Path
import glob

sys.path.insert(0, str(Path(__file__).parent))

from llama_pajamas_run import RuntimeConfig, ModelLoader


def test_gguf():
    """Test GGUF backend with llama-cpp-python."""
    print("\n" + "=" * 70)
    print("Testing GGUF Backend (llama-cpp-python)")
    print("=" * 70)

    # Find GGUF file
    gguf_files = glob.glob("../quant/models/qwen3-8b/gguf/*.gguf")
    if not gguf_files:
        print("❌ No GGUF files found")
        return False

    gguf_path = gguf_files[0]
    print(f"\nFound GGUF model: {Path(gguf_path).name}")
    print(f"Size: {Path(gguf_path).stat().st_size / (1024**3):.2f} GB")

    # Configure runtime
    config = RuntimeConfig(
        backend="gguf",
        model_path=gguf_path,
        max_tokens=50,
        temperature=0.7,
        n_gpu_layers=-1,  # All layers to Metal GPU on Mac
        verbose=True,
    )

    print(f"\nConfig:")
    print(f"  Backend: {config.backend}")
    print(f"  Model: {Path(config.model_path).name}")
    print(f"  Max tokens: {config.max_tokens}")
    print(f"  GPU layers: {config.n_gpu_layers}")

    print(f"\nLoading model (this may take a moment)...")
    try:
        with ModelLoader(config) as loader:
            print(f"✅ Model loaded successfully!")

            # Test 1: Basic generation
            print("\n" + "-" * 70)
            print("Test 1: Basic Generation")
            print("-" * 70)

            prompt = "Write a Python function to calculate factorial:"
            print(f"\nPrompt: {prompt}")
            print(f"\nGenerating...")

            response = loader.generate(prompt, max_tokens=50)
            print(f"\nResponse:")
            print(response)

            tokens = loader.count_tokens(prompt + response)
            print(f"\nTotal tokens: {tokens}")

            # Test 2: Chat completion
            print("\n" + "-" * 70)
            print("Test 2: OpenAI-Compatible Chat Completion")
            print("-" * 70)

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ]

            print(f"\nMessages:")
            for msg in messages:
                print(f"  {msg['role']}: {msg['content']}")

            print(f"\nGenerating...")
            response = loader.chat(messages, max_tokens=30)

            print(f"\nResponse:")
            assistant_msg = response["choices"][0]["message"]["content"]
            print(assistant_msg)

            usage = response.get("usage", {})
            print(f"\nUsage:")
            print(f"  Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
            print(f"  Completion tokens: {usage.get('completion_tokens', 'N/A')}")
            print(f"  Total tokens: {usage.get('total_tokens', 'N/A')}")

        print("\n" + "=" * 70)
        print("✅ GGUF Backend Test PASSED")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"\n❌ GGUF backend test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_gguf()
    sys.exit(0 if success else 1)
