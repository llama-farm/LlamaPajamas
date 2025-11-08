"""Simple usage example for Llama-Pajamas Runtime.

This demonstrates the basic usage patterns for loading and running models.
"""

import sys
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_pajamas_run import RuntimeConfig, ModelLoader, benchmark_generation


def example_basic_generation():
    """Example 1: Basic text generation."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Text Generation")
    print("=" * 60)

    # Configure runtime (using MLX backend for Mac)
    config = RuntimeConfig(
        backend="mlx",  # or "gguf" for CUDA/CPU
        model_path="./models/qwen3-8b",  # Path to quantized model directory
        max_tokens=100,
        temperature=0.7,
    )

    # Load and use model
    with ModelLoader(config) as loader:
        prompt = "Write a Python function to calculate fibonacci numbers:"
        print(f"\nPrompt: {prompt}")
        print("\nGenerating...\n")

        response = loader.generate(prompt)
        print(response)


def example_chat_completion():
    """Example 2: OpenAI-compatible chat completion."""
    print("\n" + "=" * 60)
    print("Example 2: OpenAI-Compatible Chat")
    print("=" * 60)

    config = RuntimeConfig(
        backend="mlx",
        model_path="./models/qwen3-8b",
        max_tokens=150,
        temperature=0.8,
    )

    with ModelLoader(config) as loader:
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "How do I reverse a string in Python?"},
        ]

        print("\nMessages:")
        for msg in messages:
            print(f"  {msg['role']}: {msg['content']}")
        print("\nGenerating...\n")

        response = loader.chat(messages)
        assistant_message = response["choices"][0]["message"]["content"]
        print(f"Assistant: {assistant_message}")


def example_streaming():
    """Example 3: Streaming generation."""
    print("\n" + "=" * 60)
    print("Example 3: Streaming Generation")
    print("=" * 60)

    config = RuntimeConfig(
        backend="mlx",
        model_path="./models/qwen3-8b",
        max_tokens=100,
        temperature=0.7,
    )

    with ModelLoader(config) as loader:
        prompt = "Explain how transformers work in machine learning:"
        print(f"\nPrompt: {prompt}")
        print("\nStreaming response:\n")

        for chunk in loader.generate(prompt, stream=True):
            print(chunk, end="", flush=True)
        print("\n")


def example_gguf_backend():
    """Example 4: Using GGUF backend (CUDA/CPU)."""
    print("\n" + "=" * 60)
    print("Example 4: GGUF Backend (CUDA/CPU)")
    print("=" * 60)

    config = RuntimeConfig(
        backend="gguf",
        model_path="./models/qwen3-8b/gguf/qwen3-8b_q4_k_m.gguf",  # Direct file path
        max_tokens=100,
        temperature=0.7,
        n_gpu_layers=-1,  # Offload all layers to GPU
    )

    with ModelLoader(config) as loader:
        prompt = "What is the capital of France?"
        print(f"\nPrompt: {prompt}")
        print("\nGenerating...\n")

        response = loader.generate(prompt)
        print(response)


def example_benchmarking():
    """Example 5: Performance benchmarking."""
    print("\n" + "=" * 60)
    print("Example 5: Performance Benchmarking")
    print("=" * 60)

    config = RuntimeConfig(
        backend="mlx",
        model_path="./models/qwen3-8b",
        max_tokens=200,
        temperature=0.7,
    )

    with ModelLoader(config) as loader:
        print("\nRunning benchmark...\n")
        result = benchmark_generation(
            loader,
            prompt="Write a detailed explanation of neural networks:",
            num_tokens=200,
            warmup_runs=1,
        )

        # Results are automatically printed, but you can also access the data
        print("\nBenchmark data as dict:")
        print(result.to_dict())


def main():
    """Run all examples (comment out as needed)."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Run examples (uncomment the ones you want to try)
    # example_basic_generation()
    # example_chat_completion()
    # example_streaming()
    # example_gguf_backend()
    # example_benchmarking()

    print(
        "\n"
        + "=" * 60
        + "\n"
        + "Note: Uncomment example functions in main() to run them.\n"
        + "Make sure you have a quantized model at ./models/qwen3-8b/\n"
        + "=" * 60
    )


if __name__ == "__main__":
    main()
