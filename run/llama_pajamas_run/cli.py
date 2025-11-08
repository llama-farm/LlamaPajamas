#!/usr/bin/env python3
"""CLI for llama-pajamas runtime with hardware auto-configuration."""

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Optional


def auto_configure(model_size: str = "7-8B", use_case: str = "general", verbose: bool = False) -> Path:
    """Auto-configure: detect hardware and generate runtime config.

    Returns:
        Path to generated config file
    """
    # Import hardware detection and config generation
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "quant" / "scripts"))

    from detect_hardware import HardwareDetector
    from generate_runtime_config import RuntimeConfigGenerator

    if verbose:
        print("Detecting hardware...")

    # Detect hardware
    detector = HardwareDetector()
    hardware = detector.detect()

    if verbose:
        print(f"  Detected: {hardware.display_name}")
        print(f"  Backend: {hardware.recommended_backend}")

    # Generate config
    generator = RuntimeConfigGenerator()
    config = generator.generate_config(
        hardware=hardware,
        model_size=model_size,
        use_case=use_case
    )

    # Write to temp file
    config_path = Path(tempfile.gettempdir()) / "llama_pajamas_runtime_config.json"
    generator.export_json(config, config_path)

    if verbose:
        print(f"  Generated config: {config_path}")
        print(f"  Settings: gpu_layers={config.n_gpu_layers}, threads={config.n_threads}, "
              f"batch={config.n_batch}, ubatch={config.n_ubatch}, ctx={config.n_ctx}")

    return config_path


def run_interactive(
    model_path: str,
    config_path: Optional[str] = None,
    auto_config: bool = False,
    model_size: str = "7-8B",
    use_case: str = "general",
    backend: Optional[str] = None,
    verbose: bool = False
):
    """Run interactive chat session."""
    # Auto-configure if requested
    if auto_config:
        config_path = str(auto_configure(model_size=model_size, use_case=use_case, verbose=verbose))

    # Load config to determine backend if not specified
    if config_path and backend is None:
        import json
        with open(config_path) as f:
            config = json.load(f)
            backend = config.get("backend", "cpu")

    # Auto-detect backend from model path if still not specified
    if backend is None:
        if Path(model_path).is_dir():
            backend = "mlx"
        elif str(model_path).endswith(".gguf"):
            backend = "gguf"
        else:
            print("Error: Could not determine backend. Use --backend or --auto-configure")
            sys.exit(1)

    # Load appropriate backend
    if backend == "mlx":
        from llama_pajamas_run.backends.mlx_backend import MLXBackend
        runtime = MLXBackend()
    else:  # gguf, cuda, metal, rocm, cpu
        from llama_pajamas_run.backends.gguf_backend import GGUFBackend
        runtime = GGUFBackend()

    # Load model
    print(f"Loading model: {model_path}")
    print(f"Backend: {backend}")

    if config_path:
        runtime.load_model(model_path, config_path=config_path, verbose=verbose)
    else:
        runtime.load_model(model_path, verbose=verbose)

    print("\nModel loaded! Type your message (Ctrl+C to exit)\n")

    # Interactive loop with chat history
    messages = []
    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            # Add user message
            messages.append({"role": "user", "content": user_input})

            print("Assistant: ", end="", flush=True)

            # Use chat_completion if available (GGUF), otherwise fallback to generate
            if hasattr(runtime, 'chat_completion'):
                # Stream chat completion (uses model's chat template)
                response_chunks = runtime.chat_completion(
                    messages=messages,
                    max_tokens=200,
                    temperature=0.7,
                    stream=True
                )

                assistant_message = ""
                for chunk in response_chunks:
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            print(content, end="", flush=True)
                            assistant_message += content

                # Add assistant response to history
                messages.append({"role": "assistant", "content": assistant_message})
            else:
                # Fallback: raw generate (MLX)
                assistant_message = ""
                for token in runtime.generate(
                    user_input,
                    max_tokens=200,
                    temperature=0.7,
                    stream=True
                ):
                    print(token, end="", flush=True)
                    assistant_message += token

                messages.append({"role": "assistant", "content": assistant_message})

            print("\n")

    except KeyboardInterrupt:
        print("\n\nExiting...")
        runtime.unload()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="llama-pajamas runtime with hardware auto-configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-configure and run (detects hardware automatically)
  llama-pajamas-run --model ./models/qwen3-8b/gguf/Q4_K_M/*.gguf --auto-configure

  # Use pre-generated config
  llama-pajamas-run --model ./models/qwen3-8b/gguf/Q4_K_M/*.gguf --config config.json

  # Manual settings
  llama-pajamas-run --model ./models/qwen3-8b/mlx/ --backend mlx

  # Auto-configure with specific use case
  llama-pajamas-run --model model.gguf --auto-configure --use-case speed
        """
    )

    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to model (GGUF file or MLX directory)"
    )
    parser.add_argument(
        "--config",
        help="Path to runtime config JSON"
    )
    parser.add_argument(
        "--auto-configure",
        action="store_true",
        help="Auto-detect hardware and generate optimal config"
    )
    parser.add_argument(
        "--model-size",
        choices=["7-8B", "13B", "30B+"],
        default="7-8B",
        help="Model size category (for auto-configure)"
    )
    parser.add_argument(
        "--use-case",
        choices=["general", "long_context", "speed", "quality"],
        default="general",
        help="Use case optimization (for auto-configure)"
    )
    parser.add_argument(
        "--backend",
        choices=["mlx", "gguf", "cuda", "metal", "rocm", "cpu"],
        help="Backend to use (auto-detected if not specified)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    run_interactive(
        model_path=args.model,
        config_path=args.config,
        auto_config=args.auto_configure,
        model_size=args.model_size,
        use_case=args.use_case,
        backend=args.backend,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
