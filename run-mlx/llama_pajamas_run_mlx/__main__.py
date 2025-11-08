"""CLI entrypoint for Llama-Pajamas MLX Runtime."""

import argparse
import uvicorn

from llama_pajamas_run_core import RuntimeConfig, create_app
from .backend import MLXBackend


def main():
    """Run OpenAI-compatible API server with MLX backend."""
    parser = argparse.ArgumentParser(
        description="Llama-Pajamas MLX Runtime - OpenAI-compatible API server for Apple Silicon"
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to MLX model directory or manifest.json parent directory",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind server to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind server to (default: 8000)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Default max tokens for generation (default: 2048)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Default sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Default nucleus sampling parameter (default: 0.9)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Create runtime config
    config = RuntimeConfig(
        model_path=args.model_path,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        host=args.host,
        port=args.port,
        verbose=args.verbose,
    )

    # Create backend
    backend = MLXBackend()

    # Create FastAPI app
    app = create_app(config, backend)

    # Run server
    print(f"\nStarting Llama-Pajamas MLX Runtime")
    print(f"Backend: MLX (Apple Silicon)")
    print(f"Model: {config.model_path}")
    print(f"Server: http://{config.host}:{config.port}")
    print(f"\nOpenAI-compatible endpoints:")
    print(f"  POST http://{config.host}:{config.port}/v1/chat/completions")
    print(f"  POST http://{config.host}:{config.port}/v1/completions")
    print(f"  GET  http://{config.host}:{config.port}/v1/models")
    print(f"  GET  http://{config.host}:{config.port}/health")
    print(f"\nPress Ctrl+C to stop\n")

    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
