"""Test MLX runtime with API server."""

import requests
import json
import time
import subprocess
import sys
from pathlib import Path


def wait_for_server(host="127.0.0.1", port=8001, timeout=60):
    """Wait for server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(f"http://{host}:{port}/health", timeout=2)
            if response.status_code == 200:
                print(f"Server is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    return False


def test_chat_completion(host="127.0.0.1", port=8001):
    """Test chat completion endpoint."""
    print("\n" + "=" * 70)
    print("Test 1: Chat Completion (Non-Streaming)")
    print("=" * 70)

    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": "llama-pajamas",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Python in one sentence?"},
        ],
        "max_tokens": 50,
        "temperature": 0.7,
        "stream": False,
    }

    print(f"\nRequest:")
    print(json.dumps(payload, indent=2))

    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()

    result = response.json()
    print(f"\nResponse:")
    print(json.dumps(result, indent=2))

    assert "choices" in result
    assert len(result["choices"]) > 0
    assert "message" in result["choices"][0]
    assert "content" in result["choices"][0]["message"]

    print("\n✅ Chat completion test PASSED")
    return True


def test_streaming_chat(host="127.0.0.1", port=8001):
    """Test streaming chat completion."""
    print("\n" + "=" * 70)
    print("Test 2: Streaming Chat Completion")
    print("=" * 70)

    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": "llama-pajamas",
        "messages": [
            {"role": "user", "content": "Count from 1 to 5."},
        ],
        "max_tokens": 30,
        "stream": True,
    }

    print(f"\nRequest:")
    print(json.dumps(payload, indent=2))
    print(f"\nStreaming response:")
    print("-" * 70)

    response = requests.post(url, json=payload, stream=True, timeout=60)
    response.raise_for_status()

    chunks_received = 0
    for line in response.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    print("\n" + "-" * 70)
                    print("Stream complete")
                    break
                try:
                    chunk = json.loads(data)
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            print(delta["content"], end="", flush=True)
                            chunks_received += 1
                except json.JSONDecodeError:
                    pass

    assert chunks_received > 0, "No chunks received"
    print(f"\nReceived {chunks_received} chunks")
    print("\n✅ Streaming chat test PASSED")
    return True


def test_list_models(host="127.0.0.1", port=8001):
    """Test list models endpoint."""
    print("\n" + "=" * 70)
    print("Test 3: List Models")
    print("=" * 70)

    url = f"http://{host}:{port}/v1/models"
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    result = response.json()
    print(f"\nResponse:")
    print(json.dumps(result, indent=2))

    assert "data" in result
    assert len(result["data"]) > 0

    print("\n✅ List models test PASSED")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Llama-Pajamas MLX Runtime Test Suite")
    print("=" * 70)

    model_path = str(Path(__file__).parent.parent / "quant" / "models" / "qwen3-8b" / "mlx")
    print(f"\nModel path: {model_path}")

    # Note: Server should be started separately:
    # cd run-mlx
    # uv run llama-pajamas-mlx --model-path ../quant/models/qwen3-8b/mlx --port 8001

    host = "127.0.0.1"
    port = 8001

    print(f"\nChecking server at http://{host}:{port}...")
    if not wait_for_server(host, port, timeout=10):
        print("\n❌ Server not ready. Please start the server first:")
        print(f"   cd run-mlx")
        print(f"   uv run llama-pajamas-mlx --model-path {model_path} --port {port}")
        sys.exit(1)

    # Run tests
    results = []
    try:
        results.append(("List Models", test_list_models(host, port)))
        results.append(("Chat Completion", test_chat_completion(host, port)))
        results.append(("Streaming Chat", test_streaming_chat(host, port)))
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

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
