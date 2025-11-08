"""OpenAI-compatible API server for Llama-Pajamas runtime."""

import time
import json
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .config import RuntimeConfig
from .model_loader import ModelLoader
from .backends import Backend


# Global model loader instance
model_loader: Optional[ModelLoader] = None


class ChatMessage(BaseModel):
    """OpenAI chat message format."""

    role: str = Field(..., description="Role of the message sender (system, user, or assistant)")
    content: str = Field(..., description="Content of the message")


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request."""

    model: str = Field(default="llama-pajamas", description="Model identifier")
    messages: List[ChatMessage] = Field(..., description="List of messages in the conversation")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=None, description="Sampling temperature (0-2)")
    top_p: Optional[float] = Field(default=None, description="Nucleus sampling parameter (0-1)")
    stream: bool = Field(default=False, description="Whether to stream the response")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")


class CompletionRequest(BaseModel):
    """OpenAI completion request."""

    model: str = Field(default="llama-pajamas", description="Model identifier")
    prompt: str = Field(..., description="Input prompt")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=None, description="Sampling temperature (0-2)")
    top_p: Optional[float] = Field(default=None, description="Nucleus sampling parameter (0-1)")
    stream: bool = Field(default=False, description="Whether to stream the response")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")


def create_app(config: RuntimeConfig, backend: Backend) -> FastAPI:
    """Create FastAPI application with OpenAI-compatible endpoints.

    Args:
        config: Runtime configuration
        backend: Backend implementation (MLX or GGUF)

    Returns:
        FastAPI application
    """
    global model_loader

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Load model on startup, unload on shutdown."""
        global model_loader
        print(f"Loading model from {config.model_path}...")
        model_loader = ModelLoader(config, backend)
        model_loader.load()
        print(f"Model loaded successfully!")
        yield
        print("Unloading model...")
        model_loader.unload()
        print("Shutdown complete")

    app = FastAPI(
        title="Llama-Pajamas Runtime",
        description="OpenAI-compatible API for local LLM inference",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/v1/models")
    async def list_models():
        """List available models (OpenAI-compatible)."""
        return {
            "object": "list",
            "data": [
                {
                    "id": "llama-pajamas",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "llama-pajamas",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        """Generate chat completion (OpenAI-compatible).

        Supports both streaming and non-streaming responses.
        """
        if model_loader is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Convert Pydantic models to dicts
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

        if request.stream:
            # Streaming response
            return StreamingResponse(
                stream_chat_completion(
                    messages=messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=request.stop,
                ),
                media_type="text/event-stream",
            )
        else:
            # Non-streaming response
            response = model_loader.chat(
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
                stream=False,
            )
            return response

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        """Generate text completion (OpenAI-compatible).

        Supports both streaming and non-streaming responses.
        """
        if model_loader is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        if request.stream:
            # Streaming response
            return StreamingResponse(
                stream_completion(
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=request.stop,
                ),
                media_type="text/event-stream",
            )
        else:
            # Non-streaming response
            response_text = model_loader.generate(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
                stream=False,
            )

            return {
                "id": f"cmpl-{int(time.time())}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "text": response_text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": model_loader.count_tokens(request.prompt),
                    "completion_tokens": model_loader.count_tokens(response_text),
                    "total_tokens": model_loader.count_tokens(request.prompt + response_text),
                },
            }

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        if model_loader is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return {"status": "healthy", "model_loaded": model_loader._loaded}

    return app


async def stream_chat_completion(
    messages: List[Dict[str, str]],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stop: Optional[List[str]] = None,
):
    """Stream chat completion in Server-Sent Events format.

    Yields OpenAI-compatible streaming chunks.
    """
    global model_loader

    # Get streaming response from backend
    response_stream = model_loader.chat(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        stream=True,
    )

    # Stream chunks in SSE format
    for chunk in response_stream:
        # Format as OpenAI streaming chunk
        chunk_data = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "llama-pajamas",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk_data)}\n\n"

    # Send final chunk with finish_reason
    final_chunk = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "llama-pajamas",
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


async def stream_completion(
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stop: Optional[List[str]] = None,
):
    """Stream text completion in Server-Sent Events format.

    Yields OpenAI-compatible streaming chunks.
    """
    global model_loader

    # Get streaming response from backend
    response_stream = model_loader.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        stream=True,
    )

    # Stream chunks in SSE format
    for chunk in response_stream:
        chunk_data = {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "llama-pajamas",
            "choices": [
                {
                    "text": chunk,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk_data)}\n\n"

    # Send final chunk
    final_chunk = {
        "id": f"cmpl-{int(time.time())}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": "llama-pajamas",
        "choices": [
            {
                "text": "",
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"
