#!/usr/bin/env python
''' Convert Gemini API to OpenAI API format

Gemini API docs:
- https://ai.google.dev/gemini-api/docs/text-generation?lang=rest
'''
from loguru import logger
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Header, Query
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import typing
from typing import List, Dict, Optional
from .base import Message
import time
import json
import re
import uuid

router = APIRouter()


GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"
GEMINI_STREAM_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{}:streamGenerateContent"


class OpenAIProxyArgs(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    stream: bool = False
    temperature: float = 0.7
    top_p: float = 1
    n: int = 1
    max_tokens: Optional[int] = None
    presence_penalty: float = 0
    frequency_penalty: float = 0


class MessageConverter:
    def __init__(self, messages: List[Dict[str, str]]):
        self.messages = messages

    def convert(self) -> List[Dict[str, str]]:
        converted_messages = []
        for message in self.messages:
            role = "user" if message["role"] == "user" else "model"
            converted_messages.append({
                "role": role,
                "parts": [{"text": message["content"]}]
            })
        return converted_messages


def convert_gemini_to_openai_response(gemini_response: dict, model: str) -> dict:
    """Convert Gemini API response to OpenAI-compatible format."""
    return {
        "id": gemini_response.get("candidates", [{}])[0].get("content", {}).get("role", ""),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "usage": {
            "prompt_tokens": 0,  # Gemini doesn't provide token counts
            "completion_tokens": 0,
            "total_tokens": 0
        },
        "choices": [{
            "message": {
                "role": "assistant",
                "content": gemini_response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            },
            "finish_reason": "stop",
            "index": 0
        }]
    }


async def stream_gemini_response(model: str, payload: dict, api_key: str):
    """Stream Gemini API response and convert to OpenAI SSE format."""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created_time = int(time.time())
    
    # Increase timeout for Gemini API which can be slow
    timeout = httpx.Timeout(120.0, connect=30.0, read=120.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            async with client.stream(
                "POST",
                GEMINI_STREAM_ENDPOINT.format(model),
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key
                }
            ) as response:
                # Check for HTTP errors
                if response.status_code != 200:
                    error_text = await response.aread()
                    error_data = {
                        "error": {
                            "message": f"Gemini API error: {response.status_code}",
                            "type": "api_error",
                            "code": response.status_code
                        }
                    }
                    yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                    return
                
                # Process streaming response - Gemini returns JSONL format
                buffer = ""
                has_sent_content = False
                async for chunk in response.aiter_bytes():
                    try:
                        buffer += chunk.decode('utf-8', errors='ignore')
                        
                        # Process complete JSON objects (JSONL format)
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()
                            if not line:
                                continue
                            
                            try:
                                # Parse Gemini response
                                gemini_data = json.loads(line)
                                
                                # Extract text content from candidates
                                candidates = gemini_data.get("candidates", [])
                                if candidates:
                                    candidate = candidates[0]
                                    content = candidate.get("content", {})
                                    parts = content.get("parts", [])
                                    
                                    for part in parts:
                                        text = part.get("text", "")
                                        if text:
                                            has_sent_content = True
                                            # Convert to OpenAI format
                                            openai_chunk = {
                                                "id": chat_id,
                                                "object": "chat.completion.chunk",
                                                "created": created_time,
                                                "model": model,
                                                "choices": [{
                                                    "index": 0,
                                                    "delta": {
                                                        "content": text
                                                    },
                                                    "finish_reason": None
                                                }]
                                            }
                                            yield f"data: {json.dumps(openai_chunk, ensure_ascii=False)}\n\n"
                                    
                                    # Check if this is the final chunk
                                    finish_reason = candidate.get("finishReason")
                                    if finish_reason:
                                        final_chunk = {
                                            "id": chat_id,
                                            "object": "chat.completion.chunk",
                                            "created": created_time,
                                            "model": model,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {},
                                                "finish_reason": finish_reason.lower() if finish_reason else "stop"
                                            }]
                                        }
                                        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                                        return
                            except json.JSONDecodeError as e:
                                # Log and skip invalid JSON lines
                                logger.debug(f"Invalid JSON line: {line[:100]}")
                                continue
                            except Exception as e:
                                logger.error(f"Error processing Gemini stream: {e}, line: {line[:100]}")
                                continue
                    except Exception as e:
                        logger.error(f"Error reading stream chunk: {e}")
                        continue
                
                # If we didn't send any content, there might be an issue
                if not has_sent_content and buffer:
                    logger.warning(f"Received data but no content extracted. Buffer: {buffer[:200]}")
                
        except httpx.TimeoutException:
            error_data = {
                "error": {
                    "message": "Request timeout",
                    "type": "timeout_error"
                }
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            error_data = {
                "error": {
                    "message": str(e),
                    "type": "stream_error"
                }
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
        finally:
            # Always send [DONE] marker
            yield "data: [DONE]\n\n"


@router.post("/chat/completions")
async def proxy_chat_completions(
    args: OpenAIProxyArgs,
    authorization: str = Header(...),
):
    api_key = authorization.split(" ")[1]
    model = args.model

    if not api_key:
        raise HTTPException(status_code=400, detail="API key not provided")

    # Transform args into Gemini API format
    gemini_payload = {
        "contents": MessageConverter(args.messages).convert(),
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH"
            }
        ],
        "generationConfig": {
            "temperature": args.temperature,
            "maxOutputTokens": args.max_tokens,
            "topP": args.top_p,
            "topK": 10
        }
    }

    if args.stream:
        return StreamingResponse(
            stream_gemini_response(model, gemini_payload, api_key),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Experimental-Stream-Data": "true"
            }
        )
    else:
        # Increase timeout for Gemini API which can be slow
        timeout = httpx.Timeout(120.0, connect=30.0, read=120.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                GEMINI_ENDPOINT.format(model),
                json=gemini_payload,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key
                }
            )

            if response.status_code != 200:
                return JSONResponse(content=response.json(), status_code=response.status_code)

            response_json = response.json()

            # Use the new conversion function
            openai_compatible_response = convert_gemini_to_openai_response(
                response_json, args.model)

            return JSONResponse(openai_compatible_response)
