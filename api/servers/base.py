from pydantic import BaseModel, Field
import httpx
import asyncio
import json
from typing import List, Dict, Optional


class Message(BaseModel):
    role: str
    content: str


class OpenAIProxyArgs(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = False
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=1, ge=0, le=1)
    n: int = Field(default=1, ge=1)
    max_tokens: Optional[int] = None
    presence_penalty: float = Field(default=0, ge=-2, le=2)
    frequency_penalty: float = Field(default=0, ge=-2, le=2)


async def stream_openai_response(endpoint: str, payload: Dict, headers: Dict):
    """Stream OpenAI-compatible API response with proper error handling."""
    from loguru import logger
    
    timeout = httpx.Timeout(60.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            async with client.stream("POST", endpoint, json=payload, headers=headers) as response:
                # Check for HTTP errors
                if response.status_code != 200:
                    try:
                        error_text = await response.aread()
                        error_json = json.loads(error_text.decode('utf-8'))
                        error_data = {
                            "error": error_json.get("error", {
                                "message": f"API error: {response.status_code}",
                                "type": "api_error",
                                "code": response.status_code
                            })
                        }
                        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                    except Exception:
                        error_data = {
                            "error": {
                                "message": f"HTTP {response.status_code} error",
                                "type": "http_error",
                                "code": response.status_code
                            }
                        }
                        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                    return
                
                # Stream response line by line
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Forward SSE format lines
                    if line.startswith("data: "):
                        yield line + "\n\n"
                    elif line.startswith(":"):
                        # SSE comment, forward as-is
                        yield line + "\n"
                    elif line.strip() == "data: [DONE]":
                        yield "data: [DONE]\n\n"
                        break
                    else:
                        # Some APIs might send data without "data: " prefix
                        yield f"data: {line}\n\n"
                        
        except httpx.TimeoutException:
            error_data = {
                "error": {
                    "message": "Request timeout",
                    "type": "timeout_error"
                }
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            error_data = {
                "error": {
                    "message": str(e),
                    "type": "stream_error"
                }
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
