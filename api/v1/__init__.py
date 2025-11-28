from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import httpx
from typing import Dict, Optional
from ..servers.base import stream_openai_response, OpenAIProxyArgs

router = APIRouter()

PLATFORM_API_URLS: Dict[str, str] = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "mistral": "https://api.mistral.ai/v1/chat/completions",
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "cerebras": "https://api.cerebras.ai/v1/chat/completions",
    "nvidia": "https://integrate.api.nvidia.com/v1/chat/completions",
    "sambanova": "https://api.sambanova.ai/v1/chat/completions",
}


@router.post("/chat/completions")
async def proxy_v1_chat_completions(
    args: OpenAIProxyArgs, 
    authorization: str = Header(...),
    x_platform: Optional[str] = Header(None, alias="X-Platform")
):
    """
    OpenAI compatible /v1/chat/completions endpoint.
    Defaults to OpenAI, but can be overridden with X-Platform header.
    """
    # Default to openai if no platform specified
    platform = (x_platform or "openai").lower()
    
    if platform not in PLATFORM_API_URLS:
        raise HTTPException(
            status_code=404, 
            detail=f"Platform '{platform}' not supported. Supported platforms: {', '.join(PLATFORM_API_URLS.keys())}"
        )

    api_url = PLATFORM_API_URLS[platform]
    api_key = authorization.split(" ")[1] if " " in authorization else authorization
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = args.dict(exclude_none=True)

    if args.stream:
        return StreamingResponse(
            stream_openai_response(api_url, payload, headers),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Content-Type-Options": "nosniff",
                "X-Experimental-Stream-Data": "true"
            }
        )
    else:
        timeout = httpx.Timeout(60.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.post(api_url, json=payload, headers=headers)
                response.raise_for_status()
                return JSONResponse(response.json())
            except httpx.HTTPStatusError as e:
                raise HTTPException(
                    status_code=e.response.status_code, detail=str(e.response.text))
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def proxy_v1_models(
    authorization: str = Header(...),
    x_platform: Optional[str] = Header(None, alias="X-Platform")
):
    """
    OpenAI compatible /v1/models endpoint.
    Defaults to OpenAI, but can be overridden with X-Platform header.
    """
    platform = (x_platform or "openai").lower()
    
    if platform not in PLATFORM_API_URLS:
        raise HTTPException(
            status_code=404, 
            detail=f"Platform '{platform}' not supported"
        )

    # Map platform to their models endpoint
    models_urls: Dict[str, str] = {
        "openai": "https://api.openai.com/v1/models",
        "mistral": "https://api.mistral.ai/v1/models",
        "groq": "https://api.groq.com/openai/v1/models",
        "cerebras": "https://api.cerebras.ai/v1/models",
        "nvidia": "https://integrate.api.nvidia.com/v1/models",
        "sambanova": "https://api.sambanova.ai/v1/models",
    }
    
    api_url = models_urls.get(platform)
    if not api_url:
        raise HTTPException(
            status_code=404, 
            detail=f"Models endpoint not available for platform '{platform}'"
        )
    
    api_key = authorization.split(" ")[1] if " " in authorization else authorization
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    timeout = httpx.Timeout(60.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.get(api_url, headers=headers)
            response.raise_for_status()
            return JSONResponse(response.json())
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code, detail=str(e.response.text))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

