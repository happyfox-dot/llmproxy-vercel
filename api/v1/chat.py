#!/usr/bin/env python
''' OpenAI v1 compatible endpoint for Gemini API
支持 OpenAI 官方 v1 路径格式的 Gemini 代理
'''
from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import JSONResponse, StreamingResponse
from api.servers.gemini import (
    OpenAIProxyArgs,
    MessageConverter,
    convert_gemini_to_openai_response,
    stream_gemini_response,
    GEMINI_ENDPOINT
)
import httpx

router = APIRouter()


@router.post("/chat/completions")
async def v1_chat_completions(
    args: OpenAIProxyArgs,
    authorization: str = Header(...),
):
    """OpenAI v1 格式的聊天完成接口，使用 Gemini 后端"""
    api_key = authorization.split(" ")[1] if " " in authorization else authorization
    model = args.model

    if not api_key:
        raise HTTPException(status_code=400, detail="API key not provided")

    # Transform args into Gemini API format
    converter = MessageConverter(args.messages)
    contents = converter.convert()

    generation_config = {
        "temperature": args.temperature,
        "maxOutputTokens": args.max_tokens,
        "topP": args.top_p,
        "topK": 10,
        "candidateCount": args.n
    }

    if args.stop:
        if isinstance(args.stop, str):
            generation_config["stopSequences"] = [args.stop]
        else:
            generation_config["stopSequences"] = args.stop

    gemini_payload = {
        "contents": contents,
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH"
            }
        ],
        "generationConfig": generation_config
    }
    
    if converter.system_instruction:
        gemini_payload["system_instruction"] = converter.system_instruction

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

