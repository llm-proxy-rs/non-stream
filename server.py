#!/usr/bin/env python3
"""
Proxy that converts SSE streaming responses from llm-proxy-rs
back into non-streaming Anthropic /v1/messages responses.

Listens on port 3001, forwards to llm-proxy-rs on port 3000.
"""

import json
import os

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

UPSTREAM = os.environ.get("UPSTREAM", "http://localhost:3000")
PORT = int(os.environ.get("PORT", "3001"))
TIMEOUT = float(os.environ.get("TIMEOUT", "600"))

app = FastAPI()

# Hop-by-hop headers that should not be forwarded by a proxy.
_HOP_BY_HOP = frozenset({
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "upgrade",
    "te",
    "trailer",
})


def _proxy_headers(request: Request) -> dict[str, str]:
    return {k: v for k, v in request.headers.items() if k not in _HOP_BY_HOP}


def reassemble_message(events: list[dict]) -> dict:
    """Reassemble SSE events into a single Anthropic Messages API response."""
    message = {}
    content_blocks: dict[int, dict] = {}

    for event in events:
        etype = event.get("type")

        if etype == "message_start":
            message = event["message"]
            message.pop("type", None)

        elif etype == "content_block_start":
            idx = event["index"]
            content_blocks[idx] = event["content_block"]

        elif etype == "content_block_delta":
            idx = event["index"]
            delta = event["delta"]
            block = content_blocks[idx]
            delta_type = delta.get("type", "")

            if delta_type == "text_delta":
                block.setdefault("text", "")
                block["text"] += delta.get("text", "")
            elif delta_type == "thinking_delta":
                block.setdefault("thinking", "")
                block["thinking"] += delta.get("thinking", "")
            elif delta_type == "signature_delta":
                block["signature"] = block.get("signature", "") + delta.get("signature", "")
            elif delta_type == "input_json_delta":
                if not isinstance(block.get("input"), str):
                    block["input"] = ""
                block["input"] += delta.get("partial_json", "")
            elif delta_type == "citations_delta":
                block.setdefault("citations", [])
                citation = delta.get("citation")
                if citation:
                    block["citations"].append(citation)

        elif etype == "content_block_stop":
            idx = event["index"]
            block = content_blocks[idx]
            # Parse accumulated tool input JSON string for any tool-like block
            if isinstance(block.get("input"), str):
                try:
                    block["input"] = json.loads(block["input"]) if block["input"] else {}
                except json.JSONDecodeError:
                    block["input"] = {}

        elif etype == "message_delta":
            delta = event.get("delta", {})
            if "stop_reason" in delta:
                message["stop_reason"] = delta["stop_reason"]
            if "stop_sequence" in delta:
                message["stop_sequence"] = delta["stop_sequence"]
            usage = event.get("usage", {})
            if usage:
                message.setdefault("usage", {}).update(usage)

    message["content"] = [content_blocks[i] for i in sorted(content_blocks)]
    message["type"] = "message"
    return message


def parse_sse(text: str) -> list[dict]:
    """Parse SSE text into a list of data payloads."""
    events = []
    for block in text.split("\n\n"):
        for line in block.strip().splitlines():
            if line.startswith("data: "):
                raw = line[6:]
                if raw.strip() == "[DONE]":
                    continue
                try:
                    events.append(json.loads(raw))
                except json.JSONDecodeError:
                    continue
    return events


@app.api_route("/v1/messages", methods=["POST"])
async def proxy_messages(request: Request):
    body = await request.json()

    headers = _proxy_headers(request)

    # If caller wants streaming, passthrough the SSE stream directly
    if body.get("stream"):
        async def stream_upstream():
            async with httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT)) as client:
                async with client.stream(
                    "POST",
                    f"{UPSTREAM}/v1/messages",
                    json=body,
                    headers=headers,
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk

        return StreamingResponse(
            stream_upstream(),
            media_type="text/event-stream",
        )

    # Non-streaming: force stream on upstream, then reassemble
    body["stream"] = True

    async with httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT)) as client:
        resp = await client.post(
            f"{UPSTREAM}/v1/messages",
            json=body,
            headers=headers,
        )

    if resp.status_code != 200:
        try:
            content = resp.json()
        except Exception:
            content = {"error": resp.text or "upstream error"}
        return JSONResponse(status_code=resp.status_code, content=content)

    events = parse_sse(resp.text)
    message = reassemble_message(events)
    return JSONResponse(content=message)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def passthrough(request: Request, path: str):
    """Forward everything else to upstream as-is."""
    body = await request.body()
    headers = _proxy_headers(request)

    async with httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT)) as client:
        resp = await client.request(
            method=request.method,
            url=f"{UPSTREAM}/{path}",
            content=body,
            headers=headers,
        )

    return JSONResponse(status_code=resp.status_code, content=resp.json())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
