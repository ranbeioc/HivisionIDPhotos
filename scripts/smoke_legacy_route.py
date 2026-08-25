"""Render the internal Gradio route through the mounted ASGI application."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.main import app


async def main() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://loopback.invalid") as client:
        response = await client.get("/legacy/gradio/")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/html")
    print("Legacy Gradio route rendered successfully")


if __name__ == "__main__":
    asyncio.run(main())
