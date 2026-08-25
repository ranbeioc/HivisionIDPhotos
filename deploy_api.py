"""Private XHalo compute entrypoint."""

from __future__ import annotations

import os

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=os.environ.get("HIVISION_HOST", "0.0.0.0"),
        port=int(os.environ.get("HIVISION_PORT", "8090")),
        workers=1,
        proxy_headers=False,
        server_header=False,
    )
