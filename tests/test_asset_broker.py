import asyncio

import httpx

from services.asset_broker import AssetBrokerClient


def _response(status: int, url: str, payload: dict | None = None, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(status, json=payload, headers=headers, request=httpx.Request("POST", url))


def _descriptor(variant: str = "standard") -> dict:
    return {
        "assetId": f"asset-{variant}",
        "variant": variant,
        "mime": "image/jpeg",
        "width": 295,
        "height": 413,
        "bytes": 3,
        "dpi": 300,
        "previewUrl": "https://assets.test/preview",
        "downloadUrl": "https://assets.test/download",
        "urlExpiresAt": "2026-09-04T00:05:00Z",
        "saved": False,
    }


def test_upload_retries_transient_put_and_finalize_failures(monkeypatch):
    calls: list[tuple[str, str]] = []
    failures = {"put": 1, "finalize": 1}

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def request(self, method, url, **_kwargs):
            calls.append((method, url))
            if method == "POST" and url.endswith("/asset-uploads"):
                return _response(201, url, {
                    "uploadSessionId": "session-1",
                    "uploadUrl": "https://assets.test/internal/v1/uploads/token",
                    "requiredHeaders": {"content-type": "image/jpeg"},
                })
            if method == "PUT" and failures["put"]:
                failures["put"] -= 1
                return _response(500, url)
            if url.endswith("/finalize") and failures["finalize"]:
                failures["finalize"] -= 1
                return _response(503, url, headers={"retry-after": "0"})
            if method == "PUT":
                return _response(204, url)
            return _response(201, url, {"asset": _descriptor()})

    async def no_wait(_seconds):
        return None

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(asyncio, "sleep", no_wait)
    broker = AssetBrokerClient("https://gateway.test/internal/v1/asset-uploads", "grant", "request-1", "secret")
    descriptor = asyncio.run(broker.upload("standard", "image/jpeg", b"abc", 295, 413, 300, {}))

    assert descriptor.asset_id == "asset-standard"
    assert len([call for call in calls if call[0] == "PUT"]) == 2
    assert len([call for call in calls if call[1].endswith("/finalize")]) == 2


def test_upload_retries_session_creation_only_once(monkeypatch):
    calls = 0

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def request(self, method, url, **_kwargs):
            nonlocal calls
            if method == "POST" and url.endswith("/asset-uploads"):
                calls += 1
                if calls == 1:
                    return _response(503, url)
                return _response(201, url, {
                    "uploadSessionId": "session-2",
                    "uploadUrl": "https://assets.test/internal/v1/uploads/token-2",
                    "requiredHeaders": {"content-type": "image/jpeg"},
                })
            if method == "PUT":
                return _response(204, url)
            return _response(201, url, {"asset": _descriptor()})

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    broker = AssetBrokerClient("https://gateway.test/internal/v1/asset-uploads", "grant", "request-2", "secret")

    descriptor = asyncio.run(broker.upload("standard", "image/jpeg", b"abc", 295, 413, 300, {}))

    assert descriptor.asset_id == "asset-standard"
    assert calls == 2
