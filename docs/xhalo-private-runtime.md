# XHalo private runtime

This fork exposes Hivision through a private, signed FastAPI adapter. It is not
an Internet-facing API and must be reached only through the XHalo Image Gateway,
Cloudflare Tunnel and Cloudflare Access.

## Image targets

The default `production` target contains the service/API runtime only. Gradio is
intentionally absent, and the `production-default` model set excludes the
RetinaFace weight whose provenance is still under review.

```bash
docker build --platform linux/arm64 \
  --target production \
  --build-arg MODEL_SET=production-default \
  -t xhalo/hivision-compute:<immutable-version> .
```

The `legacy` target adds the internal Gradio regression client. It mounts
Gradio at `/legacy/gradio/` in the same process and shares the same
request-scoped service and `ModelRegistry`; it does not start a second model
server.

```bash
docker build --platform linux/arm64 \
  --target legacy \
  --build-arg MODEL_SET=preview-default \
  -t xhalo/hivision-legacy:<immutable-version> .
```

Only the immutable Production target may be published by the release workflow.
The Legacy target remains a loopback-only Preview/regression artifact.

## Preview startup

1. Copy `.env.preview.example` to `.env.preview`.
2. Replace `HIVISION_HMAC_SECRET` with an environment-specific random secret.
3. For a local build, leave `HIVISION_IMAGE_REF` on its local-only default and
   start `docker compose -f compose.preview.yaml up -d --build`.
4. For a VPS Preview deployment, set `HIVISION_IMAGE_REF` to the verified
   registry digest (`repository/image@sha256:...`), pull it, and start with
   `docker compose -f compose.preview.yaml up -d --no-build`. Never resolve a
   mutable tag during rollout.
5. Confirm the host listener is `127.0.0.1:18090`; never publish port 8090 on
   `0.0.0.0` at the VPS boundary.
6. Route the loopback listener through a dedicated Preview Tunnel protected by
   an Access service token. The browser must never receive Access or HMAC
   credentials.

The container is read-only, drops all Linux capabilities, enables
`no-new-privileges`, limits CPU/RAM/PIDs and places Gradio/Hugging Face caches
under the bounded `/tmp` tmpfs.

Preview reserves an 8 GiB cgroup ceiling for the optional BiRefNet v1 Lite
model. Its model-specific admission gate requires at least 7168 MiB available
before inference; lower headroom returns `MODEL_UNAVAILABLE` without loading the
ONNX graph. This does not change the single-compute semaphore or authorize the
Preview-only weight for Production.

## Release checks

Before using an image digest in Preview or Production:

```bash
python -m pytest -q -p no:cacheprovider \
  tests/test_contracts.py tests/test_release_gates.py
python scripts/verify_model_provenance.py --set preview-default
python scripts/verify_model_provenance.py --set production-default
docker run --rm --platform linux/arm64 \
  --network none --memory 1g --memory-swap 1g --pids-limit 256 \
  --read-only --tmpfs /tmp:size=64m,mode=1777 \
  --security-opt no-new-privileges:true --cap-drop ALL \
  --entrypoint python xhalo/hivision-compute:<immutable-version> \
  scripts/probe_memory_pressure.py
python -m pip_audit -r requirements.txt -r requirements-app.txt
python -m pip_audit -r requirements.txt -r requirements-app.txt \
  -r requirements-legacy.txt
docker compose -f compose.preview.yaml config --quiet
```

The licensed five-lane Golden corpus and real ARM benchmark are separate
release gates. Synthetic tests and a successful image build do not authorize a
Production rollout.

Every Docker target validates `models/provenance.json` after checksum
verification. A Preview-only model cannot enter a Production model set by a
manifest-only change; its provenance decision must first be explicitly updated
with the missing evidence and pass the release-gate tests.

Compute admission is non-blocking at concurrency one and requires at least
`HIVISION_MIN_AVAILABLE_MEMORY_MB` (default 2048 MiB) across both host
`MemAvailable` and cgroup headroom. Busy or memory-pressure requests return a
503 `MODEL_UNAVAILABLE` envelope with `Retry-After`; they do not wait for the
Gateway timeout or continue toward an OOM.

The full processing path is additionally bounded by
`HIVISION_MAX_INFLIGHT_REQUESTS` (default 4). The 2026-08-25 live ARM Preview
test kept four simultaneous default-output requests below the 30-second gate;
six exceeded it. Requests above the four-slot bound fail fast with the same
retryable 503 contract instead of increasing latency and retained buffers.
