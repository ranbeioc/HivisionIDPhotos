# Preview multi-user capacity check — 2026-08-25

## Scope

- Live path: `image-api-preview.xhalo.co` → Gateway → private Tunnel/Access → Hivision ARM64 → Assets D1/R2.
- Fixture: repository-owned `public/assets/example-id-photo-blue.png`; no user attachment was used.
- Workload: default UI output set (standard, HD, two transparent variants, layout and templates), each with preview/thumbnail derivatives.
- Runtime: immutable `xhalo/hivision-legacy@sha256:7aa5bac7989fec446b52389157604554c0f3329590091773990ddcecc9d4424a`, 2 CPU / 6 GiB container limit, 4-core / 23.4-GiB host, no swap.

## Results

| Virtual users | Requests | Success | End-to-end p95 | Hivision max processing | Peak container CPU | Peak container memory |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 100% | 14.315 s | 7.968 s | 92.16% | 311.7 MiB |
| 2 | 2 | 100% | 15.922 s | 10.629 s | 143.58% | 400.8 MiB |
| 4 | 4 | 100% | 24.496 s | 20.396 s | 170.75% | 549.9 MiB |
| 6 | 6 | 100% | 31.921 s | 27.465 s | 180.12% | 818.4 MiB |
| 4 (repeat after peak) | 4 | 100% | 24.172 s | 20.364 s | not resampled | 846.4 MiB idle after run |
| 6 with admission cap 4 | 6 | 4 × 200, 2 × 503 | 23.638 s (accepted) | 19.746 s | 173.73% | 570.2 MiB |

All 17 pre-cap full-path requests returned HTTP 200. Two post-cap six-user runs each accepted four requests and rejected two with `MODEL_UNAVAILABLE` in about five seconds; accepted requests completed within 25.901 seconds. No OOM, container restart, Queue/DLQ error or co-tenant health failure was observed. The host retained at least 18.7 GiB `MemAvailable`; Auth, OCR, Crawl, Search and database containers remained running/healthy.

## Decision

- The Preview runtime is usable for up to four simultaneous default generations on this VPS: p95 remains below the 30-second product gate and memory headroom is large.
- Six simultaneous default generations are not release-ready because end-to-end p95 exceeds 30 seconds even though the Hivision origin itself remains below its 30-second timeout.
- The deployed Preview runtime sets `HIVISION_MAX_INFLIGHT_REQUESTS=4`; requests above that bound return `503` with `Retry-After`. Production Canary must retain this bound and deterministic allowlisting. Do not raise it from CPU/RAM headroom alone; the synchronous latency gate is the limiting factor.
- Resident memory rose from about 308 MiB after the first warm request to 846 MiB after the boundary/repeat runs. This is compatible with model/output allocator high-water retention, but a longer soak is still required before Production to distinguish a stable high-water mark from gradual growth.
- The benchmark script is `xhalo-image/scripts/load-preview.mjs`; repeat it with the same immutable image digest and keep generated anonymous fixtures for administrator-managed archive/purge rather than automatic cleanup.
