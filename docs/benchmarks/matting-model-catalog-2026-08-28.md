# Preview matting model catalog gate — 2026-08-28

- Host: Oracle ARM64 VPS, 2 CPU allocated per isolated test container.
- Image: `xhalo-hivision-preview:2b3843f`, Python 3.11.13, ONNX Runtime 1.20.1.
- Fixture: repository-owned `demo/images/test0.jpg`; no user upload was used.
- Each model ran one cold and two warm ID-photo passes with MTCNN.

| Model | Status | Warm p95 | Peak RSS | Preview decision |
| --- | --- | ---: | ---: | --- |
| MODNet photographic portrait | pass | 0.853 s | 332 MiB | enabled |
| Hivision MODNet | pass | 0.839 s | 336 MiB | enabled |
| RMBG 1.4 | pass | 5.626 s | 956 MiB | enabled, Preview evaluation only |
| BiRefNet v1 Lite | pass at 8 GiB; 6 GiB run was OOM-killed | 23.994 s | 6682 MiB | enabled behind 7168 MiB admission, Preview evaluation only |

All successful runs stayed below the 30-second synchronous compute gate. The
BiRefNet result requires the Preview container ceiling to remain 8 GiB and the
model-specific admission check to fail closed before graph allocation. Compute
concurrency remains one. Production continues to contain only the approved
MODNet photographic portrait weight.
