# ARM benchmark evidence

`2026-08-25-vps-arm-preview.json` records a native ARM Preview preflight on the
production VPS. The container had no network or published port, used a read-only
root filesystem, dropped all capabilities, and was limited to 2 CPUs, 6 GiB RAM
and 256 PIDs. Existing containers were not restarted or modified; zero unhealthy
containers and 19 GiB available memory were observed after the run.
After the report hash was verified and the evidence below was recorded, the
dedicated `/tmp/xhalo-hivision-benchmark-20260825` directory and the exact
benchmark image digest were removed. No Hivision image, container or listener
was left on the VPS.

The test input was the visual reference supplied with the XHalo Image task. It
is suitable for exercising model loading, face detection and capacity, but it
is not the licensed Golden fixture corpus. These numbers therefore validate the
benchmark path and the current Preview capacity only. They do not satisfy the
Golden Matrix, legacy-baseline comparison, RMBG or BiRefNet release gates. The
no-swap memory-pressure rejection gate was verified separately against the
final Production candidate; it is not inferred from this benchmark report.

The JSON preserves the exact image digest actually benchmarked. Capacity and
benchmark-runner fixes were mounted read-only for the successful run and then
baked into the later final local candidates; the historical digest is not
rewritten to imply that a different image was measured.

The benchmark exposed and fixed two release-tool defects before producing the
passing report: append-style CLI model values were not unwrapped in child
processes, and legacy progress logs polluted the child's JSON stdout. Tests now
cover both boundaries.
