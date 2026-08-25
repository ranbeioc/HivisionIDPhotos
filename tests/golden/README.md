# Licensed Golden Matrix input

Release evidence must contain all five lanes: `gradio`, `service`, `gateway`,
`pages`, and `vps`. Do not commit private ID photos, biometric material, model
outputs derived from unlicensed photos, or signed Asset URLs.

1. Copy `manifest.example.json` to an encrypted/private fixture workspace.
2. Replace every placeholder hash and path with an approved fixture/output.
3. Produce the same relative output tree for all five lanes.
4. Run:

```text
python scripts/run_golden_matrix.py \
  --manifest D:/private-xhalo-image-golden/manifest.json \
  --lane gradio=D:/private-xhalo-image-golden/output/gradio \
  --lane service=D:/private-xhalo-image-golden/output/service \
  --lane gateway=D:/private-xhalo-image-golden/output/gateway \
  --lane pages=D:/private-xhalo-image-golden/output/pages \
  --lane vps=D:/private-xhalo-image-golden/output/vps \
  --report D:/private-xhalo-image-golden/reports/matrix.json
```

Release runs must not use `--allow-subset`. A missing lane, missing output,
contract mismatch, deterministic hash drift, SSIM below 0.995, alpha-mask IoU
below 0.995, or alpha geometry deviation above 1 px fails the command.

The fixture set must cover front, slight profile, multiple faces, no face,
large/small inputs, JPEG/PNG/transparent PNG, EXIF rotation, light/dark
backgrounds, every enabled model, sizing, geometry, beauty, watermark, DPI,
target-KB, print-layout and selected-output combination required by the release
plan. The manifest is evidence metadata; the actual private images stay outside
Git.

Run the ARM model matrix inside a fresh compute container or on the isolated VPS
Preview runtime so each model/face-model combination gets a separate child
process:

```text
python scripts/benchmark_models.py \
  --image /private-fixtures/front.jpg --mime image/jpeg \
  --matting-model modnet_photographic_portrait_matting \
  --face-model mtcnn --face-model retinaface-resnet50 \
  --iterations 10 --report /reports/arm-models.json
```

Add RMBG and BiRefNet only after their model files and license gates are present.
The benchmark intentionally fails unavailable combinations instead of silently
skipping them.
