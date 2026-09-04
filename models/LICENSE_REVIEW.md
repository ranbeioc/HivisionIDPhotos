# Model license and provenance gate

Reviewed: 2026-08-28. Machine-readable evidence and release decisions live in
`models/provenance.json`; `scripts/verify_model_provenance.py` is enforced by
CI and every Docker build.

| Model | Current decision | Evidence |
| --- | --- | --- |
| MODNet photographic portrait matting | Preview permitted; production license gate satisfied for the upstream code/model family. The downloaded Hivision release object remains checksum-pinned. | Upstream states its code, models, and demos are Apache-2.0: https://github.com/ZHKKKe/MODNet |
| Hivision MODNet | Preview-only until the exact training checkpoint, dataset rights and ONNX conversion provenance are documented. The Hivision release asset is immutable by byte length and SHA-256. | https://github.com/Zeyi-Lin/HivisionIDPhotos/releases/tag/pretrained-model |
| RetinaFace ResNet-50 ONNX | Preview-only. Hivision PR #90 and release asset `191588278` establish uploader, date and integration commit; ONNX metadata identifies a PyTorch `torch-jit-export` graph. They do not establish the exact checkpoint, conversion command, training-data/weight redistribution statement, or an explicit mapping to the MIT implementation. The Production Docker model set excludes this weight and the provenance gate rejects attempts to add it. | https://github.com/Zeyi-Lin/HivisionIDPhotos/pull/90 and https://github.com/biubug6/Pytorch_Retinaface/blob/master/LICENSE.MIT |
| BRIA RMBG-1.4 | Preview evaluation only; disabled for production. The immutable Hugging Face revision is under a BRIA non-commercial model license, not a general permissive license. | https://huggingface.co/briaai/RMBG-1.4 |
| BiRefNet | Preview evaluation only; disabled for production until the selected weight and its training-data/model license are reviewed. Upstream explicitly notes that some third-party weights are non-commercial. | https://github.com/ZhengPeng7/BiRefNet |

No model may be added to the production model set solely because its identifier is supported by legacy source code. A model entry needs an immutable source URL, byte length, SHA-256, architecture/runtime compatibility, and a reviewed license/provenance decision.

The RetinaFace restriction can be lifted only after the missing evidence listed
in `models/provenance.json` is supplied and reviewed. A code-family license or
an inference-success result is not a substitute for binary-weight provenance.
