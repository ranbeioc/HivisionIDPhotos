from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.capacity import CapacityUnavailable, available_memory_mib, configured_min_available_mib
from services.model_registry import ModelRegistry


def main() -> int:
    available = available_memory_mib()
    minimum = configured_min_available_mib()
    try:
        ModelRegistry().ensure_ready()
    except CapacityUnavailable as error:
        print(json.dumps({
            "status": "pass",
            "decision": "rejected",
            "availableMemoryMiB": round(available, 2),
            "minimumMemoryMiB": minimum,
            "reason": str(error),
            "retryAfterSeconds": error.retry_after_seconds,
        }, separators=(",", ":")))
        return 0
    print(json.dumps({
        "status": "fail",
        "decision": "accepted-under-pressure",
        "availableMemoryMiB": round(available, 2),
        "minimumMemoryMiB": minimum,
    }, separators=(",", ":")))
    return 1


if __name__ == "__main__":
    sys.exit(main())
