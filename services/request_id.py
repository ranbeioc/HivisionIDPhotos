from __future__ import annotations

import re


_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def is_valid_request_id(value: str) -> bool:
    return _REQUEST_ID_PATTERN.fullmatch(value) is not None
