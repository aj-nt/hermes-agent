"""Vision utility functions for image detection and data URL processing.

Extracted from run_agent.py to decouple vision handling from the AIAgent god-object.
"""

import base64
import mimetypes
import tempfile
from pathlib import Path
from typing import Any, Optional

def content_has_image_parts(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    for part in content:
        if isinstance(part, dict) and part.get("type") in {"image_url", "input_image"}:
            return True
    return False


def materialize_data_url_for_vision(image_url: str) -> tuple[str, Optional[Path]]:
    header, _, data = str(image_url or "").partition(",")
    mime = "image/jpeg"
    if header.startswith("data:"):
        mime_part = header[len("data:"):].split(";", 1)[0].strip()
        if mime_part.startswith("image/"):
            mime = mime_part
    suffix = {
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
    }.get(mime, ".jpg")
    tmp = tempfile.NamedTemporaryFile(prefix="anthropic_image_", suffix=suffix, delete=False)
    with tmp:
        tmp.write(base64.b64decode(data))
    path = Path(tmp.name)
    return str(path), path
