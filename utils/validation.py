"""Input validation helpers."""
import re
from typing import Optional

from werkzeug.utils import secure_filename


def safe_gesture_name(name: str) -> Optional[str]:
    if not name or not isinstance(name, str):
        return None
    name = name.strip().upper()
    if not re.match(r"^[A-Z0-9_ ]{1,50}$", name):
        return None
    return name


def allowed_upload(filename: str, allowed: set[str]) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


def secure_upload_name(filename: str) -> str:
    return secure_filename(filename) or "upload"
