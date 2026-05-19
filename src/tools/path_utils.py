__all__ = [
    "get_root_directory",
    "get_default_results_directory",
    "shorten_path_component",
]

import hashlib
from pathlib import Path


def get_root_directory():
    return Path(__file__).parent.parent.parent


def get_default_results_directory():
    return get_root_directory() / "results"


def shorten_path_component(value: str, max_len: int = 120) -> str:
    """Truncate a path component so pipeline cache dirs stay under NAME_MAX (255)."""
    value = str(value)
    if len(value) <= max_len:
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{value[: max_len - len(digest) - 1]}_{digest}"
