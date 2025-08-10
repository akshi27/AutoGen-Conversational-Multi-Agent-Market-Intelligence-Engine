"""
Small helpers used by multiple agents.
"""
from typing import Dict, Any

def make_payload(status: str, data: Any = None, files: list | None = None, error: str | None = None) -> Dict[str, Any]:
    return {"status": status, "data": data, "files": files or [], "error": error}
