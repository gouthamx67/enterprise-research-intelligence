import hashlib


def content_hash(text: str) -> str:
    normalized = text.encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()