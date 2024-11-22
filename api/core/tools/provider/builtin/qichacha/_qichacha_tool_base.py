import hashlib
from datetime import datetime


def build_headers(key: str, secret_key: str) -> dict:
    time_span = str(int(datetime.now().timestamp()))
    headers = {
        "Timespan": time_span,
        "Token": hashlib.md5((key + time_span + secret_key).encode("utf-8"))
        .hexdigest()
        .upper(),
    }
    return headers
