#!/usr/bin/env python3
"""带 TTL 的 JSON 文件缓存。

重复运行时优先复用本地行情快照和财务数据，避免反复请求数据源。高频变化的数据
使用较短有效期，低频财务数据使用较长有效期；传入空缓存目录即可关闭缓存。
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _cache_file(cache_dir: Path, namespace: str, key: str) -> Path:
    safe_key = "".join(c if (c.isalnum() or c in "-._") else "_" for c in key.upper())
    return cache_dir / namespace / f"{safe_key}.json"


def read_cache(
    cache_dir: str | Path | None,
    namespace: str,
    key: str,
    ttl_seconds: float | None,
) -> Any | None:
    """缓存存在且未过期时返回数据，否则返回 None。"""
    if not cache_dir:
        return None
    path = _cache_file(Path(cache_dir), namespace, key)
    if not path.exists():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if ttl_seconds is not None:
        age = time.time() - float(blob.get("cached_at_epoch", 0))
        if age < 0 or age > ttl_seconds:
            return None
    return blob.get("data")


def write_cache(
    cache_dir: str | Path | None,
    namespace: str,
    key: str,
    data: Any,
) -> None:
    """Atomically persist data under cache_dir/namespace/key.json."""
    if not cache_dir:
        return
    path = _cache_file(Path(cache_dir), namespace, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "cached_at_epoch": time.time(),
        "cached_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "data": data,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(blob, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
