from __future__ import annotations
import os
import yaml
from typing import Any, Dict, List, Tuple, Optional

class CfgNode:
    """
    Thin wrapper over dict to allow dot-access:
      cfg.model.vision_backbone
    """
    def __init__(self, d: Dict[str, Any]):
        self._d = d

    def __getattr__(self, k: str) -> Any:
        if k in self._d:
            v = self._d[k]
            if isinstance(v, dict):
                return CfgNode(v)
            return v
        raise AttributeError(k)

    def get(self, k: str, default: Any=None) -> Any:
        v = self._d.get(k, default)
        if isinstance(v, dict):
            return CfgNode(v)
        return v

    def as_dict(self) -> Dict[str, Any]:
        return self._d

def _deep_set(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value

def _parse_value(s: str) -> Any:
    # Try YAML parsing for types: numbers, bool, lists, etc.
    try:
        return yaml.safe_load(s)
    except Exception:
        return s

def load_config(path: str, overrides: Optional[List[str]] = None) -> CfgNode:
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    if overrides:
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"override must be key=value, got: {item}")
            k, v = item.split("=", 1)
            _deep_set(d, k.strip(), _parse_value(v.strip()))
    return CfgNode(d)

def resolve_device(device: str) -> str:
    device = (device or "auto").lower()
    if device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    if device in ("cuda", "cpu"):
        return device
    return "cpu"
