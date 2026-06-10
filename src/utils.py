"""Shared utilities: config loading, logging setup, path resolution."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load YAML config file relative to project root."""
    path = PROJECT_ROOT / config_path
    if not path.exists():
        # Try absolute path as given
        path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with coloured console output."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
    datefmt = "%H:%M:%S"

    try:
        import colorlog
        handler = colorlog.StreamHandler(sys.stdout)
        handler.setFormatter(
            colorlog.ColoredFormatter(
                "%(log_color)s" + fmt,
                datefmt=datefmt,
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
            )
        )
    except ImportError:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    root = logging.getLogger()
    root.setLevel(numeric)
    if root.handlers:
        root.handlers.clear()
    root.addHandler(handler)


def resolve_path(cfg: Dict[str, Any], key: str) -> Path:
    """Resolve a paths.* config entry relative to project root."""
    return PROJECT_ROOT / cfg["paths"][key]
