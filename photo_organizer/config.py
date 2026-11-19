"""Load and validate config.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config.toml"

KINDS = ("nef", "jpg", "heic", "xmp")


@dataclass(frozen=True)
class ScanConfig:
    format_subfolders: frozenset[str] = frozenset({"raw", "nef", "jpg", "jpeg"})
    skip_dirs: frozenset[str] = frozenset({"_quarantine", ".photoorganizer"})
    path_warn_length: int = 240
    # lowercase extension (with dot) -> kind
    extensions: dict[str, str] = field(default_factory=lambda: {
        ".nef": "nef", ".jpg": "jpg", ".jpeg": "jpg",
        ".heic": "heic", ".heif": "heic", ".xmp": "xmp",
    })


@dataclass(frozen=True)
class Config:
    library_root: Path | None = None
    scan: ScanConfig = field(default_factory=ScanConfig)


class ConfigError(ValueError):
    pass


def _str_set(value, key: str) -> frozenset[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"{key} must be a list of strings")
    return frozenset(v.lower() for v in value)


def load(path: Path | str | None = None) -> Config:
    """Load config.toml. A missing default file gives the built-in defaults."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        if path == DEFAULT_CONFIG_PATH:
            return Config()
        raise ConfigError(f"config file not found: {path}")
    with path.open("rb") as f:
        data = tomllib.load(f)

    lib = data.get("library", {})
    root = lib.get("root")
    library_root = None
    if root:
        library_root = Path(root)
        if not library_root.is_absolute():
            library_root = (path.parent / library_root).resolve()

    s = data.get("scan", {})
    defaults = ScanConfig()
    extensions = dict(defaults.extensions)
    if "extensions" in s:
        extensions = {}
        for kind, exts in s["extensions"].items():
            if kind not in KINDS:
                raise ConfigError(f"scan.extensions: unknown kind {kind!r} (expected one of {KINDS})")
            for ext in _str_set(exts, f"scan.extensions.{kind}"):
                if not ext.startswith("."):
                    raise ConfigError(f"scan.extensions.{kind}: {ext!r} must start with '.'")
                if ext in extensions:
                    raise ConfigError(f"scan.extensions: {ext!r} listed under two kinds")
                extensions[ext] = kind

    warn_len = s.get("path_warn_length", defaults.path_warn_length)
    if not isinstance(warn_len, int) or warn_len <= 0:
        raise ConfigError("scan.path_warn_length must be a positive integer")

    scan = ScanConfig(
        format_subfolders=_str_set(s["format_subfolders"], "scan.format_subfolders")
        if "format_subfolders" in s else defaults.format_subfolders,
        skip_dirs=_str_set(s["skip_dirs"], "scan.skip_dirs")
        if "skip_dirs" in s else defaults.skip_dirs,
        path_warn_length=warn_len,
        extensions=extensions,
    )
    return Config(library_root=library_root, scan=scan)
