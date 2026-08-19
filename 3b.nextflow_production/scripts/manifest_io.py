"""Small YAML reader/writer for the pilot manifest shape."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    if value in {"null", "None", "~"}:
        return None
    if value in {"true", "false"}:
        return value == "true"
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load the small pilot manifest from YAML or JSON."""
    path = Path(path)
    text = path.read_text()
    try:
        import yaml

        loaded = yaml.safe_load(text)
        return loaded or {}
    except Exception:
        pass

    stripped = text.lstrip()
    if stripped.startswith("{"):
        return json.loads(text)

    root: dict[str, Any] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped_line = raw.strip()
        if not stripped_line or stripped_line.startswith("#"):
            i += 1
            continue
        if raw.startswith(" ") or ":" not in raw:
            raise ValueError(f"Unsupported manifest syntax at line {i + 1}: {raw}")

        key, value = raw.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            root[key] = _parse_scalar(value)
            i += 1
            continue

        children: list[str] = []
        mapping: dict[str, Any] = {}
        i += 1
        while i < len(lines) and (lines[i].startswith("  ") or not lines[i].strip()):
            child = lines[i].strip()
            i += 1
            if not child or child.startswith("#"):
                continue
            if child.startswith("- "):
                children.append(child[2:].strip())
            elif ":" in child:
                child_key, child_value = child.split(":", 1)
                mapping[child_key.strip()] = _parse_scalar(child_value)
            else:
                raise ValueError(f"Unsupported manifest syntax near {key}: {child}")
        root[key] = children if children else mapping
    return root


def dump_manifest(
    data: dict[str, Any],
    path: str | Path,
    field_comments: dict[str, str] | None = None,
) -> None:
    """Write the small pilot manifest in stable YAML order. When
    field_comments is given, a '# ...' comment line is written directly
    above any top-level key present in it -- PyYAML can't emit comments
    from a plain dict, so each key's block is dumped separately instead of
    the whole dict at once when comments are requested."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml

        if field_comments:
            blocks: list[str] = []
            for key, value in data.items():
                comment = field_comments.get(key)
                if comment:
                    blocks.append(f"# {comment}")
                blocks.append(yaml.safe_dump({key: value}, sort_keys=False).rstrip("\n"))
            path.write_text("\n".join(blocks) + "\n")
        else:
            path.write_text(yaml.safe_dump(data, sort_keys=False))
        return
    except Exception:
        pass

    lines: list[str] = []
    for key, value in data.items():
        comment = (field_comments or {}).get(key)
        if comment:
            lines.append(f"# {comment}")
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for child_key, child_value in value.items():
                lines.append(f"  {child_key}: {child_value}")
        elif isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines) + "\n")


def require_manifest_paths(manifest: dict[str, Any]) -> list[str]:
    """Return missing path validation messages for one pilot manifest row."""
    errors: list[str] = []
    mask_paths = manifest.get("mask_paths") or {}
    if isinstance(mask_paths, dict) and mask_paths:
        for compartment, path in mask_paths.items():
            mask_path = Path(str(path or ""))
            if not mask_path.is_file():
                errors.append(f"mask_paths.{compartment} is not a file: {mask_path}")
    else:
        mask_path = Path(str(manifest.get("mask_path") or ""))
        if not mask_path.is_file():
            errors.append(f"mask_path is not a file: {mask_path}")
    channel_paths = manifest.get("channel_paths") or {}
    if not isinstance(channel_paths, dict) or not channel_paths:
        errors.append("channel_paths must be a non-empty mapping")
        return errors
    for channel, path in channel_paths.items():
        channel_path = Path(str(path or ""))
        if not channel_path.is_file():
            errors.append(f"channel_paths.{channel} is not a file: {channel_path}")
    return errors
