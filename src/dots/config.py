from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_EXPLICIT_PATH_KEYS = {
    "base_model_path",
    "cache_dir",
    "data_file",
    "dataset_path",
    "eval_script",
    "input_dir",
    "input_file",
    "original_data_path",
    "output_dir",
    "output_file",
    "output_img",
    "output_plot_dir",
    "path",
    "score_file_path",
    "source_model_path",
    "temp_model_dir",
}


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    project_root = _find_project_root(path)
    return _resolve_value(data, project_root)


def dump_json(payload: Any, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _resolve_value(value: Any, project_root: Path, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {item_key: _resolve_value(item_value, project_root, item_key) for item_key, item_value in value.items()}

    if isinstance(value, list):
        return [_resolve_value(item, project_root, key) for item in value]

    if isinstance(value, str) and _looks_like_path(key, value):
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            return str(candidate)
        return str((project_root / candidate).resolve())

    return value


def _looks_like_path(key: str | None, value: str) -> bool:
    if key in _EXPLICIT_PATH_KEYS:
        return True

    if key is not None and (
        key.endswith("_path")
        or key.endswith("_file")
        or key.endswith("_dir")
        or key in {"script", "cache"}
    ):
        return True

    if value.startswith(("/", "./", "../", "~/")):
        return True

    if "/" in value and not value.startswith(("http://", "https://")):
        return True

    return False


def _find_project_root(config_path: Path) -> Path:
    env_root = os.getenv("DOTS_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if (candidate / "pyproject.toml").exists():
            return candidate

    for candidate in [config_path.parent, *config_path.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate

    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate

    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError(f"Unable to locate project root for config: {config_path}")
