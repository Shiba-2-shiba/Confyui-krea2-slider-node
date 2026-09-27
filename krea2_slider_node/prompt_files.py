"""Packaged prompt selection, safe YAML parsing, and content-based invalidation."""
import hashlib
from pathlib import Path

import yaml

from .conditioning import validate_prompt_records


def list_prompt_files(directory):
    root = Path(directory).resolve()
    return sorted(path.name for path in root.iterdir()
                  if path.is_file() and path.suffix.lower() in (".yaml", ".yml") and path.resolve().parent == root)


def resolve_prompt_file(directory, name):
    root = Path(directory).resolve()
    if name not in list_prompt_files(root) or Path(name).name != name:
        raise ValueError("Select a YAML from the Krea2 prompts list")
    path = (root / name).resolve()
    if path.parent != root:
        raise ValueError("Prompt YAML must stay inside the prompts directory")
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Prompt YAML exceeds the 2 MiB limit")
    return path


def prompt_file_fingerprint(directory, name):
    return hashlib.sha256(resolve_prompt_file(directory, name).read_bytes()).hexdigest()


def load_prompt_file(directory, name):
    path = resolve_prompt_file(directory, name)
    try:
        records = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid prompt YAML in {name}: {error}") from error
    return validate_prompt_records(records)
