from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import uuid

import torch
from safetensors.torch import save_file


def validate_output_name(name):
    if not re.fullmatch(r"[\w-][\w.-]{0,95}", name) or name in (".", ".."):
        raise ValueError("Output name must be a filename stem, without directory separators")


def save_adapter(state, report, directory, name="krea2_slider"):
    validate_output_name(name)
    if not state or any(not torch.isfinite(t).all() for t in state.values()):
        raise ValueError("Refusing to save empty or non-finite LoRA tensors")
    report_text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    target, report_path = directory / (stem + ".safetensors"), directory / (stem + ".json")
    temporary = directory / (stem + ".tmp.safetensors")
    temporary_report = directory / (stem + ".tmp.json")
    settings = report.get("settings", {})
    metadata = {"base_model_architecture": "krea2", "prediction_type": "flow_velocity",
                "ss_network_module": "networks.lora", "ss_network_dim": str(settings.get("rank", "")),
                "ss_network_alpha": str(settings.get("alpha", "")), "slider_settings": json.dumps(settings)}
    try:
        save_file({key: value.detach().cpu().contiguous() for key, value in state.items()}, str(temporary), metadata=metadata)
        temporary_report.write_text(report_text, encoding="utf-8")
        os.replace(temporary, target)
        os.replace(temporary_report, report_path)
    finally:
        temporary.unlink(missing_ok=True)
        temporary_report.unlink(missing_ok=True)
    return str(target.resolve()), str(report_path.resolve())
