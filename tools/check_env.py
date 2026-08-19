#!/usr/bin/env python3
"""Collect reproducible Phase 0 environment and CUDA observability data."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

def run(cmd: list[str]) -> dict:
    try:
        p = subprocess.run(cmd, text=True, capture_output=True, check=False)
        return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except FileNotFoundError:
        return {"cmd": cmd, "returncode": 127, "stdout": "", "stderr": "not found"}

def module_version(name: str) -> str | None:
    try:
        module = __import__(name)
        return getattr(module, "__version__", "installed")
    except Exception:
        return None

def main() -> None:
    out = Path(os.environ.get("PHASE0_OUT", "artifacts/phase0"))
    out.mkdir(parents=True, exist_ok=True)
    data: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "path": os.environ.get("PATH", ""),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "tools": {name: shutil.which(name) for name in ["nvcc", "nvidia-smi", "cmake", "g++", "git", "nsys", "ncu"]},
        "nvcc": run(["nvcc", "--version"]),
        "nvidia_smi": run(["nvidia-smi"]),
        "gpu_query": run(["nvidia-smi", "--query-gpu=index,name,compute_cap,memory.total,driver_version,pstate", "--format=csv,noheader"]),
        "git": run(["git", "rev-parse", "--show-toplevel"]),
        "packages": {name: module_version(name) for name in ["torch", "transformers", "safetensors", "sentencepiece", "psutil", "pynvml"]},
    }
    try:
        import torch
        data["torch_cuda"] = {
            "version": torch.version.cuda,
            "is_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [{"index": i, "name": torch.cuda.get_device_name(i), "capability": list(torch.cuda.get_device_capability(i)), "total_memory": torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())],
        }
        if torch.cuda.is_available():
            torch.cuda.init()
            data["torch_cuda"]["memory_allocated"] = torch.cuda.memory_allocated()
            data["torch_cuda"]["memory_reserved"] = torch.cuda.memory_reserved()
    except Exception as exc:
        data["torch_cuda_error"] = repr(exc)
    path = out / "environment.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"written": str(path), "gpu_query": data["gpu_query"]["stdout"], "packages": data["packages"]}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
