#!/usr/bin/env python3
"""Offline Transformers reference for the native Qwen3 tokenizer tests."""
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    except Exception as exc:
        raise SystemExit(f"offline tokenizer load failed: {exc}") from exc

    samples = [
        {"name": "ascii", "text": "Hello CUDA LLM."},
        {"name": "chinese", "text": "你好，世界！"},
        {"name": "mixed", "text": "Hello，世界!  CUDA\tLLM."},
        {"name": "special_eos", "text": "<|im_end|>"},
    ]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for sample in samples:
        ids = tokenizer.encode(sample["text"], add_special_tokens=False)
        records.append({
            "name": sample["name"],
            "text": sample["text"],
            "input_ids": ids,
            "decoded": tokenizer.decode(ids, skip_special_tokens=False),
        })
    payload = {
        "tokenizer_file": str((Path(args.model) / "tokenizer.json").resolve()),
        "tokenizer_config_file": str((Path(args.model) / "tokenizer_config.json").resolve()),
        "model_dir": str(Path(args.model).resolve()),
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "empty_input_ids": tokenizer.encode("", add_special_tokens=False),
        "samples": records,
    }
    (output / "reference.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output / "reference.json")
    print(f"eos_token_id={tokenizer.eos_token_id}")


if __name__ == "__main__":
    main()
