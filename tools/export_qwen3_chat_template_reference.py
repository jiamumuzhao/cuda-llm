#!/usr/bin/env python3
"""Offline Transformers reference for the fixed Qwen3 text chat template."""
import argparse
import json
from pathlib import Path


CASES = [
    {"name": "single_user", "messages": [{"role": "user", "content": "Hello CUDA LLM!"}]},
    {"name": "system_user", "messages": [
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": "你好，世界！"},
    ]},
    {"name": "multiturn", "messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "OK"},
        {"role": "user", "content": "你好？"},
    ]},
    {"name": "empty_special", "messages": [
        {"role": "system", "content": ""},
        {"role": "user", "content": "line1\nline2 <|im_end|> !"},
    ]},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        from transformers import AutoTokenizer, __version__ as transformers_version
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    except Exception as exc:
        raise SystemExit(f"offline tokenizer load failed: {exc}") from exc

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for case in CASES:
        rendered = tokenizer.apply_chat_template(
            case["messages"], tokenize=False, add_generation_prompt=True)
        templated = tokenizer.apply_chat_template(
            case["messages"], tokenize=True, add_generation_prompt=True)
        templated_ids = templated["input_ids"] if hasattr(templated, "keys") else templated
        encoded_ids = tokenizer.encode(rendered, add_special_tokens=False)
        if templated_ids != encoded_ids:
            raise SystemExit(f"template/tokenize mismatch for {case['name']}")
        records.append({**case, "rendered_prompt": rendered,
                        "apply_chat_template_ids": templated_ids,
                        "rendered_encode_ids": encoded_ids,
                        "token_count": len(encoded_ids)})
        if len(encoded_ids) > 32:
            raise SystemExit(f"{case['name']} exceeds seq_len 32: {len(encoded_ids)}")

    model_dir = Path(args.model).resolve()
    config = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    payload = {
        "model_dir": str(model_dir),
        "tokenizer_file": str((model_dir / "tokenizer.json").resolve()),
        "tokenizer_config_file": str((model_dir / "tokenizer_config.json").resolve()),
        "transformers_version": transformers_version,
        "chat_template": config.get("chat_template", tokenizer.chat_template),
        "add_generation_prompt": True,
        "cases": records,
    }
    path = output / "reference.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
