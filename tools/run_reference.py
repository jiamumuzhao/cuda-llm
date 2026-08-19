#!/usr/bin/env python3
"""Generate a deterministic Qwen reference/golden record for later kernel tests."""
from __future__ import annotations
import argparse
import json
import platform
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--prompt", default="Explain why key-value caching improves autoregressive transformer decoding.")
    p.add_argument("--out-dir", default="artifacts/phase0")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" and torch.cuda.is_available() else "cpu")
    if args.dtype == "auto":
        dtype = torch.float16 if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] < 8 else torch.bfloat16
    else:
        dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, trust_remote_code=True)
    model.eval().to(device)
    encoded = tokenizer(args.prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device); attention_mask = encoded["attention_mask"].to(device)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    with torch.inference_mode():
        t0 = time.perf_counter(); first = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        if device.type == "cuda": torch.cuda.synchronize(device)
        prefill_s = time.perf_counter() - t0
        t1 = time.perf_counter(); generated = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id)
        if device.type == "cuda": torch.cuda.synchronize(device)
        generate_s = time.perf_counter() - t1
    prompt_len = int(input_ids.shape[-1]); generated_ids = generated[0].detach().cpu().tolist()
    next_id = int(first.logits[0, -1].argmax().item())
    record = {
        "model": args.model, "prompt": args.prompt, "seed": args.seed, "device": str(device), "dtype": str(dtype), "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu", "platform": platform.platform(), "input_ids": input_ids[0].detach().cpu().tolist(), "prompt_length": prompt_len,
        "next_token_id": next_id, "next_token": tokenizer.decode([next_id]), "generated_ids": generated_ids, "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=False),
        "prefill_seconds": prefill_s, "generation_seconds": generate_s, "generated_tokens": max(0, len(generated_ids) - prompt_len), "prefill_tokens_per_second": prompt_len / prefill_s,
        "generation_tokens_per_second": max(0, len(generated_ids) - prompt_len) / generate_s, "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0, "model_config": model.config.to_dict(),
    }
    (out / "reference.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    torch.save({"input_ids": input_ids.cpu(), "logits_last": first.logits[0, -1].float().cpu(), "generated_ids": generated.cpu()}, out / "golden.pt")
    print(json.dumps({k: record[k] for k in ["model", "dtype", "gpu", "prompt_length", "next_token_id", "next_token", "prefill_tokens_per_second", "generation_tokens_per_second", "peak_memory_bytes"]}, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
