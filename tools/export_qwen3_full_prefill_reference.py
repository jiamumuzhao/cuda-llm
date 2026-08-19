#!/usr/bin/env python3
"""Export deterministic offline Qwen3 full-prefill F32 golden fixtures."""

import hashlib
import json
import sys
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-0.6B"
OUT = Path("artifacts/phase25/qwen3_full_prefill")
CASES = {
    "case_seq4": [1, 17, 257, 4097],
    "case_seq8": [1, 17, 257, 4097, 8193, 16385, 32769, 65537],
}


def fail(message):
    print("BLOCKER: " + message, file=sys.stderr)
    raise SystemExit(2)


def write_tensor(torch, case_dir, name, tensor):
    value = tensor.detach().cpu().contiguous()
    if value.is_floating_point():
        value = value.float()
        if not torch.isfinite(value).all().item():
            fail(f"{case_dir.name}/{name} contains NaN or Inf")
    filename = name + (".f32.bin" if value.dtype == torch.float32 else ".i32.bin")
    (case_dir / filename).write_bytes(value.numpy().tobytes(order="C"))
    return {
        "file": filename,
        "rank": value.ndim,
        "shape": [int(d) for d in value.shape],
        "dtype": "f32" if value.dtype == torch.float32 else "i32",
    }


def source_checksum():
    report = Path("artifacts/phase15/qwen3-0.6b-f32/conversion_report.txt")
    if report.exists():
        for line in report.read_text().splitlines():
            if line.startswith("source_sha256 "):
                return line.split(None, 1)[1]
    return ""


def main():
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM
    except Exception as exc:
        fail("offline export requires torch and transformers: " + repr(exc))

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, local_files_only=True, torch_dtype=torch.float32
        ).cpu().float().eval()
    except Exception as exc:
        fail("Qwen/Qwen3-0.6B is not completely available in the local cache; "
             "no network download was attempted: " + repr(exc))

    config = model.config
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None:
        rope_theta = config.rope_parameters["rope_theta"]
    source = {
        "hidden_size": int(config.hidden_size),
        "num_hidden_layers": int(config.num_hidden_layers),
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(config.num_key_value_heads),
        "head_dim": int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)),
        "intermediate_size": int(config.intermediate_size),
        "vocab_size": int(config.vocab_size),
        "rms_norm_eps": float(config.rms_norm_eps),
        "rope_theta": float(rope_theta),
    }
    expected = {
        "hidden_size": 1024, "num_hidden_layers": 28,
        "num_attention_heads": 16, "num_key_value_heads": 8,
        "head_dim": 128, "intermediate_size": 3072,
        "vocab_size": 151936,
    }
    for key, value in expected.items():
        if source[key] != value:
            fail(f"config {key} actual={source[key]} expected={value}")

    original_attn = getattr(config, "_attn_implementation", None)
    config._attn_implementation = "eager"
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_id": MODEL_ID,
        "export_dtype": "f32",
        "source_config": source,
        "source_checksum_sha256": source_checksum(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "cases": {},
    }
    try:
        for case_name, ids in CASES.items():
            case_dir = OUT / case_name
            case_dir.mkdir(parents=True, exist_ok=True)
            input_ids = torch.tensor([ids], dtype=torch.long)
            position_ids = torch.arange(len(ids), dtype=torch.long).unsqueeze(0)
            captured = {}

            def capture(name):
                def hook(_module, _inputs, output):
                    captured[name] = output[0] if isinstance(output, tuple) else output
                return hook

            hooks = [
                model.model.embed_tokens.register_forward_hook(capture("embedding_output")),
                model.model.norm.register_forward_hook(capture("final_hidden")),
            ]
            try:
                with torch.no_grad():
                    output = model(input_ids=input_ids, position_ids=position_ids,
                                   use_cache=False, return_dict=True)
            finally:
                for hook in hooks:
                    hook.remove()
            if "embedding_output" not in captured or "final_hidden" not in captured:
                fail(f"{case_name}: failed to capture embedding/final hidden")
            embedding = captured["embedding_output"][0]
            final_hidden = captured["final_hidden"][0]
            logits = output.logits[0]
            if list(embedding.shape) != [len(ids), 1024]:
                fail(f"{case_name}: embedding shape {list(embedding.shape)}")
            if list(final_hidden.shape) != [len(ids), 1024]:
                fail(f"{case_name}: final_hidden shape {list(final_hidden.shape)}")
            if list(logits.shape) != [len(ids), 151936]:
                fail(f"{case_name}: logits shape {list(logits.shape)}")
            top_values, top_ids = torch.topk(logits[-1].float(), 5, largest=True, sorted=True)
            greedy = int(torch.argmax(logits[-1].float()).item())
            if greedy != int(top_ids[0].item()):
                fail(f"{case_name}: greedy/top1 mismatch")
            tensors = {
                "input_ids": input_ids[0].to(torch.int32),
                "position_ids": position_ids[0].to(torch.int32),
                "embedding_output": embedding,
                "final_hidden": final_hidden,
                "logits": logits,
                "last_token_greedy_id": torch.tensor(greedy, dtype=torch.int32),
                "last_token_top5_ids": top_ids.to(torch.int32),
                "last_token_top5_logits": top_values.float(),
            }
            manifest = {name: write_tensor(torch, case_dir, name, value)
                        for name, value in tensors.items()}
            lines = []
            for name, info in manifest.items():
                lines.append("{} {} {} {} {}".format(
                    name, info["file"], info["rank"],
                    " ".join(str(x) for x in info["shape"]), info["dtype"]))
            (case_dir / "manifest.txt").write_text("\n".join(lines) + "\n")
            (case_dir / "metadata.txt").write_text("\n".join([
                "model_id " + MODEL_ID,
                "token_ids " + " ".join(str(x) for x in ids),
                "position_ids " + " ".join(str(x) for x in range(len(ids))),
                "last_token_greedy_id " + str(greedy),
                "last_token_top5_ids " + " ".join(str(int(x)) for x in top_ids.tolist()),
                "last_token_top5_logits " + " ".join(str(float(x)) for x in top_values.tolist()),
            ]) + "\n")
            metadata["cases"][case_name] = {
                "token_ids": ids,
                "position_ids": list(range(len(ids))),
                "tensors": manifest,
                "last_token_greedy_id": greedy,
                "last_token_top5_ids": [int(x) for x in top_ids.tolist()],
                "last_token_top5_logits": [float(x) for x in top_values.tolist()],
            }
    finally:
        config._attn_implementation = original_attn
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print("exported", ", ".join(CASES), "to", OUT)


if __name__ == "__main__":
    main()
