#!/usr/bin/env python3
import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "artifacts/phase15/qwen3-0.6b-f32")
TOKENIZER = "/root/huggingface/Qwen3-0.6B/tokenizer.json"


def parse(output):
    result = {}
    for line in output.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def ids(value):
    body = value.strip()[1:-1]
    return [] if not body else [int(x) for x in body.split(",")]


def run_checked(command):
    try:
        return subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                              check=True, timeout=180)
    except subprocess.TimeoutExpired as error:
        print("TIMEOUT after 180s: " + " ".join(command), file=sys.stderr)
        print("stdout:\n" + (error.stdout or ""), file=sys.stderr)
        print("stderr:\n" + (error.stderr or ""), file=sys.stderr)
        try:
            print("GPU state:\n" + subprocess.run(["nvidia-smi"], text=True,
                  capture_output=True, timeout=10).stdout, file=sys.stderr)
        except Exception as gpu_error:
            print("GPU state unavailable: " + str(gpu_error), file=sys.stderr)
        raise
    except subprocess.CalledProcessError as error:
        print("COMMAND FAILED: " + " ".join(command), file=sys.stderr)
        print("stdout:\n" + (error.stdout or ""), file=sys.stderr)
        print("stderr:\n" + (error.stderr or ""), file=sys.stderr)
        raise


def main():
    prompt = "你好，CUDA LLM！"
    chat = run_checked([
        os.path.join(ROOT, "build/cuda_llm_chat"), "--model", MODEL,
        "--tokenizer", TOKENIZER, "--prompt", prompt,
        "--max-new-tokens", "3", "--max-seq-len", "32",
    ])
    c = parse(chat.stdout)
    prompt_ids = ids(c["prompt_ids"])
    generate = run_checked([
        os.path.join(ROOT, "build/cuda_llm_generate"), "--model", MODEL,
        "--ids", ",".join(str(x) for x in prompt_ids),
        "--max-new-tokens", "3", "--max-seq-len", "32",
    ])
    g = parse(generate.stdout)
    for key in ("generated_ids", "stop_reason", "final_cache_length"):
        if c[key] != g[key]:
            raise AssertionError(f"CLI mismatch for {key}: chat={c[key]!r} generate={g[key]!r}")
    from tokenizers import Tokenizer
    decoded = Tokenizer.from_file(TOKENIZER).decode(ids(c["generated_ids"]), skip_special_tokens=False)
    if c["generated_text"] != decoded:
        raise AssertionError(f"generated_text mismatch: {c['generated_text']!r} != {decoded!r}")
    generate_zero = run_checked([
        os.path.join(ROOT, "build/cuda_llm_generate"), "--model", MODEL,
        "--ids", ",".join(str(x) for x in prompt_ids), "--max-new-tokens", "3",
        "--max-seq-len", "32", "--temperature", "0", "--top-k", "1",
        "--top-p", "1", "--seed", "99",
    ])
    if parse(generate_zero.stdout) != g:
        raise AssertionError("temperature=0 CLI behavior differs from default greedy")
    sampled_args = ["--temperature", "0.8", "--top-k", "50", "--top-p", "0.9", "--seed", "20260727"]
    chat_sampled = run_checked([
        os.path.join(ROOT, "build/cuda_llm_chat"), "--model", MODEL,
        "--tokenizer", TOKENIZER, "--prompt", prompt,
        "--max-new-tokens", "3", "--max-seq-len", "32", *sampled_args,
    ])
    chat_sampled_repeat = run_checked([
        os.path.join(ROOT, "build/cuda_llm_chat"), "--model", MODEL,
        "--tokenizer", TOKENIZER, "--prompt", prompt,
        "--max-new-tokens", "3", "--max-seq-len", "32", *sampled_args,
    ])
    if chat_sampled.stdout != chat_sampled_repeat.stdout:
        raise AssertionError("sampled chat CLI is not deterministic")
    cs = parse(chat_sampled.stdout)
    generate_sampled = run_checked([
        os.path.join(ROOT, "build/cuda_llm_generate"), "--model", MODEL,
        "--ids", ",".join(str(x) for x in ids(cs["prompt_ids"])),
        "--max-new-tokens", "3", "--max-seq-len", "32", *sampled_args,
    ])
    gs = parse(generate_sampled.stdout)
    for key in ("generated_ids", "stop_reason", "final_cache_length"):
        if cs[key] != gs[key]:
            raise AssertionError(f"sampled CLI mismatch for {key}: chat={cs[key]!r} generate={gs[key]!r}")
    print("test_qwen3_chat_cli passed")


if __name__ == "__main__":
    main()
