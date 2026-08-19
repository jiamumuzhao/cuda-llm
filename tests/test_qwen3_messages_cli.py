#!/usr/bin/env python3
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "artifacts/phase15/qwen3-0.6b-f32")
TOKENIZER = "/root/huggingface/Qwen3-0.6B/tokenizer.json"
MESSAGES = os.path.join(ROOT, "tests/fixtures/phase29/multiturn.json")
REFERENCE = os.path.join(ROOT, "artifacts/phase29/qwen3_chat_template_reference/reference.json")

def parse(output, include_text=True):
    result = {}
    markers = ["prompt_ids=", "generated_ids="]
    if include_text: markers.append("generated_text=")
    markers += ["stop_reason=", "final_cache_length="]
    for index, marker in enumerate(markers):
        start = output.find(marker)
        if start < 0: raise AssertionError(f"missing {marker}")
        start += len(marker); end = len(output)
        for next_marker in markers[index + 1:]:
            position = output.find("\n" + next_marker, start)
            if position >= 0: end = min(end, position)
        result[marker[:-1]] = output[start:end].rstrip("\n")
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

def run_chat(extra=()):
    return run_checked([os.path.join(ROOT, "build/cuda_llm_chat"), "--model", MODEL,
        "--tokenizer", TOKENIZER, "--messages-file", MESSAGES, "--max-new-tokens", "3",
        "--max-seq-len", "32", *extra])

def run_generate(prompt_ids, extra=()):
    return run_checked([os.path.join(ROOT, "build/cuda_llm_generate"), "--model", MODEL,
        "--ids", ",".join(str(x) for x in prompt_ids), "--max-new-tokens", "3",
        "--max-seq-len", "32", *extra])

def main():
    reference = json.load(open(REFERENCE, encoding="utf-8"))
    expected = next(case for case in reference["cases"] if case["name"] == "multiturn")
    greedy = parse(run_chat().stdout)
    if ids(greedy["prompt_ids"]) != expected["apply_chat_template_ids"]:
        raise AssertionError("messages prompt IDs differ from chat-template reference")
    id_greedy = parse(run_generate(ids(greedy["prompt_ids"])).stdout, include_text=False)
    for key in ("generated_ids", "stop_reason", "final_cache_length"):
        if greedy[key] != id_greedy[key]: raise AssertionError(f"greedy messages/ID mismatch for {key}")
    args = ("--temperature", "0.8", "--top-k", "50", "--top-p", "0.9", "--seed", "20260727")
    sampled = parse(run_chat(args).stdout)
    if sampled != parse(run_chat(args).stdout): raise AssertionError("sampled messages CLI is not deterministic")
    id_sampled = parse(run_generate(ids(sampled["prompt_ids"]), args).stdout, include_text=False)
    for key in ("generated_ids", "stop_reason", "final_cache_length"):
        if sampled[key] != id_sampled[key]: raise AssertionError(f"sampled messages/ID mismatch for {key}")
    print("test_qwen3_messages_cli passed")

if __name__ == "__main__": main()
