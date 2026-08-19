#!/usr/bin/env python3
"""Export an offline CPU/F32 Qwen3 layer-0 golden fixture without downloading anything."""
import sys
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-0.6B"
TOKENS = [840, 20772, 3170, 1376]
OUT = Path("artifacts/phase1/qwen3_layer0")

def fail(message):
    print("BLOCKER: " + message, file=sys.stderr)
    raise SystemExit(2)

def main():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import transformers.models.qwen3.modeling_qwen3 as qwen3
    except Exception as exc:
        fail("offline export requires torch and transformers in the selected interpreter: " + repr(exc))
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, local_files_only=True, torch_dtype=torch.float32).cpu().eval()
        AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    except Exception as exc:
        fail("Qwen/Qwen3-0.6B is not completely available in the local Hugging Face cache: " + repr(exc))
    if not hasattr(model, "model") or len(model.model.layers) == 0:
        fail("loaded model has no decoder layers")
    model.config._attn_implementation = "eager"
    layer = model.model.layers[0]
    captured = {}

    def normalize(x):
        x = x.detach().float().cpu().contiguous()
        return x[0].contiguous() if x.ndim > 1 and x.shape[0] == 1 else x

    def save_output(name):
        def hook(_module, _args, output):
            x = output[0] if isinstance(output, tuple) else output
            captured[name] = normalize(x)
        return hook

    hooks = []
    hooks.append(layer.register_forward_pre_hook(
        lambda _m, args, kwargs: captured.__setitem__("hidden_states", normalize(args[0])) if args else None,
        with_kwargs=True))
    hooks.append(layer.register_forward_hook(lambda _m, _args, output: captured.__setitem__("layer_output", normalize(output))))
    modules = {
        "input_norm": layer.input_layernorm,
        "q_linear": layer.self_attn.q_proj, "k_linear": layer.self_attn.k_proj, "v_linear": layer.self_attn.v_proj,
        "q_norm_output": layer.self_attn.q_norm, "k_norm_output": layer.self_attn.k_norm,
        "o_proj_output": layer.self_attn.o_proj,
        "post_attention_norm": layer.post_attention_layernorm,
        "gate_proj_output": layer.mlp.gate_proj, "up_proj_output": layer.mlp.up_proj,
        "down_proj_output": layer.mlp.down_proj,
    }
    for name, module in modules.items():
        hooks.append(module.register_forward_hook(save_output(name)))
    hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(
        lambda _m, args: captured.__setitem__("attention_output", normalize(args[0]))))
    hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(
        lambda _m, args: captured.__setitem__("attention_residual", normalize(args[0]))))
    hooks.append(layer.mlp.down_proj.register_forward_pre_hook(
        lambda _m, args: captured.__setitem__("swiglu_output", normalize(args[0]))))
    original_rope = qwen3.apply_rotary_pos_emb
    rope_calls = [0]
    def capture_rope(*args, **kwargs):
        result = original_rope(*args, **kwargs)
        if rope_calls[0] == 0:
            q, k = result
            captured["q_rope"] = q[0].transpose(0, 1).contiguous()
            captured["k_rope"] = k[0].transpose(0, 1).contiguous()
        rope_calls[0] += 1
        return result
    qwen3.apply_rotary_pos_emb = capture_rope
    input_ids = torch.tensor([TOKENS], dtype=torch.long)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
    finally:
        qwen3.apply_rotary_pos_emb = original_rope
        for hook in hooks:
            hook.remove()
    position_ids = torch.arange(len(TOKENS), dtype=torch.float32)
    captured["position_ids"] = position_ids
    required = ["hidden_states", "input_norm", "q_linear", "k_linear", "v_linear", "q_norm_output", "k_norm_output", "q_rope", "k_rope", "attention_output", "o_proj_output", "attention_residual", "post_attention_norm", "gate_proj_output", "up_proj_output", "swiglu_output", "down_proj_output", "layer_output", "position_ids"]
    missing = [n for n in required if n not in captured]
    if missing: fail("missing captured layer-0 nodes: " + ", ".join(missing))
    state = model.state_dict(); prefix = "model.layers.0."
    weight_keys = {
        "input_layernorm_weight": prefix+"input_layernorm.weight", "q_proj_weight": prefix+"self_attn.q_proj.weight", "k_proj_weight": prefix+"self_attn.k_proj.weight", "v_proj_weight": prefix+"self_attn.v_proj.weight", "q_norm_weight": prefix+"self_attn.q_norm.weight", "k_norm_weight": prefix+"self_attn.k_norm.weight", "o_proj_weight": prefix+"self_attn.o_proj.weight", "post_attention_layernorm_weight": prefix+"post_attention_layernorm.weight", "gate_proj_weight": prefix+"mlp.gate_proj.weight", "up_proj_weight": prefix+"mlp.up_proj.weight", "down_proj_weight": prefix+"mlp.down_proj.weight"}
    tensors = {n: state[k].detach().float().cpu().contiguous() for n,k in weight_keys.items()}
    tensors.update(captured)
    OUT.mkdir(parents=True, exist_ok=True)
    lines = []
    for name, tensor in tensors.items():
        tensor = tensor.contiguous().float(); filename = name + ".f32"
        (OUT/filename).write_bytes(tensor.numpy().tobytes(order="C"))
        lines.append("{} {} {} {}".format(name, filename, tensor.ndim, " ".join(str(int(d)) for d in tensor.shape)))
    (OUT/"manifest.txt").write_text("\n".join(lines) + "\n")
    config = model.config
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None:
        rope_theta = config.rope_parameters["rope_theta"]
    (OUT/"config.txt").write_text("\n".join([
        "model_id Qwen/Qwen3-0.6B", "layer_index 0", "token_ids " + " ".join(map(str, TOKENS)),
        f"hidden_size {config.hidden_size}", f"q_heads {config.num_attention_heads}",
        f"kv_heads {getattr(config, 'num_key_value_heads', config.num_attention_heads)}",
        f"head_dim {getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)}",
        f"rms_norm_eps {config.rms_norm_eps}", f"rope_theta {rope_theta}", "dtype float32", ""
    ]))
    print("exported", len(tensors), "tensors to", OUT)

if __name__ == "__main__":
    main()
