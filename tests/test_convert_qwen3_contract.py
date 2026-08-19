#!/usr/bin/env python3
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("converter", ROOT / "tools" / "convert_qwen3_hf.py")
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)

# Independent contract: do not derive test expectations from converter helpers.
EXPECTED = {
    "token_embedding": [151936, 1024],
    "final_norm": [1024],
}
PER_LAYER = {
    "input_norm": [1024],
    "q_proj": [2048, 1024],
    "k_proj": [1024, 1024],
    "v_proj": [1024, 1024],
    "q_norm": [128],
    "k_norm": [128],
    "o_proj": [1024, 2048],
    "post_attention_norm": [1024],
    "gate_proj": [3072, 1024],
    "up_proj": [3072, 1024],
    "down_proj": [1024, 3072],
}
for layer in range(28):
    for name, shape in PER_LAYER.items():
        EXPECTED[f"layers.{layer}.{name}"] = shape[:]

actual = converter.expected_shapes()
assert len(EXPECTED) == 310, len(EXPECTED)
assert set(actual) == set(EXPECTED), "converter logical-name set differs from independent contract"
assert len(actual) == 310, len(actual)
for name, expected in EXPECTED.items():
    assert actual[name] == expected, f"{name}: actual={actual[name]} expected={expected}"

for logical in ("layers.0.q_proj", "layers.0.k_proj", "layers.0.o_proj", "layers.0.down_proj"):
    expected = EXPECTED[logical]
    source = "model.layers.0." + logical.rsplit(".", 1)[1] + ".weight"
    bad_shape = expected[:-1] + [expected[-1] + 1]
    try:
        converter.validate_shape(logical, source, bad_shape)
    except ValueError as exc:
        message = str(exc)
        for required in ("logical tensor", "Hugging Face source tensor", "actual shape", "expected shape"):
            assert required in message, message
        assert logical in message and source in message, message
    else:
        raise AssertionError(f"bad shape accepted for {logical}")

print("test_convert_qwen3_contract passed: independent 310 logical tensor shapes")
