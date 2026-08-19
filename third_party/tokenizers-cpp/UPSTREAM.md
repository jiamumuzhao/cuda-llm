# tokenizers-cpp provenance

- Project: https://github.com/mlc-ai/tokenizers-cpp
- Vendored upstream commit: `4bb753377680e249345b54c6b10e6d0674c8af03d`
- Upstream release line: `v0.1.0` / Hugging Face `tokenizers` 0.20.x
- License: Apache License 2.0; see [`LICENSE`](LICENSE).
- Rust crates under `vendor/` were captured with Cargo vendor and are built with
  `cargo --offline`. The CMake target does not use FetchContent, ExternalProject,
  git, or any network operation.
- Vendored msgpack and sentencepiece sources retain their upstream licenses in
  their respective directories; this build only compiles the Hugging Face JSON
  binding.
