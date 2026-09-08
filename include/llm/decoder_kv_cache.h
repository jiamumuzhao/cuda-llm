#pragma once

namespace llm {

class DecoderKvCache {
 public:
  virtual ~DecoderKvCache() = default;
};

class DecoderPagedKvCache {
 public:
  virtual ~DecoderPagedKvCache() = default;
};

}  // namespace llm
