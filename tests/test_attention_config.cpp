#include "llm/attention_config.h"

#include <iostream>
#include <stdexcept>

using namespace llm;

static void expect(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

int main() {
  const AttentionConfig gqa{16, 8, 128, 16, 512, true};
  const AttentionConfig mha{8, 8, 64, 16, 2048, true};
  const AttentionConfig invalid_ratio{12, 8, 128, 16, 512, true};
  const AttentionConfig invalid_zero{0, 8, 128, 16, 512, true};

  expect(gqa.is_gqa(), "GQA configuration was rejected");
  expect(mha.is_gqa(), "MHA configuration was rejected");
  expect(!invalid_ratio.is_gqa(), "non-divisible head ratio was accepted");
  expect(!invalid_zero.is_gqa(), "zero-head configuration was accepted");

  std::cout << "attention config MHA/GQA validation passed\n";
  return 0;
}
