#pragma once
#include "tensor.h"
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace llm {
struct PackageTensorInfo { std::string name, dtype; std::vector<int64_t> shape; uint64_t offset=0, byte_size=0, checksum=0; };
struct PackageAlias { std::string name, target; };
class ModelPackage {
 public:
  explicit ModelPackage(std::filesystem::path root);
  const std::filesystem::path& root() const { return root_; }
  std::string config(const std::string& key) const;
  std::vector<PackageTensorInfo> tensors() const;
  std::vector<PackageAlias> aliases() const;
  std::string resolve_name(const std::string& name) const;
  Tensor load_tensor(const std::string& name) const;
 private:
  std::filesystem::path root_; std::map<std::string,std::string> config_; std::map<std::string,PackageTensorInfo> tensors_; std::map<std::string,std::string> aliases_;
  void parse_meta();
};
uint64_t fnv1a64(const uint8_t* data, size_t size);
}
