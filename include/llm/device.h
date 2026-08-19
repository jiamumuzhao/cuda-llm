#pragma once
#include <stdexcept>
namespace llm { enum class DeviceType { CPU, CUDA }; inline const char* device_name(DeviceType d) { return d == DeviceType::CPU ? "CPU" : "CUDA"; } }
