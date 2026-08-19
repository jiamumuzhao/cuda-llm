#include <cuda_runtime.h>

#include <cstdio>

static bool check(cudaError_t status, const char* expression) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "%s failed: %s\n", expression, cudaGetErrorString(status));
    return false;
}

__global__ void add_one(const float* in, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] + 1.0f;
}

int main() {
    int count = 0;
    if (!check(cudaGetDeviceCount(&count), "cudaGetDeviceCount") || count == 0) return 1;
    cudaDeviceProp prop{};
    if (!check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties")) return 1;
    constexpr int n = 1024;
    float h_in[n]{};
    float h_out[n]{};
    for (int i = 0; i < n; ++i) h_in[i] = static_cast<float>(i);
    float *d_in = nullptr, *d_out = nullptr;
    if (!check(cudaMalloc(&d_in, sizeof(h_in)), "cudaMalloc(d_in)")) return 1;
    if (!check(cudaMalloc(&d_out, sizeof(h_out)), "cudaMalloc(d_out)")) return 1;
    if (!check(cudaMemcpy(d_in, h_in, sizeof(h_in), cudaMemcpyHostToDevice), "cudaMemcpy H2D")) return 1;
    add_one<<<(n + 255) / 256, 256>>>(d_in, d_out, n);
    if (!check(cudaGetLastError(), "add_one launch")) return 1;
    if (!check(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) return 1;
    if (!check(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost), "cudaMemcpy D2H")) return 1;
    if (!check(cudaFree(d_in), "cudaFree(d_in)")) return 1;
    if (!check(cudaFree(d_out), "cudaFree(d_out)")) return 1;
    for (int i = 0; i < n; ++i) {
        if (h_out[i] != h_in[i] + 1.0f) return 2;
    }
    std::printf("CUDA smoke OK: device=%s cc=%d.%d\n", prop.name, prop.major, prop.minor);
    return 0;
}
