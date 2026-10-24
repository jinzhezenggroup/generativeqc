// Standalone device-graph synccheck diagnostic; no GenerativeQC code is linked.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

static void checked(cudaError_t code) {
  if (code != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(code));
    std::exit(2);
  }
}
__global__ void prepare(unsigned* value) {
  if (threadIdx.x == 0) *value = 7;
}
__global__ void consume(const unsigned* value, unsigned* output) {
  // Every thread in every block unconditionally reaches this barrier.
  __syncthreads();
  output[blockIdx.x * blockDim.x + threadIdx.x] = *value;
}
__global__ void invalid(unsigned* output) {
  // Deliberate negative control: only half of the first warp participates.
  if (threadIdx.x < 16) {
    __syncthreads();
    output[threadIdx.x] = 9;
  }
  output[32 + threadIdx.x] = 7;
}
int main(int argc, char** argv) {
  const int mode = argc > 1 ? std::atoi(argv[1]) : 0;
  unsigned *value, *output;
  checked(cudaMalloc(&value, sizeof(unsigned)));
  checked(cudaMalloc(&output, 8 * 352 * sizeof(unsigned)));
  checked(cudaMemset(value, 0, sizeof(unsigned)));
  checked(cudaMemset(output, 0, 8 * 352 * sizeof(unsigned)));
  cudaStream_t stream;
  checked(cudaStreamCreate(&stream));
  cudaGraph_t graph;
  cudaGraphExec_t executable;
  if (mode != 0) checked(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  prepare<<<1, 32, 0, stream>>>(value);
  if (mode == 3)
    invalid<<<1, 32, 0, stream>>>(output);
  else
    consume<<<8, 352, 0, stream>>>(value, output);
  checked(cudaGetLastError());
  if (mode != 0) {
    checked(cudaStreamEndCapture(stream, &graph));
    checked(cudaGraphInstantiate(&executable, graph,
                                 mode == 2 ? cudaGraphInstantiateFlagDeviceLaunch : 0));
    checked(cudaGraphUpload(executable, stream));
    checked(cudaStreamSynchronize(stream));
    checked(cudaGraphLaunch(executable, stream));
  }
  checked(cudaStreamSynchronize(stream));
  std::vector<unsigned> host(8 * 352);
  checked(cudaMemcpy(host.data(), output, host.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
  if (mode != 3)
    for (auto v : host)
      if (v != 7) return 3;
  if (mode != 0) {
    checked(cudaGraphExecDestroy(executable));
    checked(cudaGraphDestroy(graph));
  }
  checked(cudaStreamDestroy(stream));
  checked(cudaFree(output));
  checked(cudaFree(value));
  std::printf("mode=%d verified %zu outputs\n", mode, host.size());
}
