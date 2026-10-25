// Device arithmetic qualification for the compiler-owned Hermite norm bound.
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include "generated_df_md_j.cuh"
using generativeqc::scf::cuda_execution::df_md_hermite_norm;
__global__ void evaluate(double exponent, double* output) {
  if (threadIdx.x || blockIdx.x) return;
  unsigned offset = 0;
  for (unsigned angular = 0; angular <= 6; ++angular)
    for (unsigned index = 0; index < (angular + 1) * (angular + 2) * (angular + 3) / 6; ++index)
      output[offset++] = df_md_hermite_norm(exponent, angular, index);
}
int main(int argc, char** argv) {
  if (argc != 2 || !std::getenv("SLURM_JOB_ID")) return 3;
  double* output{};
  if (cudaMalloc(&output, 210 * sizeof(double)) != cudaSuccess) return 2;
  evaluate<<<1, 1>>>(std::strtod(argv[1], nullptr), output);
  std::vector<double> result(210);
  const auto status =
      cudaMemcpy(result.data(), output, result.size() * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(output);
  if (status != cudaSuccess) return 2;
  std::cout << std::setprecision(17);
  for (double value : result) std::cout << value << '\n';
}
