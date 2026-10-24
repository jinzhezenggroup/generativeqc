// Compile the complete production adapter against host resource/provider doubles.
#include <cassert>
#include <iostream>

#include "dft/cuda_ks_one_electron_response.cu"
#include "providers.hpp"

int main() {
  using namespace generativeqc;
  methods::PreparedBatch batch;
  dft::CudaKsFinalStateToken token;
  constexpr std::size_t charges = 2 * sizeof(double);
  constexpr auto minimum = integrals::mandatory_bytes + charges;
  for (const auto budget :
       {std::size_t{0}, charges - 1, minimum - 1, minimum, integrals::optional_bytes - 1,
        integrals::optional_bytes, integrals::optional_bytes + charges - 1,
        integrals::optional_bytes + charges}) {
    {
      dft::CudaKsOneElectronResponse owner;
      integrals::CudaOneElectronOptions options;
      options.max_workspace_bytes = budget;
      std::string detail;
      const auto status = owner.prepare(batch, 0, token, options, detail);
      assert((status == GENERATIVEQC_STATUS_SUCCESS) == (budget >= minimum));
      if (status == GENERATIVEQC_STATUS_SUCCESS) {
        assert(owner.work().device_bytes <= budget);
        assert(owner.work().device_bytes == owner.work().integrals.workspace_bytes + charges);
      }
    }
    assert(stub::live_allocations == 0);
  }
  // An unrelated mandatory provider admission failure must remain a failure.
  {
    dft::CudaKsOneElectronResponse owner;
    integrals::CudaOneElectronOptions options;
    options.max_jobs = 1;
    std::string detail;
    assert(owner.prepare(batch, 0, token, options, detail) != GENERATIVEQC_STATUS_SUCCESS);
  }
  assert(stub::live_allocations == 0);
  std::cout << "PASS mandatory charge reservation and bounded optional fallback\n";
}
