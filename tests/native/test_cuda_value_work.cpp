#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>

#include "scf/cuda_value_work.hpp"

namespace {
void require(bool value) {
  if (!value) throw std::runtime_error("CUDA value-provider census contract failed");
}
}  // namespace

int main() {
  using generativeqc::scf::CoulombExchangeWork;
  using generativeqc::scf::observe_cuda_value_dispatch;
  auto work = CoulombExchangeWork::empty();
  observe_cuda_value_dispatch(work, true, true, 0, 1);
  require(work.complete && work.coulomb_applications == 1 && work.exchange_applications == 0);
  observe_cuda_value_dispatch(work, true, true, 1, 1);
  require(work.complete && work.coulomb_applications == 2 && work.exchange_applications == 1);
  observe_cuda_value_dispatch(work, true, false, 1, 1);
  require(work.complete && work.coulomb_applications == 2 && work.exchange_applications == 2);

  auto unrestricted = CoulombExchangeWork::empty();
  observe_cuda_value_dispatch(unrestricted, true, true, 2, 2);
  require(unrestricted.complete && unrestricted.coulomb_applications == 1 &&
          unrestricted.exchange_applications == 4);
  // Rebase/validation applications count even without an SCF proposal.
  observe_cuda_value_dispatch(unrestricted, true, true, 1, 2);
  require(unrestricted.coulomb_applications == 2 && unrestricted.exchange_applications == 6);

  auto failed = work;
  observe_cuda_value_dispatch(failed, false, true, 1, 1);
  observe_cuda_value_dispatch(failed, true, true, 1, 1);
  require(!failed.complete);
  auto retry = CoulombExchangeWork::empty();
  retry.add(work);
  retry.add(unrestricted);
  require(retry.complete && retry.coulomb_applications == 4 && retry.exchange_applications == 8);
  retry.add(failed);
  require(!retry.complete);

  // A new execution resets independently; it cannot revive an old receipt.
  auto next = CoulombExchangeWork::empty();
  observe_cuda_value_dispatch(next, true, true, 1, 1);
  require(next.complete && next.coulomb_applications == 1 && next.exchange_applications == 1);
  require(!failed.complete && !retry.complete);
  for (unsigned spin : {0U, 3U}) {
    auto unknown = CoulombExchangeWork::empty();
    observe_cuda_value_dispatch(unknown, true, true, 1, spin);
    require(!unknown.complete);
  }
  auto unknown = CoulombExchangeWork::empty();
  observe_cuda_value_dispatch(unknown, true, true, 3, 1);
  require(!unknown.complete);
  auto graph = CoulombExchangeWork{};
  observe_cuda_value_dispatch(graph, true, true, 1, 1);
  require(!graph.complete);
  auto overflow = CoulombExchangeWork{true, std::numeric_limits<std::uint64_t>::max(), 0};
  observe_cuda_value_dispatch(overflow, true, true, 1, 1);
  require(!overflow.complete);
  overflow = {true, 0, std::numeric_limits<std::uint64_t>::max()};
  observe_cuda_value_dispatch(overflow, true, false, 1, 2);
  require(!overflow.complete);
}
