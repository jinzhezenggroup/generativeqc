/** Opt-in linked-core fixture. No molecular input, capture or timing claim.
 * Build with assertions enabled and link the exact reviewed native core. */
#if defined(NDEBUG)
#error "VV10 history qualification requires assertions"
#endif
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "dft/cuda_ks_kernels.hpp"
#include "scf/cuda/matrix_library.hpp"
#include "scf/cuda/scf_diis_kernels.hpp"
#include "solver/diis_coefficients.hpp"
#include "vv10_rebase_cases.hpp"

namespace p = generativeqc::dft::vv10_rebase;
namespace k = generativeqc::dft::cuda_ks_detail;
namespace scf = generativeqc::scf::cuda_execution;
namespace c = generativeqc::tests::vv10_rebase;
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void status(generativeqc_status value) {
  if (value != GENERATIVEQC_STATUS_SUCCESS) throw std::runtime_error("matrix product failed");
}
template <class T>
struct Device {
  T* pointer{};
  explicit Device(std::size_t count) { check(cudaMalloc(&pointer, count * sizeof(T))); }
  ~Device() {
    if (pointer) cudaFree(pointer);
  }
  Device(const Device&) = delete;
};
template <class T>
void upload(cudaStream_t stream, T* destination, const T* source, std::size_t count) {
  check(cudaMemcpyAsync(destination, source, count * sizeof(T), cudaMemcpyHostToDevice, stream));
  // Fixture host sources may be temporary. Production uses its existing owners.
  check(cudaStreamSynchronize(stream));
}
template <class T>
std::vector<T> read(cudaStream_t stream, const T* source, std::size_t count) {
  std::vector<T> output(count);
  check(cudaMemcpyAsync(output.data(), source, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
  check(cudaStreamSynchronize(stream));
  return output;
}
void matrix(const double* actual, const c::Matrix& expected, const char* name) {
  c::Matrix value{};
  std::copy_n(actual, c::m, value.begin());
  c::close(value, expected);
  std::printf("VV10_REBASE_MATRIX_PASS %s\n", name);
}
void run(cudaStream_t stream, bool zero, bool seed, int refusal) {
  constexpr unsigned H = c::history;
  constexpr auto M = c::m;
  auto input = c::fixture(zero);
  if (seed) input.old_potential.fill(0.0);
  const auto layout = *p::layout(c::n, H);
  Device<unsigned char> lease(layout.total_bytes);
  auto* bytes = lease.pointer;
  k::RebaseStorage s{reinterpret_cast<double*>(bytes + layout.density),
                     reinterpret_cast<double*>(bytes + layout.bank_density),
                     reinterpret_cast<double*>(bytes + layout.bank_fock),
                     reinterpret_cast<double*>(bytes + layout.bank_residual),
                     reinterpret_cast<double*>(bytes + layout.old_potential),
                     reinterpret_cast<p::SlotTag*>(bytes + layout.density_tags),
                     reinterpret_cast<p::SlotTag*>(bytes + layout.bank_tags),
                     reinterpret_cast<p::Transaction*>(bytes + layout.transaction)};
  Device<double> liveD(M), liveF(M), liveR(M), fh(H * M), rh(H * M), gram((H + 1) * (H + 1)),
      raw(H * H), coefficients(H + 1), effective(M), va(M), vb(M), overlap(M), tmp1(M), tmp2(M);
  Device<std::uint32_t> count(1), head(1);
  Device<std::uint8_t> enabled(1);
  const std::uint8_t one = 1;
  const std::uint32_t none = 0;
  upload(stream, enabled.pointer, &one, 1);
  upload(stream, count.pointer, &none, 1);
  upload(stream, head.pointer, &none, 1);
  upload(stream, va.pointer, input.old_potential.data(), M);
  upload(stream, vb.pointer, input.new_potential.data(), M);
  upload(stream, overlap.pointer, input.overlap.data(), M);
  k::rebase_reset(stream, H, s);
  check(cudaGetLastError());
  scf::MatrixLibraryOwner products;
  status(products.prepare(stream, c::n, 1, 1));
  const p::Key old{11, 17, seed ? 0U : 31U, 1}, target{11, 17, 43, 1};
  std::array<c::Matrix, H> physicalD{}, physicalF{}, physicalR{};
  auto insert = [&](const c::Matrix& d, const c::Matrix& f, const c::Matrix& r, p::Key key) {
    const auto slot = read(stream, head.pointer, 1)[0];
    assert(slot < H);
    physicalD[slot] = d;
    physicalF[slot] = f;
    physicalR[slot] = r;
    upload(stream, liveD.pointer, d.data(), M);
    upload(stream, liveF.pointer, f.data(), M);
    upload(stream, liveR.pointer, r.data(), M);
    k::rebase_record_density(stream, M, H, liveD.pointer, enabled.pointer, count.pointer,
                             head.pointer, key, s);
    check(cudaGetLastError());
    scf::launch_update_diis_cached_gram(1, 32, 0, stream, 1, c::n, 1, H, liveF.pointer,
                                        liveR.pointer, enabled.pointer, fh.pointer, rh.pointer,
                                        gram.pointer, coefficients.pointer, count.pointer,
                                        head.pointer, effective.pointer, raw.pointer, true);
    check(cudaGetLastError());
    check(cudaStreamSynchronize(stream));
  };
  // Six insertions exercise actual wrap and the incumbent normalized retirement.
  for (unsigned i = 0; i < 6; ++i) {
    const auto slot = i % H;
    insert(input.density[slot], input.fock[slot],
           c::residual(input.fock[slot], input.density[slot], input.overlap), old);
  }
  const auto oldCount = read(stream, count.pointer, 1)[0],
             oldHead = read(stream, head.pointer, 1)[0];
  assert(p::valid_window(H, oldCount, oldHead) && oldCount);
  k::rebase_snapshot(stream, M, H, fh.pointer, rh.pointer, seed ? nullptr : va.pointer,
                     count.pointer, head.pointer, old, s);
  check(cudaGetLastError());
  auto tx = read(stream, s.transaction, 1)[0];
  assert(tx.ready && tx.bank_count);
  std::vector<c::Matrix> bankD, bankF, bankR;
  for (unsigned i = 0; i < tx.bank_count; ++i) {
    auto slot = p::bank_source_slot(H, oldCount, oldHead, i);
    bankD.push_back(physicalD[slot]);
    bankF.push_back(physicalF[slot]);
    bankR.push_back(physicalR[slot]);
  }
  // Original target reset/insertion/proposal, before any rebase work.
  upload(stream, count.pointer, &none, 1);
  upload(stream, head.pointer, &none, 1);
  const auto targetD = input.density[3], targetF = c::shifted_fock(input, 3);
  const auto targetR = c::residual(targetF, targetD, input.overlap);
  insert(targetD, targetF, targetR, target);
  assert(read(stream, count.pointer, 1)[0] == 1 && read(stream, head.pointer, 1)[0] == 1);
  auto newKey = target;
  if (refusal == 1) ++newKey.solve;
  if (refusal == 2) {
    auto bad = input.new_potential;
    bad[0] = std::numeric_limits<double>::infinity();
    upload(stream, vb.pointer, bad.data(), M);
  }
  k::rebase_prepare_target(stream, M, H, vb.pointer, count.pointer, head.pointer, newKey, s);
  check(cudaGetLastError());
  tx = read(stream, s.transaction, 1)[0];
  if (!refusal) assert(tx.ready);
  auto multiply = [&](const double* a, bool aSpin, const double* b, bool bSpin, double* out) {
    status(scf::launch_spin_matrix_product(products.view(), 1, 1, c::n, a, aSpin, false, b, bSpin,
                                           nullptr, out));
  };
  if (tx.ready)
    for (unsigned bank = 0; bank < tx.bank_count; ++bank) {
      const auto* d = s.bank_density + bank * M;
      multiply(s.potential, true, d, true, tmp1.pointer);
      k::rebase_check_finite(stream, M, tmp1.pointer, s);
      check(cudaGetLastError());
      multiply(tmp1.pointer, true, overlap.pointer, false, tmp2.pointer);
      if (refusal == 3 && bank == 0) {
        auto bad = targetR;
        bad[0] = std::numeric_limits<double>::infinity();
        upload(stream, tmp2.pointer, bad.data(), M);
      }
      k::rebase_accumulate(stream, M, bank, tmp2.pointer, false, true, s);
      check(cudaGetLastError());
      multiply(overlap.pointer, false, d, true, tmp1.pointer);
      k::rebase_check_finite(stream, M, tmp1.pointer, s);
      check(cudaGetLastError());
      multiply(tmp1.pointer, true, s.potential, true, tmp2.pointer);
      k::rebase_accumulate(stream, M, bank, tmp2.pointer, true, false, s);
      check(cudaGetLastError());
    }
  k::rebase_restore(stream, M, H, fh.pointer, rh.pointer, s);
  check(cudaGetLastError());
  tx = read(stream, s.transaction, 1)[0];
  if (tx.ready && refusal == 4) {
    auto bad = targetR;
    bad[0] = std::numeric_limits<double>::infinity();
    upload(stream, rh.pointer + (H - tx.bank_count) * M, bad.data(), M);
  }
  // Supply the proposed window only if target admission formed one. Even a
  // refused transaction may call publication, which cannot activate it.
  if (tx.proposed_count >= 2)
    scf::launch_rebuild_ordered_diis_gram(stream, M, H, tx.proposed_count, 1, rh.pointer,
                                          raw.pointer, &s.transaction->ready);
  check(cudaGetLastError());
  k::rebase_publish(stream, H, count.pointer, head.pointer, s);
  check(cudaGetLastError());
  const auto finalCount = read(stream, count.pointer, 1)[0],
             finalHead = read(stream, head.pointer, 1)[0];
  tx = read(stream, s.transaction, 1)[0];
  const auto allD = read(stream, s.density, H * M), allF = read(stream, fh.pointer, H * M),
             allR = read(stream, rh.pointer, H * M);
  matrix(allD.data(), targetD, "target-D");
  matrix(allF.data(), targetF, "target-F");
  matrix(allR.data(), targetR, "target-R");
  assert(finalHead == 1);
  if (refusal) {
    assert(finalCount == 1 && !tx.work.successful_rebases && tx.work.refusals);
    std::printf("VV10_REBASE_REFUSAL_PASS reason=%d error=%u\n", refusal,
                static_cast<unsigned>(tx.error));
  } else {
    assert(finalCount == bankD.size() + 1 && tx.work.successful_rebases == 1 &&
           tx.work.entries_transferred == bankD.size());
    for (unsigned bank = 0; bank < bankD.size(); ++bank) {
      const auto slot = H - bankD.size() + bank;
      c::Matrix expectedF{};
      for (std::size_t i = 0; i < M; ++i)
        expectedF[i] = bankF[bank][i] + (input.new_potential[i] - input.old_potential[i]);
      matrix(allD.data() + slot * M, bankD[bank], "bank-D");
      matrix(allF.data() + slot * M, expectedF, "bank-F");
      matrix(allR.data() + slot * M, c::residual(expectedF, bankD[bank], input.overlap), "bank-R");
      physicalD[slot] = bankD[bank];
      physicalF[slot] = expectedF;
      physicalR[slot] = c::residual(expectedF, bankD[bank], input.overlap);
    }
  }
  if (!refusal) {
    const auto cache = read(stream, raw.pointer, H * H);
    for (unsigned i = 0; i < finalCount; ++i)
      for (unsigned j = 0; j < finalCount; ++j) {
        const auto a = p::chronological_slot(H, finalCount, finalHead, i),
                   b = p::chronological_slot(H, finalCount, finalHead, j);
        c::Matrix ra{}, rb{};
        std::copy_n(allR.data() + a * M, M, ra.begin());
        std::copy_n(allR.data() + b * M, M, rb.begin());
        const auto expected = c::dot(ra, rb);
        assert(std::isfinite(cache[a * H + b]) &&
               std::abs(cache[a * H + b] - expected) <= 2e-11 * (1 + std::abs(expected)));
      }
    std::printf("VV10_REBASE_FULL_GRAM_PASS count=%u head=%u\n", finalCount, finalHead);
  }
  // Ordinary insertion after success or refusal must refresh every live cross
  // term, including the target diagonal after a count1 refusal.
  insert(input.density[2], c::shifted_fock(input, 2),
         c::residual(c::shifted_fock(input, 2), input.density[2], input.overlap), target);
  const auto nextCount = read(stream, count.pointer, 1)[0],
             nextHead = read(stream, head.pointer, 1)[0];
  const auto cache = read(stream, raw.pointer, H * H);
  if (nextCount > 1)
    for (unsigned i = 0; i < nextCount; ++i)
      for (unsigned j = 0; j < nextCount; ++j) {
        const auto a = p::chronological_slot(H, nextCount, nextHead, i),
                   b = p::chronological_slot(H, nextCount, nextHead, j);
        const auto expected = c::dot(physicalR[a], physicalR[b]);
        assert(std::isfinite(cache[a * H + b]) &&
               std::abs(cache[a * H + b] - expected) <= 2e-11 * (1 + std::abs(expected)));
      }
  std::printf("VV10_REBASE_GRAM_PASS zero=%d seed=%d refusal=%d count=%u head=%u\n", zero, seed,
              refusal, nextCount, nextHead);
  // New solve epoch revokes all old tags; a stale snapshot must refuse.
  auto stale = target;
  ++stale.solve;
  k::rebase_snapshot(stream, M, H, fh.pointer, rh.pointer, vb.pointer, count.pointer, head.pointer,
                     stale, s);
  check(cudaGetLastError());
  tx = read(stream, s.transaction, 1)[0];
  assert(!tx.ready);
  check(cudaStreamSynchronize(stream));
  products.reset();
}
int main() {
  cudaStream_t stream{};
  check(cudaStreamCreate(&stream));
  try {
    run(stream, false, false, 0);
    run(stream, true, false, 0);
    run(stream, false, true, 0);
    for (int refusal = 1; refusal <= 4; ++refusal) run(stream, false, false, refusal);
    check(cudaStreamSynchronize(stream));
    check(cudaStreamDestroy(stream));
    std::puts("VV10_REBASE_LINKED_HELPERS_PASS cases=7 shutdown=0");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "VV10_REBASE_LINKED_HELPERS_FAILED %s\n", error.what());
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    return 1;
  }
}
