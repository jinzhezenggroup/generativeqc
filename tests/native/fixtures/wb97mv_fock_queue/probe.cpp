#include <iostream>

#include "physical_fock_host.cpp"

namespace {
void expect(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void run(unsigned spins, bool fitted) {
  using namespace generativeqc;
  queue_test::spins = spins;
  queue_test::fitted = fitted;
  queue_test::revoked = queue_test::capturing = false;
  std::fill(queue_test::metadata.begin(), queue_test::metadata.end(), 0);
  auto& primary = scf::primary.model;
  primary = {};
  primary.spec.spin = spins == 1 ? scf::FockSpin::Restricted : scf::FockSpin::Unrestricted;
  primary.spec.exchange.coefficient = -0.15 / (spins == 1 ? 2 : 1);
  primary.spec.coulomb.approximation = primary.spec.exchange.approximation =
      fitted ? scf::FockApproximation::DensityFitted : scf::FockApproximation::Exact;
  scf::correction = {};
  scf::correction.spec.spin = primary.spec.spin;
  scf::correction.spec.coulomb.present = false;
  scf::correction.spec.exchange.op = scf::FockOperator::LongRange;
  scf::correction.spec.exchange.coefficient = -0.85 / (spins == 1 ? 2 : 1);
  scf::correction.spec.exchange.omega = 0.3;
  dft::CudaKsFinalStateToken token;
  token.identity.determinant.model = primary;
  token.identity.model.spins = spins;
  token.identity.model.range_correction = scf::correction;
  methods::PreparedBatch batch;
  dft::CudaKsWb97mvResponse owner;
  std::string detail;
  queue_test::reset_action();
  queue_test::valid_fock_binding = false;
  expect(owner.prepare(batch, 0, token, {}, detail) != GENERATIVEQC_STATUS_SUCCESS,
         "physical model accepted a binding without an owner");
  queue_test::valid_fock_binding = true;
  expect(owner.prepare(batch, 0, token, {}, detail) == GENERATIVEQC_STATUS_SUCCESS, detail.c_str());
  std::vector<double> direction(4 * spins, 1), output(4 * spins, 27);

  // Admission-only errors must not fence unrelated producer work or publish.
  queue_test::reset_action();
  bool unrelated = false;
  queue_test::pending.emplace_back([&] { unrelated = true; });
  expect(owner.apply(nullptr, output.data(), queue_test::stream) == cudaErrorInvalidValue,
         "null direction was accepted");
  expect(!unrelated && queue_test::fences == 0, "validation rejection fenced producer work");
  cudaStreamSynchronize(queue_test::stream);

  queue_test::reset_action();
  expect(owner.apply(direction.data(), output.data(), queue_test::stream) == cudaSuccess,
         "ordinary physical action failed");
  const auto launch_count = queue_test::launches;
  expect(launch_count > 0 && queue_test::pending.empty(), "ordinary action did not complete");
  for (int failure = 1; failure <= launch_count; ++failure) {
    queue_test::reset_action(failure);
    std::fill(output.begin(), output.end(), 27);
    const auto error = owner.apply(direction.data(), output.data(), queue_test::stream);
    queue_test::returned = true;
    const bool drained = queue_test::pending.empty();
    // Deliberately keep borrowed storage alive while observing a broken owner.
    // This turns missing completion into an assertion rather than host UB.
    cudaStreamSynchronize(queue_test::stream);
    expect(error != cudaSuccess, "injected launch failure was accepted");
    expect(drained && queue_test::late_reads == 0,
           "failed physical action returned with borrowed work still queued");
    expect(std::all_of(output.begin(), output.end(), [](double value) { return value == 27; }),
           "failed physical action published output");
  }
  queue_test::reset_action();
  expect(owner.apply(direction.data(), output.data(), queue_test::stream) == cudaSuccess,
         "physical action could not recover after launch rejection");
  queue_test::reset_action();
  queue_test::capturing = true;
  expect(owner.apply(direction.data(), output.data(), queue_test::stream) ==
             cudaErrorStreamCaptureUnsupported,
         "captured stream was accepted");
  expect(queue_test::fences == 0 && queue_test::pending.empty(), "capture rejection fenced");
  queue_test::capturing = false;
  std::cout << "PASS spins=" << spins << " fitted=" << fitted
            << " launch_failure_points=" << launch_count << '\n';
}
}  // namespace

int main() {
  try {
    for (unsigned spins : {1U, 2U})
      for (bool fitted : {false, true}) run(spins, fitted);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
