#include <iostream>

#include "physical_fock_host.cpp"

namespace {
void expect(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
generativeqc::dft::CudaKsFinalStateToken configure(unsigned spins, bool fitted) {
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
  return token;
}
void run(unsigned spins, bool fitted) {
  using namespace generativeqc;
  const auto token = configure(spins, fitted);
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
void run_resident(unsigned spins, queue_test::ResidentFailure failure) {
  using namespace generativeqc;
  using Failure = queue_test::ResidentFailure;
  expect(queue_test::live_plans == 0, "previous owner leaked a Direct plan");
  queue_test::resident_failure = failure;
  queue_test::plan_creations = queue_test::range_replays = queue_test::range_cached = 0;
  {
    const auto token = configure(spins, false);
    methods::PreparedBatch batch;
    dft::CudaKsWb97mvResponse owner;
    dft::CudaKsWb97mvResponseOptions options;
    options.defer_resident_integrals = true;
    std::string detail;
    queue_test::reset_action();
    expect(owner.prepare(batch, 0, token, options, detail) == GENERATIVEQC_STATUS_SUCCESS,
           "mandatory response preparation failed");
    const auto mandatory_bytes = owner.work().device_bytes;
    expect(queue_test::live_plans == 1 && owner.work().resident_integral_bytes == 0,
           "deferred preparation retained optional integrals");
    const auto result = owner.retain_integrals(1024, detail);
    const auto expected = failure == Failure::numerical     ? GENERATIVEQC_STATUS_NUMERICAL_FAILURE
                          : failure == Failure::unsupported ? GENERATIVEQC_STATUS_NOT_IMPLEMENTED
                          : failure == Failure::creation_device ? GENERATIVEQC_STATUS_CUDA_ERROR
                                                                : GENERATIVEQC_STATUS_SUCCESS;
    expect(result == expected, "resident admission changed failure classification");
    const auto complete = failure == Failure::none;
    const auto work = owner.work();
    expect(work.device_bytes == mandatory_bytes + (complete ? 256 : 64),
           "optional refusal lost or double-charged device storage");
    expect(work.resident_integral_bytes == (complete ? 128 : 64) &&
               work.resident_source_evaluations == (complete ? 16 : 8),
           "optional refusal changed the retained primary source");
    expect(queue_test::live_plans == (complete ? 2 : 1), "optional plan was not retired");
    expect(owner.retain_integrals(1024, detail) == GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
               owner.work().device_bytes == work.device_bytes,
           "repeated admission altered a retained or failed owner");
    queue_test::range_replays = queue_test::range_cached = 0;
    queue_test::reset_action();
    std::vector<double> direction(4 * spins, 1), output(4 * spins, 27);
    const auto action = owner.apply(direction.data(), output.data(), queue_test::stream);
    if (expected == GENERATIVEQC_STATUS_SUCCESS) {
      expect(action == cudaSuccess, "optional refusal disabled exact response replay");
      expect(queue_test::range_cached == (complete ? 1 : 0) &&
                 queue_test::range_replays == (complete ? 0 : 1),
             "range response used the wrong retained or replay source");
    } else {
      expect(action != cudaSuccess && std::all_of(output.begin(), output.end(),
                                                  [](double value) { return value == 27; }),
             "failed source admission published response output");
    }
    expect(queue_test::pending.empty(), "resident admission or action left borrowed work");
  }
  expect(queue_test::live_plans == 0 && queue_test::pending.empty(),
         "resident response teardown leaked a plan or queued work");
  queue_test::resident_failure = Failure::none;
  std::cout << "PASS resident spins=" << spins << " failure=" << static_cast<int>(failure) << '\n';
}
}  // namespace

int main() {
  try {
    for (unsigned spins : {1U, 2U})
      for (bool fitted : {false, true}) run(spins, fitted);
    for (unsigned spins : {1U, 2U})
      for (auto failure :
           {queue_test::ResidentFailure::none, queue_test::ResidentFailure::canonical_oom,
            queue_test::ResidentFailure::values_oom, queue_test::ResidentFailure::numerical,
            queue_test::ResidentFailure::unsupported, queue_test::ResidentFailure::creation_device})
        run_resident(spins, failure);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
