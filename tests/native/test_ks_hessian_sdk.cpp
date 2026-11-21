#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>

#include "generativeqc/generativeqc.hpp"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Call>
generativeqc_status rejection(Call call) {
  try {
    call();
  } catch (const generativeqc::Error& error) {
    return error.status();
  }
  throw std::runtime_error("unqualified SDK Hessian request succeeded");
}

void run(bool device) {
  const generativeqc_context_descriptor context_options{
      sizeof(generativeqc_context_descriptor), GENERATIVEQC_ABI_VERSION, 0,
      device ? GENERATIVEQC_BACKEND_CUDA : GENERATIVEQC_BACKEND_CPU_REFERENCE};
  generativeqc::Context context(context_options);
  const std::array<generativeqc_atom, 2> atoms{{{1, 0, 0, 0}, {1, 0.15, 0.13, 1.55}}};
  generativeqc::System system(context, atoms, "sto-3g", 0, 1, GENERATIVEQC_BASIS_SPHERICAL);
  const std::array<const generativeqc::System*, 1> systems{&system};
  const std::array<generativeqc_ks_semilocal_component, 2> semilocal{
      {{"GGA_C_PBE", 1.0}, {"GGA_X_PBE", 1.0}}};
  const std::array<generativeqc_ks_exchange_term, 2> exchange{
      {{GENERATIVEQC_KS_EXCHANGE_SHORT_RANGE, .2, .33, -.1},
       {GENERATIVEQC_KS_EXCHANGE_LONG_RANGE, .65, .33, -.325}}};
  generativeqc_ks_options ks{};
  ks.struct_size = sizeof(ks);
  ks.abi_version = GENERATIVEQC_ABI_VERSION;
  ks.scf_domain = "semilocal-scaled-v1/pbe-spin-c2-1e-18";
  ks.spin_channels = 1;
  ks.grid_version = 1;
  ks.radial_points = 24;
  ks.angular_polar = 12;
  ks.angular_azimuth = 24;
  ks.partition_iterations = 3;
  ks.coincident_tolerance = 1e-12;
  ks.xc_execution_schedule = GENERATIVEQC_XC_EXECUTION_DEVICE_FUSED;
  ks.semilocal_components = semilocal.data();
  ks.semilocal_component_count = semilocal.size();
  ks.exchange_terms = exchange.data();
  ks.exchange_term_count = exchange.size();
  generativeqc_method_descriptor method{};
  method.struct_size = sizeof(method);
  method.abi_version = GENERATIVEQC_ABI_VERSION;
  method.method = GENERATIVEQC_METHOD_PBE_RKS;
  method.max_iterations = 200;
  method.diis_history = 8;
  method.energy_tolerance = 1e-13;
  method.density_tolerance = 1e-11;
  method.screening_tolerance = 1e-14;
  method.precision_mode = GENERATIVEQC_PRECISION_FP64;
  method.ks_options = device ? &ks : nullptr;
  generativeqc::Batch batch(context, systems, method);

  std::array<double, 6> direction{};
  for (std::size_t coordinate = 0; coordinate < direction.size(); ++coordinate)
    direction[coordinate] = std::sin(coordinate + .5) / 2;
  std::array<double, 6> action;
  action.fill(37);
  const auto sentinel = action;
  if (device) {
    (void)rejection([&] { batch.ks_hvp(0, direction, action); });
    require(action == sentinel, "unsolved batch modified the HVP output");
  }
  const auto solved = batch.execute({}, false);
  require(solved.size() == 1 && solved[0].converged && solved[0].forces.empty() &&
              solved[0].executed_backend ==
                  (device ? GENERATIVEQC_BACKEND_CUDA : GENERATIVEQC_BACKEND_CPU_REFERENCE),
          "SDK KS state failed to converge on its requested provider");
  const auto saved = direction;
  require(rejection([&] { batch.ks_hvp(0, direction, direction); }) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              direction == saved,
          "overlapping HVP input and output were not rejected transactionally");
  if (!device) {
    require(rejection([&] { batch.ks_hvp(0, direction, action); }) ==
                    GENERATIVEQC_STATUS_NOT_IMPLEMENTED &&
                action == sentinel,
            "CPU state entered CUDA Hessian execution or modified output");
    return;
  }

  generativeqc_ks_hessian_options_v2 tight{};
  generativeqc::check(generativeqc_ks_hessian_default_options_v2(&tight));
  tight.device_bytes = 256;
  require(rejection([&] { batch.ks_hvp(0, direction, action, &tight); }) ==
                  GENERATIVEQC_STATUS_OUT_OF_MEMORY &&
              action == sentinel,
          "bounded Hessian failure did not preserve the caller's output");
  const auto hvp = batch.ks_hvp(0, direction, action);
  std::array<double, 36> matrix{};
  const auto dense = batch.ks_hessian(0, matrix);
  require(hvp.completed_directions == 1 && dense.completed_directions == 6 &&
              hvp.exact_primitive_components && !hvp.df_value_passes && !hvp.vv10_reference_pairs,
          "receipt does not describe the completed Direct range-PBE endpoint");
  for (std::size_t row = 0; row < direction.size(); ++row) {
    double direct_action = 0;
    for (std::size_t column = 0; column < direction.size(); ++column) {
      const auto coefficient = matrix[6 * row + column];
      require(std::abs(coefficient - matrix[6 * column + row]) < 2e-8,
              "SDK Hessian is not symmetric before publication");
      direct_action += coefficient * direction[column];
    }
    require(std::abs(direct_action - action[row]) < 2e-8, "SDK HVP differs from dense action");
  }
  std::cout << std::setprecision(17) << "[";
  for (std::size_t coordinate = 0; coordinate < action.size(); ++coordinate)
    std::cout << (coordinate ? "," : "") << action[coordinate];
  std::cout << "]\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const bool cuda = argc == 2 && std::string_view(argv[1]) == "--cuda";
    require(argc == 1 || cuda, "expected no options (host) or --cuda (Slurm only)");
    require(
        !cuda || (std::getenv("SLURM_JOB_ID") && generativeqc_ks_hessian_cuda_version_v2() == 2),
        "CUDA SDK qualification requires a linked package and Slurm allocation");
    run(cuda);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
