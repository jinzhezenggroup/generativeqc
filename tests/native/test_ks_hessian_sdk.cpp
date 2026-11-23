#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

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
    generativeqc_ks_host_hessian_options_v1 options{};
    generativeqc::check(generativeqc_ks_hessian_cpu_default_options_v1(&options));
    options.expected_atoms = atoms.data();
    options.expected_atom_count = atoms.size();
    options.expected_multiplicity = 1;
    std::size_t minimum_bytes{};
    const auto admission = generativeqc_ks_hessian_cpu_minimum_bytes_v1(
        context.get(), system.get(), "pbe-rks", 0, &options, &minimum_bytes);
    if (admission == GENERATIVEQC_STATUS_SUCCESS) {
      const auto work = batch.ks_cpu_hvp(0, direction, action, options);
      require(work.completed_directions == 1 && work.numeric_reservation_bytes == minimum_bytes &&
                  action != sentinel,
              "linked spherical CPU PBE did not publish its HVP");
    } else {
      require(admission == GENERATIVEQC_STATUS_NOT_IMPLEMENTED &&
                  rejection([&] { batch.ks_cpu_hvp(0, direction, action, options); }) ==
                      GENERATIVEQC_STATUS_NOT_IMPLEMENTED &&
                  action == sentinel,
              "an unlinked CPU PBE request modified the HVP output");
    }
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

void run_linked_cpu(const char* selector) {
  require(generativeqc_ks_hessian_cpu_version_v1() == 1, "CPU Hessian package is not linked");
  const generativeqc_context_descriptor context_options{sizeof(generativeqc_context_descriptor),
                                                        GENERATIVEQC_ABI_VERSION, 0,
                                                        GENERATIVEQC_BACKEND_CPU_REFERENCE};
  generativeqc::Context context(context_options);
  std::array<generativeqc_atom, 2> atoms{{{1, .03, -.02, .01}, {1, .15, .13, 1.5}}};
  const std::string_view method_name(selector);
  const bool unrestricted = method_name.ends_with("-uks");
  const int multiplicity = unrestricted ? 2 : 1;
  if (unrestricted) atoms[0].atomic_number = 2;
  const bool semilocal =
      method_name == "lda-rks" || method_name == "pbe-rks" || method_name == "pbe-d3(bj)-rks";
  const bool semilocal_uks =
      method_name == "lda-uks" || method_name == "pbe-uks" || method_name == "pbe-d3(bj)-uks";
  generativeqc::System system(context, atoms, "sto-3g", 0, multiplicity,
                              GENERATIVEQC_BASIS_SPHERICAL);
  const std::array<const generativeqc::System*, 1> systems{&system};
  generativeqc_ks_options grid{};
  grid.struct_size = sizeof(grid);
  grid.abi_version = GENERATIVEQC_ABI_VERSION;
  grid.grid_version = 1;
  grid.radial_points = 12;
  grid.angular_polar = 4;
  grid.angular_azimuth = 8;
  grid.partition_iterations = 3;
  grid.coincident_tolerance = 1e-12;
  grid.tile_points = 64;
  generativeqc_method_descriptor method{};
  method.struct_size = sizeof(method);
  method.abi_version = GENERATIVEQC_ABI_VERSION;
  method.method = unrestricted ? GENERATIVEQC_METHOD_PBE_UKS : GENERATIVEQC_METHOD_PBE_RKS;
  method.max_iterations = 200;
  method.diis_history = 8;
  method.energy_tolerance = 1e-12;
  method.density_tolerance = 1e-10;
  method.screening_tolerance = 1e-14;
  method.precision_mode = GENERATIVEQC_PRECISION_FP64;
  method.ks_options = &grid;
  const auto* named_selector = method_name.starts_with("lda-")
                                   ? (unrestricted ? "lda_xc_pw-uks" : "lda_xc_pw-rks")
                                   : selector;
  const generativeqc_named_ks_descriptor named{
      sizeof(named), GENERATIVEQC_ABI_VERSION, named_selector, nullptr, 16U << 20, 0};
  generativeqc_ks_host_hessian_options_v1 options{};
  generativeqc::check(generativeqc_ks_hessian_cpu_default_options_v1(&options));
  std::size_t hvp_floor{}, dense_floor{}, unchanged_floor{37};
  generativeqc::check(generativeqc_ks_hessian_cpu_minimum_bytes_v1(
      context.get(), system.get(), named_selector, 0, &options, &hvp_floor));
  generativeqc::check(generativeqc_ks_hessian_cpu_minimum_bytes_v1(
      context.get(), system.get(), named_selector, 1, &options, &dense_floor));
  require(hvp_floor > 0 && dense_floor > hvp_floor &&
              generativeqc_ks_hessian_cpu_minimum_bytes_v1(context.get(), system.get(), "unknown",
                                                           0, &options, &unchanged_floor) ==
                  GENERATIVEQC_STATUS_NOT_IMPLEMENTED &&
              unchanged_floor == 37,
          "CPU Hessian query did not publish a transactional byte floor");
  auto batch = generativeqc::Batch::prepare_named_ks(context, systems, method, named);

  options.expected_atoms = atoms.data();
  options.expected_atom_count = atoms.size();
  options.expected_multiplicity = multiplicity;
  const std::array<double, 6> direction{.02, -.03, .1, -.02, .03, -.1};
  std::array<double, 6> action;
  action.fill(37);
  const auto sentinel = action;
  (void)rejection([&] { batch.ks_cpu_hvp(0, direction, action, options); });
  (void)rejection([&] { batch.ks_cpu_hvp_batch_minimum_bytes(0, 2, options); });
  require(action == sentinel, "unsolved CPU batch modified the caller output");

  auto solved = batch.execute({}, false);
  require(solved[0].converged && solved[0].forces.empty() &&
              solved[0].executed_backend == GENERATIVEQC_BACKEND_CPU_REFERENCE,
          "named SDK batch did not solve on CPU");
  const auto hvp = batch.ks_cpu_hvp(0, direction, action, options);
  std::array<double, 12> seeds{}, actions{};
  for (std::size_t index = 0; index < direction.size(); ++index) {
    seeds[index] = direction[index];
    seeds[direction.size() + index] = -.5 * direction[index];
  }
  const auto hvp_batch = batch.ks_cpu_hvp_batch(0, seeds, actions, options);
  std::size_t panel_floor{};
  generativeqc::check(generativeqc_ks_hvp_cpu_batch_minimum_bytes_v1(
      context.get(), system.get(), named_selector, 2, &options, &panel_floor));
  require(batch.ks_cpu_hvp_batch_minimum_bytes(0, 2, options) == panel_floor &&
              panel_floor <= hvp_batch.numeric_reservation_bytes,
          "prepared and pre-SCF panel floors diverged");
  require(hvp_batch.completed_directions == 2 && hvp_batch.d3_pair_visits == 2 * hvp.d3_pair_visits,
          "SDK multi-HVP lost a requested direction or additive correction");
  for (std::size_t index = 0; index < direction.size(); ++index)
    require(std::abs(actions[index] - action[index]) < 2e-8 &&
                std::abs(actions[direction.size() + index] + .5 * action[index]) < 2e-8,
            "SDK action panel violates scalar parity or linearity");
  std::array<double, 36> matrix{};
  const auto dense = batch.ks_cpu_hessian(0, matrix, options);
  generativeqc_ks_host_hessian_receipt_v1 sequential_work{};
  require(hvp_floor == hvp.numeric_reservation_bytes &&
              dense_floor == hvp_floor + dense.output_bytes - hvp.output_bytes,
          "CPU Hessian query does not match the executed mandatory reservation");
  if (semilocal) {
    std::array<double, 36> sequential{};
    sequential_work = batch.ks_cpu_hessian_sequential(0, sequential, options);
    require(sequential_work.completed_directions == direction.size() &&
                sequential_work.first_eri_primitives == dense.first_eri_primitives &&
                sequential_work.curvature_eri_primitives == dense.curvature_eri_primitives &&
                sequential_work.d3_pair_visits == dense.d3_pair_visits,
            "sequential CPU Hessian did not preserve the complete stationary work");
    for (std::size_t element = 0; element < matrix.size(); ++element)
      require(std::abs(sequential[element] - matrix[element]) < 2e-8,
              "sequential SDK changed the raw derivative");
    auto sequential_limit = options;
    sequential_limit.output_bytes = sequential_work.output_bytes;
    std::array<double, 36> sequential_replay{};
    batch.ks_cpu_hessian_sequential(0, sequential_replay, sequential_limit);
    require(sequential_replay == sequential, "sequential Hessian could not use exact output bytes");
    --sequential_limit.output_bytes;
    require(rejection([&] {
              batch.ks_cpu_hessian_sequential(0, sequential_replay, sequential_limit);
            }) == GENERATIVEQC_STATUS_NUMERICAL_FAILURE &&
                sequential_replay == sequential,
            "one-byte-short sequential SDK call published output");
    std::array<double, 36> recycled{};
    const auto work = batch.ks_cpu_hessian_recycled(0, recycled, options, 4);
    require(work.completed_directions == direction.size() && work.gmres_workspace_bytes > 0,
            "recycled CPU Hessian did not finish all RHS, including its short last group");
    for (std::size_t element = 0; element < matrix.size(); ++element)
      require(std::abs(recycled[element] - matrix[element]) < 2e-8,
              "recycled native Hessian changed the raw derivative");
    auto refused = options;
    refused.gmres_bytes = work.gmres_workspace_bytes;
    std::array<double, 36> exact{};
    const auto exact_work = batch.ks_cpu_hessian_recycled(0, exact, refused, 4);
    require(exact == recycled && exact_work.gmres_workspace_bytes == refused.gmres_bytes,
            "recycled Hessian could not replay at its exact group capacity");
    --refused.gmres_bytes;
    recycled.fill(37);
    const auto sentinel_matrix = recycled;
    require(rejection([&] { batch.ks_cpu_hessian_recycled(0, recycled, refused, 4); }) ==
                    GENERATIVEQC_STATUS_NUMERICAL_FAILURE &&
                recycled == sentinel_matrix,
            "recycled Hessian did not refuse below group capacity without publication");
  } else if (semilocal_uks) {
    std::array<double, 36> refused{};
    refused.fill(37);
    const auto sentinel_matrix = refused;
    require(rejection([&] { batch.ks_cpu_hessian_recycled(0, refused, options, 4); }) ==
                    GENERATIVEQC_STATUS_NOT_IMPLEMENTED &&
                refused == sentinel_matrix,
            "semilocal UKS must refuse the unqualified recycled schedule before publication");
    require(rejection([&] { batch.ks_cpu_hessian_sequential(0, refused, options); }) ==
                    GENERATIVEQC_STATUS_NOT_IMPLEMENTED &&
                refused == sentinel_matrix,
            "semilocal UKS must refuse the unqualified sequential schedule before publication");
  }
  const bool corrected = std::string_view(selector).find("d3(bj)") != std::string_view::npos;
  require(hvp.completed_directions == 1 && dense.completed_directions == direction.size() &&
              (hvp.d3_pair_visits > 0) == corrected &&
              std::abs(hvp.total_energy - solved[0].energy) < 2e-10,
          "SDK did not differentiate the accepted native total method");
  for (std::size_t row = 0; row < direction.size(); ++row) {
    double dense_action = 0;
    for (std::size_t column = 0; column < direction.size(); ++column)
      dense_action += matrix[direction.size() * row + column] * direction[column];
    require(std::abs(dense_action - action[row]) < 2e-8, "CPU SDK HVP differs from dense action");
  }
  auto bounded_dense = options;
  bounded_dense.numeric_bytes = dense_floor;
  std::array<double, 36> bounded_matrix{};
  const auto bounded = batch.ks_cpu_hessian(0, bounded_matrix, bounded_dense);
  require(bounded.numeric_reservation_bytes == dense_floor,
          "CPU dense Hessian could not replay at the queried byte floor");
  for (std::size_t element = 0; element < matrix.size(); ++element)
    require(std::abs(bounded_matrix[element] - matrix[element]) < 2e-8,
            "bounded CPU Hessian changed raw curvature");
  --bounded_dense.numeric_bytes;
  bounded_matrix.fill(37);
  const auto matrix_sentinel = bounded_matrix;
  require(rejection([&] { batch.ks_cpu_hessian(0, bounded_matrix, bounded_dense); }) ==
                  GENERATIVEQC_STATUS_NUMERICAL_FAILURE &&
              bounded_matrix == matrix_sentinel,
          "CPU dense Hessian did not refuse one byte below the floor transactionally");
  const auto initial_action = action;
  if (semilocal || semilocal_uks || method_name.starts_with("pbe0")) {
    // The single-direction plan has no optional matrix panels to release.
    auto exact = options;
    exact.numeric_bytes = hvp_floor;
    exact.output_bytes = hvp.output_bytes;
    action.fill(37);
    const auto exact_receipt = batch.ks_cpu_hvp(0, direction, action, exact);
    require(
        exact_receipt.numeric_reservation_bytes == exact.numeric_bytes && action == initial_action,
        "linked CPU method could not replay at its exact HVP capacity");
    for (const auto limit : {&generativeqc_ks_host_hessian_options_v1::numeric_bytes,
                             &generativeqc_ks_host_hessian_options_v1::output_bytes}) {
      auto below = exact;
      --(below.*limit);
      action.fill(37);
      require(rejection([&] { batch.ks_cpu_hvp(0, direction, action, below); }) ==
                      GENERATIVEQC_STATUS_NUMERICAL_FAILURE &&
                  action == sentinel,
              "linked CPU method did not refuse below-capacity HVP transactionally");
    }
    action = initial_action;
  }

  const std::vector<std::optional<std::vector<double>>> moved{
      std::vector<double>{.03, -.02, .01, .15, .13, 1.57}};
  solved = batch.execute(moved, false);
  require(solved[0].converged, "moved CPU geometry failed to converge");
  require(rejection([&] { batch.ks_cpu_hvp(0, direction, action, options); }) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              action == initial_action,
          "stale expected geometry was accepted or modified output");
  atoms[1].z = 1.57;
  auto tight = options;
  tight.numeric_bytes = 1;
  require(rejection([&] { batch.ks_cpu_hvp(0, direction, action, tight); }) ==
              GENERATIVEQC_STATUS_NUMERICAL_FAILURE,
          "CPU derivative did not refuse the aggregate reservation");
  require(action == initial_action, "CPU resource refusal modified output");
  const auto replay = batch.ks_cpu_hvp(0, direction, action, options);
  require(replay.completed_directions == 1 && action != initial_action,
          "CPU SDK did not recover on the moved accepted geometry");

  std::cout << std::setprecision(17) << "{\"energy\":" << hvp.total_energy
            << ",\"d3_pair_visits\":" << hvp.d3_pair_visits << ",\"value\":[";
  for (std::size_t coordinate = 0; coordinate < initial_action.size(); ++coordinate)
    std::cout << (coordinate ? "," : "") << initial_action[coordinate];
  std::cout << "],\"hvp_panel\":[";
  for (std::size_t element = 0; element < actions.size(); ++element)
    std::cout << (element ? "," : "") << actions[element];
  std::cout << "],\"panel_d3_pair_visits\":" << hvp_batch.d3_pair_visits
            << ",\"dense_d3_pair_visits\":" << dense.d3_pair_visits << ",\"matrix\":[";
  for (std::size_t element = 0; element < matrix.size(); ++element)
    std::cout << (element ? "," : "") << matrix[element];
  std::cout << ']';
  if (semilocal)
    std::cout << ",\"sequential\":{\"output_bytes\":" << sequential_work.output_bytes
              << ",\"response_iterations\":" << sequential_work.response_iterations
              << ",\"induced_fock_actions\":" << sequential_work.induced_fock_actions
              << ",\"d3_pair_visits\":" << sequential_work.d3_pair_visits << '}';
  std::cout << "}\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const bool cuda = argc == 2 && std::string_view(argv[1]) == "--cuda";
    const bool cpu = (argc == 2 || argc == 3) && std::string_view(argv[1]) == "--cpu";
    require(argc == 1 || cuda || cpu, "expected --cpu [method] or --cuda (Slurm only)");
    require(
        !cuda || (std::getenv("SLURM_JOB_ID") && generativeqc_ks_hessian_cuda_version_v2() == 2),
        "CUDA SDK qualification requires a linked package and Slurm allocation");
    if (cpu)
      run_linked_cpu(argc == 3 ? argv[2] : "b3lyp-d3(bj)-rks");
    else
      run(cuda);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
