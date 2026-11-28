#include <array>
#include <climits>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/types.hpp"
#include "dft/ao_grid.hpp"
#include "dft/contracted_coulomb.hpp"
#include "dft/cuda_wb97mv_nuclear_jk.hpp"
#include "integrals/coulomb_derivative_envelope.hpp"
#include "molecule/basis.hpp"

namespace {
using namespace generativeqc;
using namespace generativeqc::dft;
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void check(cudaError_t error) { require(error == cudaSuccess, cudaGetErrorString(error)); }
struct Storage {
  cudaStream_t stream{};
  std::vector<double*> allocations;
  Storage() { check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
  ~Storage() {
    cudaStreamSynchronize(stream);
    for (auto* pointer : allocations) cudaFree(pointer);
    cudaStreamDestroy(stream);
  }
  double* allocate(std::size_t count) {
    double* pointer{};
    check(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(double)));
    allocations.push_back(pointer);
    return pointer;
  }
  double* upload(const std::vector<double>& values) {
    auto* pointer = allocate(values.size());
    check(cudaMemcpyAsync(pointer, values.data(), values.size() * sizeof(double),
                          cudaMemcpyHostToDevice, stream));
    return pointer;
  }
};
template <class Value>
std::vector<Value> read(std::ifstream& input, std::size_t count) {
  require(count <= (1ULL << 24), "oversized fixture");
  std::vector<Value> values(count);
  require(
      static_cast<bool>(input.read(reinterpret_cast<char*>(values.data()), count * sizeof(Value))),
      "short fixture");
  return values;
}
std::vector<double> download(const double* pointer, std::size_t count) {
  std::vector<double> values(count);
  check(cudaMemcpy(values.data(), pointer, count * sizeof(double), cudaMemcpyDeviceToHost));
  return values;
}
struct Owner {
  bool live{true};
  unsigned remaining{UINT_MAX};
  static cudaError_t authenticate(void* context) noexcept {
    auto& owner = *static_cast<Owner*>(context);
    if (!owner.live || !owner.remaining) return cudaErrorInvalidResourceHandle;
    if (owner.remaining != UINT_MAX) --owner.remaining;
    return cudaSuccess;
  }
};

/** Compare the derivative-only table against unrestricted J/K actions on the
 * same normalized basis and exercise failed publication in its transaction. */
void screened_table(const core::System& system, const CudaWb97mvJkShape& shape,
                    std::span<const integrals::CudaOneElectronShell> shells,
                    CudaWb97mvJkOptions options, const CudaWb97mvJkReference& reference,
                    Storage& storage, double* motion, double* density, double* delta,
                    CudaWb97mvJkOutput output, bool expect_indexed = false) {
  using Failure = CudaWb97mvJkAdmissionFailure;
  const auto certificate = integrals::coulomb_derivative_omissions(
      system, integrals::CoulombDerivativeOmissions::max_bound, 200, expect_indexed);
  require(certificate.size() != 0, "fixture requires certified omissions");
  std::size_t bytes{};
  check(cuda_wb97mv_jk_workspace(shape, shells, options, bytes));
  CudaWb97mvNuclearJk exact;
  check(exact.prepare(shape, shells, options, reference, storage.allocate(bytes / sizeof(double)),
                      bytes));
  auto refused = options;
  refused.max_groups = 8;
  refused.max_source_groups = 1;
  Failure reason{};
  require(cuda_wb97mv_jk_workspace(shape, shells, refused, bytes, nullptr, nullptr, &reason) ==
                  cudaErrorInvalidValue &&
              !bytes && reason == Failure::canonical_source_limit,
          "canonical source census did not report its structured refusal");
  options.source_tile = 2;
  options.prefer_fresh_direction_batches = true;
  const auto pairs = shells.size() * (shells.size() + 1) / 2;
  const auto candidates = pairs * (pairs + 1) / 2;
  const auto retained_sources = candidates - certificate.size();
  auto replay_options = options;
  replay_options.max_source_groups = 1;
  replay_options.max_frozen_replay_candidates = retained_sources - 1;
  require(cuda_wb97mv_jk_workspace(shape, shells, replay_options, bytes, nullptr, &certificate,
                                   &reason) == cudaErrorInvalidValue &&
              !bytes && reason == Failure::canonical_source_limit,
          "replay admitted more individually visited groups than budgeted");
  replay_options.max_frozen_replay_candidates = retained_sources;
  if (expect_indexed) {
    require(!certificate.omitted_rank_words.empty(), "indexed fixture requires a rank bitmap");
    auto malformed = certificate;
    ++malformed.rank_count;
    require(cuda_wb97mv_jk_workspace(shape, shells, replay_options, bytes, nullptr, &malformed) ==
                    cudaErrorInvalidValue &&
                !bytes,
            "frozen replay accepted a bitmap for a different shell census");
  }
  auto short_work = replay_options;
  short_work.max_primitive_component_evaluations = 1;
  require(cuda_wb97mv_jk_workspace(shape, shells, short_work, bytes, nullptr, &certificate) ==
                  cudaErrorInvalidValue &&
              !bytes,
          "replay bypassed the primitive work budget");
  CudaWb97mvNuclearJk::Plan plan;
  check(plan.build(shape, shells, options, nullptr, &certificate));
  bytes = plan.bytes();
  auto* arena = storage.allocate(bytes / sizeof(double));
  CudaWb97mvNuclearJk screened;
  require(screened.prepare(plan, reference, arena, bytes - 1) == cudaErrorInvalidValue,
          "screened plan admitted a short arena");
  check(screened.prepare(plan, reference, arena, bytes));
  require(!plan.bytes(), "bound plan retained a second executable schedule");
  check(plan.build(shape, shells, replay_options, nullptr, &certificate));
  CudaWb97mvNuclearJk replayed;
  const auto replay_bytes = plan.bytes();
  require(replay_bytes == bytes, "replay did not preserve the bounded scratch contract");
  auto* replay_arena = storage.allocate(replay_bytes / sizeof(double));
  require(
      replayed.prepare(plan, reference, replay_arena, replay_bytes - 1) == cudaErrorInvalidValue,
      "replay admitted a short arena");
  check(replayed.prepare(plan, reference, replay_arena, replay_bytes));
  require(replayed.apply(motion, density, delta, output, storage.stream) == cudaErrorNotSupported,
          "dense replay published ordinary J/K values or HVP");
  require(screened.apply(motion, density, delta, output, storage.stream) == cudaErrorNotSupported,
          "screened plan published ordinary J/K values or HVP");
  const auto coordinates = 3 * shape.atoms;
  const auto panels = shape.nao * shape.nao;
  const auto values = coordinates * (coordinates + panels);
  auto* table = storage.allocate(values);
  auto* transaction = reinterpret_cast<int*>(storage.allocate(1));
  const auto sweep = [&](CudaWb97mvNuclearJk& source) {
    check(cudaMemsetAsync(transaction, 0, sizeof(int), storage.stream));
    check(source.enqueue_private_frozen_dense(motion, density, delta, table, transaction,
                                              storage.stream));
    int status{};
    check(cudaMemcpyAsync(&status, transaction, sizeof(status), cudaMemcpyDeviceToHost,
                          storage.stream));
    check(cudaStreamSynchronize(storage.stream));
    return status;
  };
  require(sweep(screened) == 0, "screened table rejected a valid density");
  const auto screened_values = download(table, values);
  require(sweep(replayed) == 0 && download(table, values) == screened_values,
          "bounded host replay changed the screened matrix");
  const auto& replay_work = replayed.work();
  const auto& retained_work = screened.work();
  if (expect_indexed)
    require(retained_work.indexed_component_blocks &&
                replay_work.indexed_component_blocks == retained_work.indexed_component_blocks,
            "fixture did not exercise indexed dispatch in retained and replayed sweeps");
  require(replay_work.component_evaluations == retained_work.component_evaluations &&
              replay_work.projection_terms_bound == retained_work.projection_terms_bound &&
              replay_work.reduction_terms == retained_work.reduction_terms &&
              replay_work.source_groups == retained_work.source_groups,
          "bounded host replay changed the semantic work");
  CudaWb97mvNuclearJk contracted;
  bool contracted_supported{};
  if (options.part == CudaWb97mvJkPart::coulomb) {
    int shared_bytes{};
    check(cudaDeviceGetAttribute(&shared_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                 reference.device_id));
    contracted_supported = shared_bytes >= hessian_aot::contracted::max_quartet_shared_bytes;
  }
  if (contracted_supported) {
    auto contracted_options = replay_options;
    contracted_options.contract_frozen_coulomb = true;
    check(plan.build(shape, shells, contracted_options, nullptr, &certificate));
    const auto contracted_bytes = plan.bytes();
    check(contracted.prepare(plan, reference, storage.allocate(contracted_bytes / sizeof(double)),
                             contracted_bytes));
    require(
        contracted.apply(motion, density, delta, output, storage.stream) == cudaErrorNotSupported,
        "certified contraction published an arbitrary HVP");
    require(sweep(contracted) == 0, "contracted table rejected valid fixed-density columns");
    require(contracted.work().integral_source_builds == 1 &&
                contracted.work().source_groups == retained_sources,
            "contracted sweep did not visit the certified inventory once");
    for (std::size_t entry = 0; const auto value : download(table, values))
      require(std::abs(value - screened_values[entry++]) < 1e-13,
              "contracted frozen source differs from the certified replay");
    contracted_options.max_workspace_bytes = bytes;
    check(plan.build(shape, shells, contracted_options, nullptr, &certificate));
    require(plan.bytes() <= bytes, "contracted admission bypassed the caller's arena cap");
  }
  // Reordering and resizing certified windows changes the rounding boundaries
  // of translation folds but must not change the authenticated frozen table.
  options.source_tile = candidates;
  options.frozen_atom_block_size = 1;
  check(plan.build(shape, shells, options, nullptr, &certificate));
  CudaWb97mvNuclearJk wider;
  const auto wide_bytes = plan.bytes();
  check(wider.prepare(plan, reference, storage.allocate(wide_bytes / sizeof(double)), wide_bytes));
  require(sweep(wider) == 0, "wider window rejected a valid density");
  for (std::size_t entry = 0; const auto value : download(table, values))
    require(std::abs(value - screened_values[entry++]) < 1e-13,
            "wider window changed the frozen table");
  for (std::size_t column = 0; column < coordinates; ++column) {
    std::vector<double> unit(coordinates);
    unit[column] = 1;
    check(cudaMemcpyAsync(motion, unit.data(), coordinates * sizeof(double), cudaMemcpyHostToDevice,
                          storage.stream));
    check(exact.apply(motion, density, delta, output, storage.stream));
    for (std::size_t entry = 0; const auto value : download(output.fock_direction, panels))
      require(std::abs(value - screened_values[column * panels + entry++]) < 1e-13,
              "screened frozen Fock table differs from exact");
    for (std::size_t entry = 0; const auto value : download(output.hvp, coordinates))
      require(
          std::abs(value - screened_values[coordinates * panels + column * coordinates + entry++]) <
              1e-13,
          "screened frozen HVP table differs from exact");
  }
  const double outside = 4.25;
  check(cudaMemcpyAsync(density, &outside, sizeof(double), cudaMemcpyHostToDevice, storage.stream));
  require(sweep(screened) != 0 && sweep(replayed) != 0,
          "screened table did not reject out-of-bound density");
  if (contracted_supported)
    require(sweep(contracted) != 0, "contracted table bypassed the certificate density bound");
  const double inside = 0.5;
  check(cudaMemcpyAsync(density, &inside, sizeof(double), cudaMemcpyHostToDevice, storage.stream));
  check(cudaMemcpyAsync(delta, &inside, sizeof(double), cudaMemcpyHostToDevice, storage.stream));
  require(sweep(screened) != 0 && sweep(replayed) != 0,
          "screened frozen table accepted nonzero density response");
  if (contracted_supported)
    require(sweep(contracted) != 0, "contracted frozen table accepted nonzero density response");
  std::cout << "{\"screened_table\":true}\n";
}

void run(const char* source, const char* result, const std::string& cache_mode) {
  std::ifstream input(source, std::ios::binary);
  const auto header = read<std::uint64_t>(input, 7);
  require(header[0] > 0 && header[0] <= 8 && header[1] > 0 && header[1] <= 8 && header[2] <= 1 &&
              header[3] >= 1 && header[3] <= 2 && header[4] <= 6 && header[6] <= 6,
          "invalid fixture header");
  core::System system;
  system.basis_representation = header[2] ? core::kSphericalBasis : core::kCartesianBasis;
  const auto atoms = read<double>(input, 5 * header[0]);
  for (std::size_t atom = 0; atom < header[0]; ++atom)
    system.atoms.push_back({static_cast<int>(atoms[5 * atom]),
                            {atoms[5 * atom + 2], atoms[5 * atom + 3], atoms[5 * atom + 4]},
                            static_cast<int>(atoms[5 * atom + 1])});
  for (std::size_t index = 0; index < header[1]; ++index) {
    const auto metadata = read<std::uint64_t>(input, 3);
    require(metadata[0] < header[0] && metadata[1] <= 3 && metadata[2] > 0 && metadata[2] <= 8,
            "invalid shell");
    const auto radial = read<double>(input, 2 * metadata[2]);
    core::Shell shell{static_cast<unsigned>(metadata[0]), static_cast<unsigned>(metadata[1]), {}};
    for (std::size_t primitive = 0; primitive < metadata[2]; ++primitive)
      shell.primitives.push_back({radial[2 * primitive], radial[2 * primitive + 1]});
    system.shells.push_back(std::move(shell));
  }
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
          detail.c_str());
  AoBasis basis(system);
  const CudaWb97mvJkShape shape{basis.natom, basis.nprimitive, basis.nao, header[3]};
  std::vector<integrals::CudaOneElectronShell> shells;
  std::size_t primitive_begin{}, ao_begin{};
  for (const auto& shell : system.shells) {
    const auto count = header[2] ? 2 * shell.angular_momentum + 1
                                 : molecule::cartesian_count(shell.angular_momentum);
    shells.push_back({shell.atom_index, shell.angular_momentum, primitive_begin,
                      shell.primitives.size(), ao_begin, count});
    primitive_begin += shell.primitives.size();
    ao_begin += count;
  }
  const auto direction = read<double>(input, 3 * shape.atoms);
  const auto panels = shape.spins * shape.nao * shape.nao;
  const auto densities = read<double>(input, 2 * panels);
  CudaWb97mvJkOptions options;
  options.part = static_cast<CudaWb97mvJkPart>(header[4]);
  if (options.part == CudaWb97mvJkPart::full_hybrid)
    options.full_exchange_fraction = read<double>(input, 1)[0];
  if (options.part == CudaWb97mvJkPart::exact &&
      input.peek() != std::ifstream::traits_type::eof()) {
    const auto radial = read<double>(input, 3);
    options.range_exchange = CudaWb97mvJkOptions::RangeExchange{radial[0], radial[1], radial[2]};
  }
  options.radial_tile = header[5];
  if (cache_mode == "tiled") {
    options.max_groups = 16;
    options.source_tile = 1;
  }
  if (cache_mode == "resident-cap-tiled" || cache_mode == "resident-cap-fallback") {
    options.max_resident_integral_bytes = 1;
    options.source_tile = 2;
    if (cache_mode == "resident-cap-fallback") options.max_source_groups = 1;
  }
  if (cache_mode == "fresh-retained" || cache_mode == "fresh-streamed") {
    options.max_resident_integral_bytes = 1;
    options.source_tile = 2;
    if (cache_mode == "fresh-streamed") options.max_retained_window_source_bytes = 0;
    options.prefer_fresh_direction_batches = true;
  }
  if (cache_mode == "streamed") options.source_tile = 0;
  if (cache_mode == "batch-chunked") options.max_batch_projection_bytes = 32ULL << 10;
  if (cache_mode == "projection-budgeted") options.max_batch_projection_bytes = 1;
  if (cache_mode == "streamed" || cache_mode == "budgeted") {
    options.max_resident_integral_bytes = 0;
    if (cache_mode == "budgeted") {
      std::size_t minimum{};
      check(cuda_wb97mv_jk_workspace(shape, shells, options, minimum));
      options.max_workspace_bytes = minimum;
      options.max_resident_integral_bytes = 64ULL << 20;
    }
  }
  if (cache_mode == "unbatched" || cache_mode == "batch-budgeted") {
    options.max_batch_workspace_bytes = 0;
    if (cache_mode == "batch-budgeted") {
      std::size_t minimum{};
      check(cuda_wb97mv_jk_workspace(shape, shells, options, minimum));
      options.max_workspace_bytes = minimum;
      options.max_batch_workspace_bytes = 128ULL << 20;
    }
  }
  if (cache_mode == "fresh-batch-budgeted") {
    std::size_t full{}, unbatched{};
    check(cuda_wb97mv_jk_workspace(shape, shells, options, full));
    auto scalar = options;
    scalar.max_batch_workspace_bytes = 0;
    check(cuda_wb97mv_jk_workspace(shape, shells, scalar, unbatched));
    require(full > unbatched, "fixture needs an optional full batch");
    // Full batches can keep their metadata and shrink only projection scratch.
    // Find the true refusal boundary rather than guessing from their full size.
    auto cap = options;
    cap.source_tile = 0;
    auto lower = unbatched, upper = full;
    while (upper - lower > 1) {
      cap.max_workspace_bytes = lower + (upper - lower) / 2;
      std::size_t planned{};
      check(cuda_wb97mv_jk_workspace(shape, shells, cap, planned));
      if (planned > unbatched)
        upper = cap.max_workspace_bytes;
      else
        lower = cap.max_workspace_bytes;
    }
    options.max_workspace_bytes = lower;
    options.source_tile = 2;
    options.prefer_fresh_direction_batches = true;
  }
  std::size_t bytes{};
  check(cuda_wb97mv_jk_workspace(shape, shells, options, bytes));
  const auto ao_offset = 3 * shape.atoms + 2 * shape.primitives;
  if (header[6] == 1) basis.packed[3 * shape.atoms] = -1;
  if (header[6] == 2) basis.packed[ao_offset] = shape.atoms;
  if (header[6] == 3) basis.packed[ao_offset + 3] = 4;
  if (header[6] == 4) basis.packed.back() = std::numeric_limits<double>::quiet_NaN();
  if (header[6] == 5) basis.packed[ao_offset + 4] = 3;
  if (header[6] == 6) basis.packed[ao_offset + 3] = std::numeric_limits<double>::quiet_NaN();
  Storage storage;
  Owner owner;
  int device{-1};
  check(cudaGetDevice(&device));
  const CudaWb97mvJkReference reference{storage.upload(basis.packed), device, storage.stream,
                                        Owner::authenticate, &owner};
  auto* arena = storage.allocate(bytes / sizeof(double));
  auto* motion = storage.upload(direction);
  auto* density = storage.upload(densities);
  auto* delta = density + panels;
  const std::array<std::size_t, 5> counts{1, panels, panels, 3 * shape.atoms, 3 * shape.atoms};
  std::array<double*, 5> pointers;
  for (unsigned index = 0; index < 5; ++index) pointers[index] = storage.allocate(counts[index]);
  const auto outputs = [](const auto& buffers) {
    return CudaWb97mvJkOutput{buffers[0], buffers[1], buffers[2], buffers[3], buffers[4]};
  };
  const auto output = outputs(pointers);
  if (cache_mode == "screened-table" || cache_mode == "indexed-table") {
    screened_table(system, shape, shells, options, reference, storage, motion, density, delta,
                   output, cache_mode == "indexed-table");
    return;
  }
  CudaWb97mvNuclearJk response;
  require(response.apply(motion, density, delta, output, storage.stream) ==
              cudaErrorInvalidResourceHandle,
          "unprepared action succeeded");
  require(response.prepare(shape, shells, options, reference, arena, bytes - 1) ==
              cudaErrorInvalidValue,
          "short arena admitted");
  const auto prepared = response.prepare(shape, shells, options, reference, arena, bytes);
  if (header[6]) {
    require(prepared == cudaErrorInvalidValue &&
                response.apply(motion, density, delta, output, storage.stream) ==
                    cudaErrorInvalidResourceHandle,
            "malformed basis admitted");
    std::cout << "{\"prepare_rejected\":true}\n";
    return;
  }
  check(prepared);
  check(response.apply(motion, density, delta, output, storage.stream));
  const auto receipt = response.work();
  require(receipt.completed_actions == 1 &&
              receipt.component_evaluations == receipt.component_evaluations_per_action &&
              receipt.projection_terms_bound == receipt.projection_terms_bound_per_action &&
              receipt.reduction_terms == receipt.reduction_terms_per_action,
          "semantic work receipt mismatch");
  std::ofstream capture(result, std::ios::binary);
  for (unsigned index = 0; index < 5; ++index) {
    const auto values = download(pointers[index], counts[index]);
    require(static_cast<bool>(capture.write(reinterpret_cast<const char*>(values.data()),
                                            values.size() * sizeof(double))),
            "output write failed");
  }
  const auto expected = [&] {
    std::array<std::vector<double>, 5> values;
    for (unsigned index = 0; index < 5; ++index)
      values[index] = download(pointers[index], counts[index]);
    return values;
  }();
  auto* identical_direction = storage.upload(direction);
  check(response.apply(identical_direction, density, delta, output, storage.stream));
  require(response.work().resident_integral_hits > 0 || !receipt.resident_integral_bytes,
          "resident direction was keyed by pointer rather than values");
  for (unsigned index = 0; index < 5; ++index)
    require(download(pointers[index], counts[index]) == expected[index],
            "same direction changed numerical output");

  auto streamed = options;
  streamed.max_resident_integral_bytes = 0;
  std::size_t streamed_bytes{};
  check(cuda_wb97mv_jk_workspace(shape, shells, streamed, streamed_bytes));
  CudaWb97mvNuclearJk independent_schedule;
  check(independent_schedule.prepare(shape, shells, streamed, reference,
                                     storage.allocate(streamed_bytes / sizeof(double)),
                                     streamed_bytes));
  std::array<double*, 5> alternate;
  for (unsigned index = 0; index < 5; ++index) alternate[index] = storage.allocate(counts[index]);
  const auto compare_streamed = [&] {
    check(independent_schedule.apply(motion, density, delta, outputs(alternate), storage.stream));
    for (unsigned index = 0; index < 5; ++index) {
      const auto actual = download(pointers[index], counts[index]);
      const auto expected_streamed = download(alternate[index], counts[index]);
      for (std::size_t element = 0; element < counts[index]; ++element)
        require(std::isfinite(actual[element]) && std::isfinite(expected_streamed[element]) &&
                    std::abs(actual[element] - expected_streamed[element]) <=
                        3e-12 * (1 + std::abs(expected_streamed[element])),
                "resident and streaming schedules disagree");
    }
  };
  auto changed_densities = densities;
  for (std::size_t index = 0; index < panels; ++index) {
    changed_densities[index] = 0.7 * densities[index] + 0.2 * densities[panels + index];
    changed_densities[panels + index] = -0.3 * densities[index] + 0.4 * densities[panels + index];
  }
  check(cudaMemcpy(density, changed_densities.data(), changed_densities.size() * sizeof(double),
                   cudaMemcpyHostToDevice));
  check(response.apply(motion, density, delta, output, storage.stream));
  compare_streamed();
  auto changed_direction = direction;
  for (auto& value : changed_direction) value *= -1.3;
  check(cudaMemcpy(motion, changed_direction.data(), changed_direction.size() * sizeof(double),
                   cudaMemcpyHostToDevice));
  check(response.apply(motion, density, delta, output, storage.stream));
  compare_streamed();
  check(cudaMemcpy(motion, direction.data(), direction.size() * sizeof(double),
                   cudaMemcpyHostToDevice));
  check(cudaMemcpy(density, densities.data(), densities.size() * sizeof(double),
                   cudaMemcpyHostToDevice));
  check(response.apply(motion, density, delta, output, storage.stream));
  compare_streamed();
  const auto sentinel = [&] {
    for (unsigned index = 0; index < 5; ++index) {
      const std::vector<double> values(counts[index], 27);
      check(cudaMemcpy(pointers[index], values.data(), values.size() * sizeof(double),
                       cudaMemcpyHostToDevice));
    }
  };
  const auto unchanged = [&] {
    for (unsigned index = 0; index < 5; ++index)
      require(download(pointers[index], counts[index]) == std::vector<double>(counts[index], 27),
              "failure published partial output");
  };
  sentinel();
  require(response.apply(motion, density, delta, output, nullptr) == cudaErrorInvalidResourceHandle,
          "foreign stream admitted");
  unchanged();
  for (unsigned first = 0; first < 5; ++first)
    for (unsigned second = 0; second < first; ++second) {
      auto aliased = pointers;
      aliased[first] = aliased[second];
      require(response.apply(motion, density, delta, outputs(aliased), storage.stream) ==
                  cudaErrorInvalidValue,
              "output alias admitted");
      unchanged();
    }
  for (const auto* input_pointer :
       {motion, density, delta, arena, const_cast<double*>(reference.basis)}) {
    auto aliased = pointers;
    aliased[0] = const_cast<double*>(input_pointer);
    require(response.apply(motion, density, delta, outputs(aliased), storage.stream) ==
                cudaErrorInvalidValue,
            "input/arena output alias admitted");
    unchanged();
  }
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  check(cudaMemcpy(motion, &nan, sizeof(double), cudaMemcpyHostToDevice));
  require(response.apply(motion, density, delta, output, storage.stream) == cudaErrorInvalidValue,
          "nonfinite direction admitted");
  unchanged();
  auto* transaction = reinterpret_cast<int*>(storage.allocate(1));
  const auto private_action = [&](std::size_t frozen_column = SIZE_MAX) {
    check(cudaMemsetAsync(transaction, 0, sizeof(int), storage.stream));
    check(response.enqueue_private_fresh(motion, density, delta, output, transaction,
                                         storage.stream, frozen_column));
    int status{};
    check(cudaMemcpyAsync(&status, transaction, sizeof(status), cudaMemcpyDeviceToHost,
                          storage.stream));
    check(cudaStreamSynchronize(storage.stream));
    return status;
  };
  require(private_action() != 0, "private exact action lost a rejected direction audit");
  unchanged();
  check(cudaMemcpy(motion, direction.data(), direction.size() * sizeof(double),
                   cudaMemcpyHostToDevice));
  require(private_action() == 0, "private exact failure contaminated a valid replay");
  std::array<std::vector<double>, 5> private_output;
  for (unsigned index = 0; index < 5; ++index)
    private_output[index] = download(pointers[index], counts[index]);
  check(response.apply(motion, density, delta, output, storage.stream));
  for (unsigned index = 0; index < 5; ++index)
    require(private_output[index] == download(pointers[index], counts[index]),
            "private exact action changed the checked output");
  sentinel();
  require(private_action(0) != 0, "frozen column accepted an arbitrary direction");
  unchanged();
  std::vector<double> unit(direction.size());
  unit.front() = 1;
  check(cudaMemcpy(motion, unit.data(), unit.size() * sizeof(double), cudaMemcpyHostToDevice));
  check(cudaMemsetAsync(delta, 0, panels * sizeof(double), storage.stream));
  const auto before = response.work().component_evaluations;
  require(private_action(0) == 0, "frozen column rejected valid stationary weights");
  const auto frozen_components = response.work().component_evaluations - before;
  const auto frozen_fock = download(pointers[2], counts[2]);
  const auto frozen_hvp = download(pointers[4], counts[4]);
  for (const unsigned index : {0, 1, 3})
    require(download(pointers[index], counts[index]) == std::vector<double>(counts[index], 27),
            "frozen column overwrote a primal output");
  check(response.apply(motion, density, delta, output, storage.stream));
  for (auto index : {2, 4}) {
    const auto& frozen = index == 2 ? frozen_fock : frozen_hvp;
    const auto full = download(pointers[index], counts[index]);
    for (std::size_t element = 0; element < full.size(); ++element)
      require(std::abs(full[element] - frozen[element]) <= 1e-11 * (1 + std::abs(full[element])),
              "frozen column changed a directional output");
  }
  if (response.work().batched_workspace_bytes)
    require(frozen_components < response.work().component_evaluations - before - frozen_components,
            "frozen column evaluated inactive primitive components");
  if (cache_mode == "fresh-streamed" || cache_mode == "fresh-retained") {
    const auto coordinates = unit.size();
    const auto table_size = coordinates * (panels + coordinates);
    auto* table = storage.allocate(table_size);
    const auto dense = [&] {
      return response.enqueue_private_frozen_dense(motion, density, delta, table, transaction,
                                                   storage.stream);
    };
    check(cudaMemsetAsync(transaction, 0, sizeof(int), storage.stream));
    if (cache_mode == "fresh-retained") {
      require(dense() == cudaErrorNotSupported, "retained layout changed to source-major");
    } else {
      const auto initial_work = response.work().component_evaluations;
      check(dense());
      const auto dense_work = response.work().component_evaluations - initial_work;
      int audit{};
      check(cudaMemcpyAsync(&audit, transaction, sizeof(audit), cudaMemcpyDeviceToHost,
                            storage.stream));
      check(cudaStreamSynchronize(storage.stream));
      require(audit == 0, "frozen source sweep rejected valid weights");
      const auto values = download(table, table_size);
      const auto replay_start = response.work().component_evaluations;
      for (std::size_t column = 0; column < coordinates; ++column) {
        for (std::size_t coordinate = 0; coordinate < coordinates; ++coordinate)
          unit[coordinate] = static_cast<double>(column == coordinate);
        check(
            cudaMemcpy(motion, unit.data(), coordinates * sizeof(double), cudaMemcpyHostToDevice));
        require(private_action(column) == 0, "frozen replay rejected valid weights");
        const auto fock = download(pointers[2], panels);
        const auto hvp = download(pointers[4], coordinates);
        const auto compares = [](double expected, double actual) {
          return std::abs(expected - actual) <= 3e-12 * (1 + std::abs(expected));
        };
        require(
            std::equal(fock.begin(), fock.end(), values.begin() + column * panels, compares) &&
                std::equal(hvp.begin(), hvp.end(),
                           values.begin() + coordinates * panels + column * coordinates, compares),
            "relative source sweep changed a raw column");
      }
      require(dense_work < response.work().component_evaluations - replay_start,
              "relative source sweep did not elide primitive components");
    }
  }
  check(cudaMemcpy(motion, direction.data(), direction.size() * sizeof(double),
                   cudaMemcpyHostToDevice));
  check(cudaMemcpy(delta, densities.data() + panels, panels * sizeof(double),
                   cudaMemcpyHostToDevice));
  sentinel();
  for (unsigned index = 0; index < 2 * shape.spins; ++index) {
    const auto offset = index * shape.nao * shape.nao;
    check(cudaMemcpy(density + offset, &nan, sizeof(double), cudaMemcpyHostToDevice));
    require(response.apply(motion, density, delta, output, storage.stream) == cudaErrorInvalidValue,
            "nonfinite density admitted");
    unchanged();
    check(cudaMemcpy(density + offset, densities.data() + offset, sizeof(double),
                     cudaMemcpyHostToDevice));
    if (shape.nao > 1) {
      const auto asymmetric = densities[offset + 1] + 1;
      check(cudaMemcpy(density + offset + 1, &asymmetric, sizeof(double), cudaMemcpyHostToDevice));
      require(
          response.apply(motion, density, delta, output, storage.stream) == cudaErrorInvalidValue,
          "asymmetric density admitted");
      unchanged();
      check(cudaMemcpy(density + offset + 1, densities.data() + offset + 1, sizeof(double),
                       cudaMemcpyHostToDevice));
    }
  }
  const auto enormous = std::numeric_limits<double>::max();
  check(cudaMemcpy(density, &enormous, sizeof(double), cudaMemcpyHostToDevice));
  require(response.apply(motion, density, delta, output, storage.stream) == cudaErrorInvalidValue,
          "overflowing finite contraction published output");
  unchanged();
  check(cudaMemcpy(density, densities.data(), sizeof(double), cudaMemcpyHostToDevice));
  check(cudaStreamBeginCapture(storage.stream, cudaStreamCaptureModeGlobal));
  const auto captured = response.apply(motion, density, delta, output, storage.stream);
  cudaGraph_t graph{};
  const auto ended = cudaStreamEndCapture(storage.stream, &graph);
  if (graph) cudaGraphDestroy(graph);
  require(captured == cudaErrorStreamCaptureUnsupported && ended == cudaSuccess,
          "capture admission changed stream");
  unchanged();
  owner.remaining = 1;
  require(response.apply(motion, density, delta, output, storage.stream) ==
              cudaErrorInvalidResourceHandle,
          "late owner revocation admitted");
  unchanged();
  owner.remaining = UINT_MAX;
  check(response.prepare(shape, shells, options, reference, arena, bytes));
  owner.live = false;
  require(response.apply(motion, density, delta, output, storage.stream) ==
              cudaErrorInvalidResourceHandle,
          "stale owner admitted");
  unchanged();
  owner.live = true;
  check(response.prepare(shape, shells, options, reference, arena, bytes));
  check(response.apply(motion, density, delta, output, storage.stream));
  sentinel();
  require(response.prepare(shape, shells, options, reference, arena, bytes - 1) ==
                  cudaErrorInvalidValue &&
              response.apply(motion, density, delta, output, storage.stream) ==
                  cudaErrorInvalidResourceHandle,
          "failed reprepare retained old binding");
  unchanged();
  std::cout << "{\"lifecycle\":true,\"bytes\":" << receipt.workspace_bytes
            << ",\"groups\":" << receipt.groups
            << ",\"records\":" << receipt.primitive_records_per_action
            << ",\"components\":" << receipt.component_evaluations
            << ",\"resident_bytes\":" << receipt.resident_integral_bytes
            << ",\"retained_source_bytes\":" << receipt.retained_window_source_bytes
            << ",\"batch_bytes\":" << receipt.batched_workspace_bytes
            << ",\"streaming_tiles\":" << receipt.streaming_tiles
            << ",\"source_groups\":" << receipt.source_groups
            << ",\"integral_launches\":" << receipt.integral_launches
            << ",\"projection_launches\":" << receipt.projection_launches
            << ",\"reduction_launches\":" << receipt.reduction_launches
            << ",\"projections\":" << receipt.projection_terms_bound
            << ",\"reductions\":" << receipt.reduction_terms << "}\n";
}
}  // namespace
int main(int argc, char** argv) {
  try {
    require(argc == 3 || argc == 4, "usage: jk-probe source.bin output.bin [cache-mode]");
    const std::string mode = argc == 4 ? argv[3] : "resident";
    require(mode == "resident" || mode == "streamed" || mode == "budgeted" || mode == "unbatched" ||
                mode == "batch-budgeted" || mode == "batch-chunked" ||
                mode == "projection-budgeted" || mode == "tiled" || mode == "resident-cap-tiled" ||
                mode == "resident-cap-fallback" || mode == "fresh-batch-budgeted" ||
                mode == "fresh-retained" || mode == "fresh-streamed" || mode == "screened-table" ||
                mode == "indexed-table",
            "invalid cache mode");
    run(argv[1], argv[2], mode);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
