#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/cuda_grid_geometry.hpp"
#include "dft/cuda_wb97mv_grid.hpp"

namespace {
using namespace generativeqc::dft;
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
void check(cudaError_t error) { require(error == cudaSuccess, cudaGetErrorString(error)); }
struct Storage {
  cudaStream_t stream{};
  std::vector<double*> allocations;
  Storage() { check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
  ~Storage() {
    if (stream) cudaStreamSynchronize(stream);
    for (auto* pointer : allocations) cudaFree(pointer);
    if (stream) cudaStreamDestroy(stream);
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
std::vector<double> read(std::ifstream& input, std::size_t count) {
  require(count <= (1ULL << 26), "oversized probe fixture");
  std::vector<double> values(count);
  require(
      static_cast<bool>(input.read(reinterpret_cast<char*>(values.data()), count * sizeof(double))),
      "short probe fixture");
  return values;
}
std::vector<double> download(const double* pointer, std::size_t count) {
  std::vector<double> values(count);
  check(cudaMemcpy(values.data(), pointer, count * sizeof(double), cudaMemcpyDeviceToHost));
  return values;
}
struct Owner {
  Storage& storage;
  std::vector<const double*> immutable;
  int device{-1};
  bool live{true};
  static cudaError_t authenticate(void* context) noexcept {
    auto& owner = *static_cast<Owner*>(context);
    int device = -1;
    if (!owner.live || cudaGetDevice(&device) != cudaSuccess || device != owner.device)
      return cudaErrorInvalidResourceHandle;
    for (const auto* pointer : owner.immutable) {
      bool found = false;
      for (const auto* allocated : owner.storage.allocations) found = found || pointer == allocated;
      if (!found) return cudaErrorInvalidResourceHandle;
    }
    return cudaSuccess;
  }
};

void host_admission() {
  CudaGridGeometryOptions options;
  const CudaGridGeometryShape shape{3, 8, 4, 9, 3};
  std::size_t bytes{};
  check(cuda_grid_geometry_workspace(shape, options, bytes));
  require(bytes > 0, "empty admitted geometry arena");
  auto replay = options;
  replay.max_ao_replay_terms = 1ULL << 30;
  std::size_t resident_bytes{}, minimum{};
  check(cuda_grid_geometry_workspace(shape, replay, resident_bytes, &minimum));
  require(resident_bytes == bytes && minimum < resident_bytes, "missing bounded replay layout");
  replay.max_workspace_bytes = minimum;
  check(cuda_grid_geometry_workspace(shape, replay, bytes));
  require(bytes == minimum, "reported minimum cannot admit a geometry tile");
  --replay.max_workspace_bytes;
  require(cuda_grid_geometry_workspace(shape, replay, bytes) == cudaErrorInvalidValue,
          "geometry admitted a budget below its smallest tile");
  for (unsigned kind = 0; kind < 4; ++kind) {
    auto invalid = shape;
    if (kind == 0) invalid.points = 8;
    if (kind == 1) invalid.atoms = SIZE_MAX;
    if (kind == 2) invalid.primitives = SIZE_MAX;
    if (kind == 3) invalid.nao = SIZE_MAX;
    bytes = 27;
    require(
        cuda_grid_geometry_workspace(invalid, options, bytes) == cudaErrorInvalidValue && !bytes,
        "invalid/overflowed geometry shape admitted");
  }
  for (unsigned kind = 0; kind < 10; ++kind) {
    auto invalid = options;
    if (kind == 0) invalid.max_workspace_bytes = 1;
    if (kind == 1) invalid.max_ao_jet_values = 1;
    if (kind == 2) invalid.max_partition_pairs = 1;
    if (kind == 3) invalid.partition_iterations = 0;
    if (kind == 4) invalid.partition_iterations = 6;
    if (kind == 5) invalid.point_tile = 0;
    if (kind == 6) invalid.coincident_tolerance = -1;
    if (kind == 7) invalid.reference_absolute_tolerance = std::numeric_limits<double>::quiet_NaN();
    if (kind == 8) invalid.reference_relative_tolerance = -1;
    if (kind == 9) invalid.max_ao_primitive_terms = 1;
    bytes = 27;
    require(cuda_grid_geometry_workspace(shape, invalid, bytes) == cudaErrorInvalidValue && !bytes,
            "invalid geometry policy admitted");
  }
  options.max_partition_pairs = 0;
  check(cuda_grid_geometry_workspace({1, 2, 1, 3, 3}, options, bytes));
  CudaGridGeometryShape wide{96, 288, 96, 82944, 864, 288, true};
  CudaGridGeometryOptions wide_options;
  wide_options.max_workspace_bytes = 1024ULL << 20;
  require(cuda_grid_geometry_workspace(wide, wide_options, bytes) == cudaErrorInvalidValue,
          "compact AO storage incorrectly bypassed the Becke work limit");
  // Isolate AO admission without authorizing a large partition action.
  wide_options.max_partition_pairs = 378'224'640;
  check(cuda_grid_geometry_workspace(wide, wide_options, bytes));
  require(bytes < wide_options.max_workspace_bytes, "compact AO layout does not fit");
  wide.radial_s = false;
  require(cuda_grid_geometry_workspace(wide, wide_options, bytes) == cudaErrorInvalidValue,
          "expanded layout unexpectedly passed the wide AO ceilings");
  std::cout << "{\"host_admission_without_cuda_initialization\":true}\n";
}

void run(const char* fixture, const char* output, bool reject, std::size_t point_tile,
         const std::string& topology, bool replay, bool partial) {
  std::ifstream input(fixture, std::ios::binary);
  std::array<std::uint64_t, 6> header{};
  require(static_cast<bool>(input.read(reinterpret_cast<char*>(header.data()), sizeof(header))),
          "missing probe fixture header");
  CudaGridGeometryShape shape{header[0], header[1], header[2], header[3], header[5]};
  const bool tiled_cartesian = topology.find("tiled-cartesian") != std::string::npos;
  const bool quadrature_column = topology.starts_with("tiled-cartesian-quadrature");
  const bool sliced_column = topology == "tiled-cartesian-quadrature-sliced";
  CudaGridGeometryOptions options;
  options.point_tile = point_tile;
  options.partition_iterations = static_cast<unsigned>(header[4]);
  options.coincident_tolerance = read(input, 1)[0];
  if (topology == "tiled-cartesian-extended") {
    // This random dense fixture qualifies the derivatives rather than the
    // repair queue budget. Reserve all points for its scalar comparison.
    options.max_cartesian_repair_pairs =
        3 * shape.atoms * (shape.atoms * (shape.atoms - 1) / 2) * shape.points;
  }
  std::size_t bytes{};
  Storage storage;
  const auto basis = read(input, 3 * shape.atoms + 2 * shape.primitives + 16 * shape.nao);
  if (!topology.empty()) {
    shape.radial_s = topology.rfind("radial", 0) == 0;
    const auto offset = 3 * shape.atoms + 2 * shape.primitives;
    for (std::size_t index = 0; index < shape.nao; ++index)
      shape.ao_primitive_terms +=
          static_cast<std::size_t>(basis[offset + 16 * index + 2] * basis[offset + 16 * index + 3]);
    if (topology == "understated") --shape.ao_primitive_terms;
    options.max_ao_primitive_terms =
        (shape.radial_s ? 4 : 20) * shape.points * shape.ao_primitive_terms;
  }
  check(cuda_grid_geometry_workspace(shape, options, bytes));
  const auto points = read(input, 3 * shape.points);
  const auto weights = read(input, shape.points);
  const auto atomic = read(input, shape.points);
  const auto offsets = shape.points_per_atom ? std::vector<double>{} : read(input, shape.atoms + 1);
  const auto radii = shape.points_per_atom ? std::vector<double>{} : read(input, shape.atoms);
  const auto left = read(input, 3 * shape.atoms);
  const auto right = read(input, 3 * shape.atoms);
  require(input.peek() == std::char_traits<char>::eof(), "trailing probe fixture bytes");
  Owner owner{storage,
              {storage.upload(basis), storage.upload(points), storage.upload(weights),
               storage.upload(atomic)}};
  check(cudaGetDevice(&owner.device));
  CudaGridGeometryReference reference{
      owner.immutable[0], owner.immutable[1], owner.immutable[2],  owner.immutable[3],
      owner.device,       storage.stream,     Owner::authenticate, &owner};
  if (!shape.points_per_atom) {
    std::vector<std::uint64_t> prefix(offsets.begin(), offsets.end());
    auto* device_prefix = storage.allocate(prefix.size());
    check(cudaMemcpyAsync(device_prefix, prefix.data(), prefix.size() * sizeof(std::uint64_t),
                          cudaMemcpyHostToDevice, storage.stream));
    reference.atom_offsets = reinterpret_cast<std::uint64_t*>(device_prefix);
    reference.atomic_radii = storage.upload(radii);
    owner.immutable.push_back(device_prefix);
    owner.immutable.push_back(reference.atomic_radii);
  }
  double* prepared_pairs{};
  std::vector<double> center_table;
  if (topology == "prepared-centers") {
    // Independent scalar FP64 producer for the force table ABI. Ratio fields
    // are intentionally poisoned: only distance and its Cartesian slope are
    // reusable at mixed order.
    for (std::size_t first = 0; first < shape.atoms; ++first) {
      for (std::size_t second = 0; second < first; ++second) {
        std::array<double, 3> delta{};
        for (unsigned axis = 0; axis < 3; ++axis)
          delta[axis] = basis[3 * first + axis] - basis[3 * second + axis];
        const auto distance = std::hypot(delta[0], delta[1], delta[2]);
        center_table.push_back(distance);
        for (const auto component : delta) center_table.push_back(component / distance);
        center_table.insert(center_table.end(), 2, std::numeric_limits<double>::quiet_NaN());
      }
    }
    prepared_pairs = storage.upload(center_table);
    reference.center_pairs =
        reinterpret_cast<const generativeqc_grid_pair_geometry::CenterPair*>(prepared_pairs);
    owner.immutable.push_back(prepared_pairs);
  }
  auto* arena = storage.allocate(bytes / sizeof(double));
  auto* left_device = storage.upload(left);
  auto* right_device = storage.upload(right);
  const auto ao_count = 12 * shape.points * shape.nao;
  const auto weight_count = 3 * shape.points;
  auto* ao_output = storage.allocate(ao_count);
  auto* weight_output = storage.allocate(weight_count);
  CudaGridGeometryResponse response;
  const double* jets{};
  require(response.spatial_jets(jets) == cudaErrorInvalidResourceHandle && !jets,
          "unprepared geometry published jets");
  require(response.prepare(shape, options, reference, arena, bytes - 1) == cudaErrorInvalidValue,
          "geometry ignored arena capacity");
  auto insufficient = options;
  insufficient.max_ao_jet_values = 1;
  std::size_t invalid_bytes = 27;
  require(
      cuda_grid_geometry_workspace(shape, insufficient, invalid_bytes) == cudaErrorInvalidValue &&
          !invalid_bytes,
      "geometry ignored AO work capacity");
  if (shape.atoms > 1) {
    insufficient = options;
    insufficient.max_partition_pairs = 0;
    require(
        cuda_grid_geometry_workspace(shape, insufficient, invalid_bytes) == cudaErrorInvalidValue &&
            !invalid_bytes,
        "geometry ignored partition work capacity");
  }
  const auto prepared = response.prepare(shape, options, reference, arena, bytes);
  if (reject) {
    require(prepared == cudaErrorInvalidValue &&
                response.spatial_jets(jets) == cudaErrorInvalidResourceHandle && !jets,
            "geometry accepted a nonsmooth or unauthenticated primal source");
    std::cout << "{\"prepare_rejected\":true}\n";
    return;
  }
  check(prepared);
  std::vector<double> spatial;
  if (shape.radial_s) {
    require(response.spatial_jets(jets) == cudaErrorNotSupported && !jets,
            "compact geometry published a mislabelled expanded jet view");
  } else {
    check(response.spatial_jets(jets));
    spatial = download(jets, 20 * shape.points * shape.nao);
  }
  if (replay) {
    const auto resident_bytes = bytes;
    options.max_ao_replay_terms = 4 * response.work().prepared_ao_primitive_term_bound;
    std::size_t minimum{};
    check(cuda_grid_geometry_workspace(shape, options, bytes, &minimum));
    options.max_workspace_bytes = partial ? resident_bytes - 256 : minimum;
    check(cuda_grid_geometry_workspace(shape, options, bytes));
    require(bytes < resident_bytes, "replay did not reduce geometry storage");
    check(response.prepare(shape, options, reference, arena, bytes));
    require(response.spatial_jets(jets) == cudaErrorNotSupported && !jets,
            "replayed geometry published tile scratch as resident jets");
    check(response.admit_actions(1));
    if (partial) {
      check(response.admit_actions(5));
      require(response.work().prepared_ao_primitive_term_bound > 0,
              "partial arena retained no AO jets");
    } else {
      require(response.admit_actions(5) == cudaErrorInvalidValue,
              "geometry ignored endpoint replay work capacity");
    }
  }
  const double* point_directions{};
  require(response.point_directions(point_directions) == cudaErrorInvalidResourceHandle &&
              !point_directions,
          "geometry exposed motion before an action");
  check(response.mixed(left_device, right_device, ao_output, weight_output, storage.stream));
  check(response.point_directions(point_directions));
  const auto expected_motion = download(point_directions, 6 * shape.points);
  CudaGridGeometryMixedView view;
  if (replay) {
    check(response.mixed_view(left_device, right_device, view, storage.stream));
    require(download(view.ao, ao_count) == download(ao_output, ao_count) &&
                download(view.weights, weight_count) == download(weight_output, weight_count) &&
                download(view.point_directions, 6 * shape.points) == expected_motion,
            "borrowed geometry differs from copied results");
  }
  auto published_weights = download(weight_output, weight_count);
  if (topology == "cartesian" || topology == "radial-cartesian" || tiled_cartesian) {
    auto scalar_options = options;
    scalar_options.cache_cartesian_weights = false;
    if (tiled_cartesian) scalar_options.point_tile = CudaGridGeometryOptions{}.point_tile;
    auto scalar_shape = shape;
    scalar_shape.radial_s = false;
    if (shape.radial_s) scalar_options.max_ao_primitive_terms *= 5;
    std::size_t scalar_bytes{};
    check(cuda_grid_geometry_workspace(scalar_shape, scalar_options, scalar_bytes));
    CudaGridGeometryResponse scalar;
    check(scalar.prepare(scalar_shape, scalar_options, reference,
                         storage.allocate(scalar_bytes / sizeof(double)), scalar_bytes));
    auto* cartesian_status = reinterpret_cast<int*>(storage.allocate(1));
    auto* cartesian_left = storage.allocate(left.size());
    auto* cartesian_right = storage.allocate(right.size());
    const auto write_unit = [&](double* buffer, std::size_t column) {
      std::vector<double> unit(left.size());
      unit[column] = 1;
      check(cudaMemcpyAsync(buffer, unit.data(), unit.size() * sizeof(double),
                            cudaMemcpyHostToDevice, storage.stream));
      check(cudaStreamSynchronize(storage.stream));
    };
    const auto left_column = std::distance(left.begin(), std::find(left.begin(), left.end(), 1.));
    const auto right_column =
        std::distance(right.begin(), std::find(right.begin(), right.end(), 1.));
    require(left_column < static_cast<std::ptrdiff_t>(left.size()) &&
                right_column < static_cast<std::ptrdiff_t>(right.size()),
            "Cartesian fixture requires unit seeds");
    const auto audit = [&] {
      int audit_result{};
      check(cudaMemcpyAsync(&audit_result, cartesian_status, sizeof(audit_result),
                            cudaMemcpyDeviceToHost, storage.stream));
      check(cudaStreamSynchronize(storage.stream));
      return audit_result;
    };
    CudaGridGeometryCartesianView cartesian, baseline;
    CudaWb97mvGridResponse grid;
    double *density_motion{}, *cartesian_xc{};
    if (shape.radial_s || quadrature_column) {
      std::vector<double> density(shape.nao * shape.nao, 0);
      for (std::size_t orbital = 0; orbital < shape.nao; ++orbital)
        density[orbital * (shape.nao + 1)] = 0.65;
      CudaWb97mvGridShape grid_shape{shape.atoms, shape.primitives, shape.nao, shape.points};
      CudaWb97mvGridOptions grid_options;
      grid_options.functional_code = semilocal_family_code(SemilocalFamily::Pbe);
      grid_options.nuclear_response = true;
      grid_options.max_nuclear_adjoint_bytes = 64ULL << 20;
      if (sliced_column) grid_options.max_nuclear_affine_terms = 1ULL << 29;
      CudaWb97mvGridReference grid_reference{
          reference.basis,     reference.points, reference.weights,   storage.upload(density),
          reference.device_id, storage.stream,   Owner::authenticate, &owner};
      std::size_t grid_bytes{};
      check(cuda_wb97mv_grid_workspace(grid_shape, grid_options, grid_bytes));
      check(grid.prepare(grid_shape, grid_options, grid_reference,
                         storage.allocate(grid_bytes / sizeof(double)), grid_bytes));
      if (sliced_column) {
        auto no_adjoint = grid_options;
        no_adjoint.max_nuclear_adjoint_bytes = 0;
        std::size_t unusable{};
        require(
            cuda_wb97mv_grid_workspace(grid_shape, no_adjoint, unusable) == cudaErrorInvalidValue,
            "fixture must require a factored adjoint");
      }
      density_motion = storage.upload(std::vector<double>(density.size(), 0.0));
      cartesian_xc = storage.allocate(left.size());
      require(grid.admits_cartesian_mixed(left.size()), "radial XC column not admitted");
    }
    for (const auto right_seed : {quadrature_column ? static_cast<std::size_t>(left_column) : 0,
                                  static_cast<std::size_t>(right_column)}) {
      write_unit(cartesian_right, right_seed);
      check(cudaMemsetAsync(cartesian_status, 0, sizeof(int), storage.stream));
      CudaGridGeometryDirectionView rhs;
      const auto queued = response.enqueue_private_cartesian_column(
          cartesian_right, right_seed, cartesian, cartesian_status, storage.stream, &rhs);
      std::vector<double> column, right_weights, xc_column;
      if (queued == cudaErrorNotSupported) {
        require(
            !tiled_cartesian && shape.atoms > 3 && !cartesian.weights && !rhs.weights && !audit(),
            "column admission did not preserve the scalar fallback");
      } else {
        check(queued);
        require(cartesian.rows == left.size() && !audit(),
                "Cartesian column rejected a valid source");
        require(cartesian.radial_points == (shape.radial_s ? reference.points : nullptr),
                "Cartesian view does not identify its spatial representation");
        column = download(cartesian.weights, 2 * shape.points * left.size());
        right_weights = download(rhs.weights, shape.points);
        CudaGridGeometryMixedView diagonal;
        check(scalar.enqueue_private_mixed_view(cartesian_right, cartesian_right, diagonal,
                                                cartesian_status, storage.stream));
        require(!audit(), "scalar right direction rejected a valid source");
        const auto gathered = download(rhs.ao, ao_count / 3);
        const auto expanded = download(diagonal.ao + ao_count / 3, ao_count / 3);
        for (std::size_t index = 0; index < gathered.size(); ++index)
          require(std::abs(gathered[index] - expanded[index]) <=
                      (shape.radial_s ? 2e-13 * (1 + std::abs(expanded[index])) : 0),
                  "column RHS changed the first AO jets");
        require(download(rhs.point_directions, 3 * shape.points) ==
                    download(diagonal.point_directions + 3 * shape.points, 3 * shape.points),
                "column RHS changed the owner motion");
        if (shape.radial_s || quadrature_column) {
          check(grid.enqueue_private_nuclear_covector(rhs.ao, rhs.weights, rhs.point_directions,
                                                      density_motion, cartesian_status,
                                                      storage.stream));
          check(grid.enqueue_private_cartesian_mixed(cartesian, cartesian_xc, cartesian_status,
                                                     storage.stream));
          require(!audit(), "XC refused a Cartesian column");
          xc_column = download(cartesian_xc, left.size());
        }
      }
      for (std::size_t left_seed = 0; left_seed < left.size(); ++left_seed) {
        // Keep the scalar oracle bounded for the large fixture and for the
        // wide fused XC fixture. Cross each warp boundary and retain the
        // selected mixed row, diagonal, and first/last center's three axes.
        const auto left_atom = left_seed / 3;
        if (quadrature_column && left_seed != static_cast<std::size_t>(left_column) &&
            left_seed != static_cast<std::size_t>(right_column) &&
            !(sliced_column && (left_seed == 0 || left_seed == left.size() / 2)))
          continue;
        if (tiled_cartesian && (shape.radial_s || shape.atoms == cuda_grid_tiled_cartesian_atoms) &&
            left_atom != 0 && left_atom + 1 != shape.atoms && left_atom % 32 > 1 &&
            left_atom % 32 != 31 && left_seed != static_cast<std::size_t>(left_column) &&
            left_seed != right_seed)
          continue;
        write_unit(cartesian_left, left_seed);
        check(cudaMemsetAsync(cartesian_status, 0, sizeof(int), storage.stream));
        if (!tiled_cartesian) {
          check(response.enqueue_private_cartesian_view(cartesian_left, cartesian_right, left_seed,
                                                        right_seed, cartesian, cartesian_status,
                                                        storage.stream));
        }
        check(scalar.enqueue_private_cartesian_view(cartesian_left, cartesian_right, left_seed,
                                                    right_seed, baseline, cartesian_status,
                                                    storage.stream));
        require(!audit(), "Cartesian partition rejected a valid source");
        const auto expected = download(baseline.weights, weight_count);
        const auto cached = tiled_cartesian ? expected : download(cartesian.weights, weight_count);
        const bool capture_row = left_seed == static_cast<std::size_t>(left_column) &&
                                 right_seed == static_cast<std::size_t>(right_column);
        for (std::size_t index = 0; index < weight_count; ++index) {
          const auto point = index % shape.points, plane = index / shape.points;
          const auto tolerance = 3e-13 * (std::abs(atomic[point]) + std::abs(expected[index]));
          require(std::abs(cached[index] - expected[index]) <= tolerance,
                  "cached Cartesian row differs from scaled mixed jets");
          const double actual =
              column.empty()
                  ? cached[index]
                  : (plane == 1 ? right_weights[point]
                                : column[(2 * left_seed + (plane == 2)) * shape.points + point]);
          require(std::abs(actual - expected[index]) <= tolerance,
                  "compact Cartesian column differs from scaled mixed jets");
          if (capture_row) published_weights[index] = actual;
        }
        if (!xc_column.empty()) {
          CudaGridGeometryMixedView expanded;
          check(scalar.enqueue_private_mixed_view(cartesian_left, cartesian_right, expanded,
                                                  cartesian_status, storage.stream));
          check(grid.enqueue_private_nuclear_mixed(
              expanded.ao, expanded.ao + 2 * (ao_count / 3), expanded.weights,
              expanded.weights + 2 * shape.points, expanded.point_directions, cartesian_xc,
              cartesian_status, storage.stream));
          require(!audit(), "XC refused the expanded comparison row");
          const auto reference_row = download(cartesian_xc, 1)[0];
          require(std::abs(xc_column[left_seed] - reference_row) <=
                      3e-13 * (1 + std::abs(reference_row)),
                  "Cartesian XC contraction differs from expanded AO motion");
        }
      }
    }
    require(response.work().cartesian_partition_sweeps > 0,
            "fixture did not exercise the batched partition");
    if (topology == "tiled-cartesian-repair") {
      // Two extended measures force repair even when all denominator products
      // are representable. A full queue and a disabled queue must both fail
      // the private transaction without changing the separately published row.
      const auto repair_cost = left.size() * shape.atoms * (shape.atoms - 1) / 2;
      const auto original = download(weight_output, weight_count);
      for (const auto budget : {std::size_t{0}, repair_cost}) {
        auto limited_options = options;
        limited_options.max_cartesian_repair_pairs = budget;
        std::size_t limited_bytes{};
        check(cuda_grid_geometry_workspace(shape, limited_options, limited_bytes));
        CudaGridGeometryResponse limited;
        check(limited.prepare(shape, limited_options, reference,
                              storage.allocate(limited_bytes / sizeof(double)), limited_bytes));
        check(cudaMemsetAsync(cartesian_status, 0, sizeof(int), storage.stream));
        check(limited.enqueue_private_cartesian_column(cartesian_right, right_column, cartesian,
                                                       cartesian_status, storage.stream));
        require((audit() & cuda_grid_cartesian_repair_exhausted) &&
                    download(weight_output, weight_count) == original,
                "incomplete repair queue did not invalidate its private transaction");
        check(limited.mixed(left_device, right_device, ao_output, weight_output, storage.stream));
        require(download(weight_output, weight_count) == original,
                "repair overflow damaged the independent scaled fallback");
      }
    }
    CudaGridGeometryMixedView diagonal, scalar_diagonal;
    check(cudaMemsetAsync(cartesian_status, 0, sizeof(int), storage.stream));
    check(response.enqueue_private_cartesian_diagonal_view(cartesian_right, right_column, diagonal,
                                                           cartesian_status, storage.stream));
    check(scalar.enqueue_private_mixed_view(cartesian_right, cartesian_right, scalar_diagonal,
                                            cartesian_status, storage.stream));
    require(!audit() && (tiled_cartesian || response.work().cartesian_diagonal_sweeps > 0),
            "fixture did not exercise the diagonal Cartesian partition");
    const auto diagonal_ao = download(diagonal.ao, ao_count);
    const auto scalar_ao = download(scalar_diagonal.ao, ao_count);
    for (std::size_t index = 0; index < diagonal_ao.size(); ++index)
      require(std::abs(diagonal_ao[index] - scalar_ao[index]) <=
                  (shape.radial_s ? 2e-13 * (1 + std::abs(scalar_ao[index])) : 0),
              "diagonal partition altered the AO response");
    const auto actual_diagonal = download(diagonal.weights, weight_count);
    const auto expected_diagonal = download(scalar_diagonal.weights, weight_count);
    for (std::size_t index = 0; index < weight_count; ++index)
      require(
          std::abs(actual_diagonal[index] - expected_diagonal[index]) <=
              3e-13 * (std::abs(atomic[index % shape.points]) + std::abs(expected_diagonal[index])),
          "diagonal Cartesian partition differs from scaled mixed jets");
    check(cudaMemsetAsync(cartesian_status, 0, sizeof(int), storage.stream));
    check(response.enqueue_private_cartesian_view(cartesian_left, cartesian_right, 0, right_column,
                                                  cartesian, cartesian_status, storage.stream));
    require(audit() != 0, "Cartesian cache accepted a mismatched left direction");
    if (shape.atoms == 3 || tiled_cartesian) {
      check(cudaMemsetAsync(cartesian_status, 0, sizeof(int), storage.stream));
      check(response.enqueue_private_cartesian_column(cartesian_right, 0, cartesian,
                                                      cartesian_status, storage.stream));
      require(audit() != 0, "Cartesian column accepted a mismatched right direction");
    }
    if (topology == "tiled-cartesian-fold") {
      std::vector<double> energy(shape.points);
      for (std::size_t point = 0; point < energy.size(); ++point)
        energy[point] = std::cos(0.33 * point);
      const auto right_seed = static_cast<std::size_t>(right_column);
      const auto enqueue = [&] {
        write_unit(cartesian_right, right_seed);
        check(cudaMemsetAsync(cartesian_status, 0, sizeof(int), storage.stream));
        check(response.enqueue_private_cartesian_column(cartesian_right, right_seed, cartesian,
                                                        cartesian_status, storage.stream));
        require(!audit(), "folded fixture's private transaction failed");
      };
      enqueue();
      const auto original = download(cartesian.weights, 2 * shape.points * left.size());
      const auto fold_bytes = response.cartesian_weight_fold_workspace();
      require(fold_bytes > 0, "eligible weight fold did not fit");
      auto* scratch = storage.allocate(fold_bytes / sizeof(double));
      const auto energy_device = [&](const std::vector<double>& values) {
        auto* device = storage.allocate(values.size());
        check(cudaMemcpyAsync(device, values.data(), values.size() * sizeof(double),
                              cudaMemcpyHostToDevice, storage.stream));
        return device;
      };
      auto invalid = energy;
      invalid.front() = std::numeric_limits<double>::quiet_NaN();
      require(response.prepare_cartesian_weight_fold(energy_device(invalid), scratch, fold_bytes,
                                                     storage.stream) == cudaErrorNotSupported,
              "invalid cotangent did not retain the original Becke path");
      enqueue();
      require(!cartesian.folded_weight_column, "failed preparation published a folded view");
      const auto folding = response.prepare_cartesian_weight_fold(energy_device(energy), scratch,
                                                                  fold_bytes, storage.stream);
      if (folding != cudaSuccess) {
        std::cerr << "Becke fold refusal " << response.work().folded_numerical_refusal << '\n';
        check(folding);
      }
      enqueue();
      require(cartesian.folded_weight_column, "accepted fold did not publish a folded view");
      const auto first = download(cartesian.weights, shape.points * left.size());
      const auto mixed = download(cartesian.folded_weight_column, left.size());
      for (std::size_t row = 0; row < left.size(); ++row) {
        double weighted{};
        for (std::size_t point = 0; point < shape.points; ++point) {
          weighted += original[(2 * row + 1) * shape.points + point] * energy[point];
          const auto reference_first = original[2 * row * shape.points + point];
          require(std::abs(first[row * shape.points + point] - reference_first) <=
                      2e-12 * (1 + std::abs(reference_first)),
                  "folded first weight derivative differs from the per-column path");
        }
        require(std::abs(mixed[row] - weighted) <= 2e-12 * (1 + std::abs(weighted)),
                "folded mixed weight contraction differs from the per-column path");
      }
    }
  }
  std::ofstream capture(output, std::ios::binary);
  for (const auto& values :
       {spatial, download(ao_output, ao_count), published_weights, expected_motion})
    require(static_cast<bool>(capture.write(reinterpret_cast<const char*>(values.data()),
                                            values.size() * sizeof(double))),
            "probe output write failed");
  const auto measured_work = response.work();
  const auto sentinel_ao = std::vector<double>(ao_count, 27);
  const auto sentinel_weights = std::vector<double>(weight_count, 27);
  const auto sentinel = [&] {
    check(cudaMemcpyAsync(ao_output, sentinel_ao.data(), ao_count * sizeof(double),
                          cudaMemcpyHostToDevice, storage.stream));
    check(cudaMemcpyAsync(weight_output, sentinel_weights.data(), weight_count * sizeof(double),
                          cudaMemcpyHostToDevice, storage.stream));
    check(cudaStreamSynchronize(storage.stream));
  };
  const auto unchanged = [&] {
    require(download(ao_output, ao_count) == sentinel_ao &&
                download(weight_output, weight_count) == sentinel_weights,
            "failed geometry action published partial output");
    require(response.point_directions(point_directions) == cudaErrorInvalidResourceHandle &&
                !point_directions,
            "failed geometry action retained an expired motion lease");
  };
  sentinel();
  require(response.mixed(left_device, right_device, ao_output, weight_output, nullptr) ==
              cudaErrorInvalidResourceHandle,
          "geometry accepted another stream");
  unchanged();
  require(response.mixed(left_device, right_device, ao_output, ao_output, storage.stream) ==
              cudaErrorInvalidValue,
          "geometry accepted aliased outputs");
  unchanged();
  if (prepared_pairs) {
    require(response.mixed(left_device, right_device, ao_output, prepared_pairs, storage.stream) ==
                cudaErrorInvalidValue,
            "geometry accepted publication into borrowed center pairs");
    unchanged();
  }
  auto poisoned = left;
  poisoned[0] = std::numeric_limits<double>::quiet_NaN();
  check(cudaMemcpyAsync(left_device, poisoned.data(), poisoned.size() * sizeof(double),
                        cudaMemcpyHostToDevice, storage.stream));
  require(response.mixed(left_device, right_device, ao_output, weight_output, storage.stream) ==
              cudaErrorInvalidValue,
          "geometry accepted a nonfinite direction");
  unchanged();
  auto* transaction = reinterpret_cast<int*>(storage.allocate(1));
  const auto private_view_status = [&] {
    check(cudaMemsetAsync(transaction, 0, sizeof(int), storage.stream));
    check(response.enqueue_private_mixed_view(left_device, right_device, view, transaction,
                                              storage.stream));
    check(cudaStreamSynchronize(storage.stream));
    int status{};
    check(cudaMemcpy(&status, transaction, sizeof(status), cudaMemcpyDeviceToHost));
    unchanged();
    return status;
  };
  require(private_view_status() != 0, "private geometry did not report nonfinite motion");
  check(cudaMemcpyAsync(left_device, left.data(), left.size() * sizeof(double),
                        cudaMemcpyHostToDevice, storage.stream));
  require(private_view_status() == 0 &&
              download(view.point_directions, 6 * shape.points) == expected_motion,
          "private geometry did not recover after a failed action");
  check(cudaStreamSynchronize(storage.stream));
  check(cudaStreamBeginCapture(storage.stream, cudaStreamCaptureModeGlobal));
  const auto capture_status =
      response.mixed(left_device, right_device, ao_output, weight_output, storage.stream);
  cudaGraph_t graph{};
  check(cudaStreamEndCapture(storage.stream, &graph));
  check(cudaGraphDestroy(graph));
  require(capture_status == cudaErrorStreamCaptureUnsupported, "geometry accepted capture");
  unchanged();
  owner.live = false;
  require(response.mixed_view(left_device, right_device, view, storage.stream) ==
                  cudaErrorInvalidResourceHandle &&
              !view.ao && !view.weights && !view.point_directions,
          "geometry exposed borrowed panels after revocation");
  require(response.mixed(left_device, right_device, ao_output, weight_output, storage.stream) ==
                  cudaErrorInvalidResourceHandle &&
              response.spatial_jets(jets) == cudaErrorInvalidResourceHandle && !jets,
          "geometry accepted revoked storage");
  unchanged();
  const auto& work = measured_work;
  std::cout << "{\"capacity_alias_finite_capture_revocation_gates\":true,\"workspace_bytes\":"
            << work.workspace_bytes << ",\"prepared_ao_values\":" << work.prepared_ao_jet_values
            << ",\"prepared_partition_pairs\":" << work.prepared_partition_pairs
            << ",\"action_ao_values\":" << work.action_ao_jet_values
            << ",\"action_partition_pairs\":" << work.action_partition_pairs
            << ",\"tiled_cartesian_sweeps\":" << work.tiled_cartesian_sweeps
            << ",\"tiled_cartesian_repair_pair_bound\":" << work.tiled_cartesian_repair_pair_bound
            << ",\"ao_primitive_term_bound\":" << work.prepared_ao_primitive_term_bound
            << ",\"ao_replay_term_bound\":" << work.replayed_ao_primitive_term_bound << "}\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--host-check") {
      host_admission();
      return 0;
    }
    int devices{};
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    require(argc == 3 || argc == 4,
            "usage: cuda_grid_geometry_probe input output [reject|replay|partial|point-tile]");
    const auto partial = argc == 4 && std::string(argv[3]) == "partial";
    const std::string mode = argc == 4 ? argv[3] : "";
    const auto replay = partial || mode == "replay" || mode == "radial-replay";
    const std::string topology =
        argc == 4 && (std::string(argv[3]) == "topology" || std::string(argv[3]) == "understated" ||
                      std::string(argv[3]) == "prepared-centers" ||
                      std::string(argv[3]) == "cartesian" || mode.rfind("radial", 0) == 0 ||
                      mode.rfind("tiled-cartesian", 0) == 0)
            ? argv[3]
            : "";
    const auto reject = argc == 4 && (std::string(argv[3]) == "reject" ||
                                      topology == "understated" || topology == "radial-reject");
    // The fixtures exercise full tiles followed by a short tail.
    auto tile = CudaGridGeometryOptions{}.point_tile;
    if (mode.starts_with("tiled-cartesian") && !mode.starts_with("tiled-cartesian-quadrature"))
      tile = 7;
    else if (replay)
      tile = 64;
    else if (argc == 4 && !reject && topology.empty())
      tile = std::stoull(argv[3]);
    run(argv[1], argv[2], reject, tile, topology, replay, partial);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
