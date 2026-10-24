// CUDA allocation/execution owner. Mathematics, tiling and exact device layout
// come from the scientific compiler; the scalar reference remains in grid.cpp.
#include <algorithm>
#include <atomic>
#include <limits>
#include <map>
#include <stdexcept>

#include "dft/grid.hpp"
#include "generated_quadrature.cuh"
#include "runtime/cuda_resources.cuh"

namespace generativeqc::dft {
namespace q = generated::quadrature;
namespace {

std::uint64_t next_grid_owner() noexcept {
  static std::atomic<std::uint64_t> next{1};
  auto value = next.load(std::memory_order_relaxed);
  while (value && value != std::numeric_limits<std::uint64_t>::max()) {
    if (next.compare_exchange_weak(value, value + 1, std::memory_order_relaxed)) return value;
  }
  return 0;
}

}  // namespace

std::size_t cuda_resident_grid_bytes(std::size_t points, std::size_t topology_atoms) {
  if (!points) throw std::invalid_argument("invalid CUDA resident-grid shape");
  const auto topology = topology_atoms ? q::sum(q::product(2, topology_atoms), 1) : 0;
  return q::product(q::sum(q::product(5, points), topology), sizeof(double));
}

std::size_t cuda_quadrature_bytes(std::size_t atoms, std::size_t points) {
  const auto layout = q::layout(atoms, points);
  // Shape-only admission covers every supported rule family. Unique-element
  // xyz/weight tables cannot exceed the complete atom-expanded raw grid.
  const auto rules = q::product(4, points);
  const auto metadata = q::sum(q::product(atoms, atoms), q::sum(q::product(2, atoms), 1));
  return q::sum(q::sum(layout.device_bytes, cuda_resident_grid_bytes(points, atoms)),
                q::product(q::sum(rules, metadata), sizeof(double)));
}

MolecularGrid MolecularGrid::from_cuda(const core::System& system, GridSpec spec, int device,
                                       bool retain_device) {
  MolecularGrid result(system, spec, Deferred{});
  const bool treutler = spec.version == 4;
  const auto per_atom = treutler ? 0 : grid_points_per_atom(spec);
  const auto l = q::layout(system.atoms.size(), grid_point_count(system, spec));
  const auto [rn, rw] = legendre_rule(treutler ? 1 : spec.radial_points);
  const auto [zn, zw] = legendre_rule(treutler ? 1 : spec.angular_polar);
  // These small shared quadrature tables are input rules, not a CPU molecular
  // grid. The generated kernels own coordinates and every partition weight.
  std::vector<double> input(l.radial, 0.0);
  for (std::size_t a = 0; a < l.atoms; ++a) {
    std::copy(system.atoms[a].position.begin(), system.atoms[a].position.end(),
              input.begin() + 3 * a);
    const double radius = spec.element_radii[system.atoms[a].atomic_number];
    input[3 * l.atoms + a] = radius > 0.0 ? radius : 1.0;
  }
  std::copy(rn.begin(), rn.end(), input.begin() + l.rules);
  std::copy(rw.begin(), rw.end(), input.begin() + l.rules + 512);
  std::copy(zn.begin(), zn.end(), input.begin() + l.rules + 1024);
  std::copy(zw.begin(), zw.end(), input.begin() + l.rules + 1280);
  q::PrunedShells pruned{};
  std::size_t polar_entries = spec.angular_polar;
  std::vector<std::pair<std::size_t, std::size_t>> azimuth_rules{{spec.angular_azimuth, 0}};
  if (spec.version == 3) {
    const auto shells = grid_angular_shells(spec);
    std::map<std::pair<std::size_t, std::size_t>, std::pair<std::size_t, std::size_t>> rules;
    std::size_t azimuth_entries = 0, point_begin = 0;
    polar_entries = 0;
    azimuth_rules.clear();
    pruned.points_per_atom = per_atom;
    for (std::size_t radial = 0; radial < shells.size(); ++radial) {
      const auto shape = shells[radial];
      if (!rules.contains(shape)) {
        const auto [nodes, weights] = legendre_rule(shape.first);
        std::copy(nodes.begin(), nodes.end(), input.begin() + l.rules + 1024 + polar_entries);
        std::copy(weights.begin(), weights.end(), input.begin() + l.rules + 1280 + polar_entries);
        rules.emplace(shape, std::pair{polar_entries, azimuth_entries});
        azimuth_rules.emplace_back(shape.second, azimuth_entries);
        polar_entries += shape.first;
        azimuth_entries += shape.second;
      }
      if (radial == 0 || shape != shells[radial - 1]) {
        const auto band = pruned.bands++;
        if (band >= 5) throw std::logic_error("radial-v1 exceeded its bounded band count");
        pruned.radial_begin[band] = radial;
        pruned.point_begin[band] = point_begin;
        pruned.polar_count[band] = shape.first;
        pruned.azimuth_count[band] = shape.second;
        pruned.polar_offset[band] = rules.at(shape).first;
        pruned.azimuth_offset[band] = rules.at(shape).second;
      }
      point_begin += q::product(shape.first, shape.second);
    }
    pruned.point_begin[pruned.bands] = point_begin;
    if (polar_entries > 256 || azimuth_entries > 1024 || point_begin != per_atom)
      throw std::logic_error("radial-v1 exceeded its bounded angular tables");
  }
  result.points_.resize(q::product(3, l.points));
  result.weights_.resize(l.points);
  result.owners_.resize(l.points);
  std::vector<uint64_t> atom_offsets{0}, rule_offsets;
  std::vector<double> atomic_rules, adjustments;
  std::map<unsigned, std::pair<size_t, size_t>> element_rules;
  if (treutler) {
    adjustments.resize(q::product(l.atoms, l.atoms));
    // Reserve a finite atom-expanded upper bound, preventing vector-growth
    // peaks from escaping host preparation admission.
    atomic_rules.reserve(q::product(4, l.points));
  }
  for (std::size_t a = 0; a < l.atoms; ++a) {
    const auto z = system.atoms[a].atomic_number;
    const auto count = grid_atom_point_count(spec, z);
    std::fill_n(result.owners_.begin() + atom_offsets.back(), count, static_cast<uint32_t>(a));
    atom_offsets.push_back(atom_offsets.back() + count);
    if (treutler) {
      if (!element_rules.contains(z)) {
        const auto rule = treutler_atomic_rule(z, spec.radial_points);
        element_rules.emplace(z, std::pair{atomic_rules.size() / 4, rule.size()});
        for (const auto& row : rule)
          atomic_rules.insert(atomic_rules.end(), row.begin(), row.end());
      }
      rule_offsets.push_back(element_rules.at(z).first);
      for (size_t b = 0; b < a; ++b)
        adjustments[a * l.atoms + b] =
            grid_radius_adjustment(spec, z, system.atoms[b].atomic_number);
    }
  }

  runtime::CudaDeviceScope guard(device);
  runtime::OwnedCudaStream stream(device);
  // Retain one immutable full-grid copy beside bounded generation scratch.
  // Its lifetime is independent of this private preparation stream: all writes
  // are ordered before the final stream drain below.
  std::shared_ptr<runtime::OwnedCudaBuffer<double>> resident;
  double *resident_points = nullptr, *resident_weights = nullptr,
         *resident_atomic_weights = nullptr;
  if (retain_device) {
    resident = std::make_shared<runtime::OwnedCudaBuffer<double>>(
        device, cuda_resident_grid_bytes(l.points, treutler ? l.atoms : 0) / sizeof(double));
    resident_points = resident->get();
    resident_weights = resident_points + q::product(3, l.points);
    resident_atomic_weights = resident_weights + l.points;
  }
  runtime::OwnedCudaBuffer<double> storage(device, l.doubles, stream.get());
  runtime::OwnedCudaBuffer<int> invalid(device, 1, stream.get());
  const auto check = runtime::cuda_resource_check;
  double* data = storage.get();
  check(cudaMemcpyAsync(data, input.data(), input.size() * sizeof(double), cudaMemcpyHostToDevice,
                        stream.get()));
  check(cudaMemsetAsync(invalid.get(), 0, sizeof(int), stream.get()));
  runtime::OwnedCudaBuffer<double> rule_storage, adjustment_storage;
  runtime::OwnedCudaBuffer<uint64_t> offset_storage;
  if (treutler) {
    rule_storage.allocate(device, atomic_rules.size(), stream.get());
    adjustment_storage.allocate(device, adjustments.size(), stream.get());
    offset_storage.allocate(device, 2 * l.atoms + 1, stream.get());
    if (resident) {
      auto* offsets = reinterpret_cast<uint64_t*>(resident_atomic_weights + l.points);
      auto* radii = reinterpret_cast<double*>(offsets + l.atoms + 1);
      check(cudaMemcpyAsync(offsets, atom_offsets.data(), (l.atoms + 1) * sizeof(uint64_t),
                            cudaMemcpyHostToDevice, stream.get()));
      check(cudaMemcpyAsync(radii, input.data() + 3 * l.atoms, l.atoms * sizeof(double),
                            cudaMemcpyHostToDevice, stream.get()));
      result.cuda_atom_offsets_ = offsets;
      result.cuda_atomic_radii_ = radii;
    }
    check(cudaMemcpyAsync(rule_storage.get(), atomic_rules.data(),
                          atomic_rules.size() * sizeof(double), cudaMemcpyHostToDevice,
                          stream.get()));
    check(cudaMemcpyAsync(adjustment_storage.get(), adjustments.data(),
                          adjustments.size() * sizeof(double), cudaMemcpyHostToDevice,
                          stream.get()));
    check(cudaMemcpyAsync(offset_storage.get(), atom_offsets.data(),
                          atom_offsets.size() * sizeof(uint64_t), cudaMemcpyHostToDevice,
                          stream.get()));
    check(cudaMemcpyAsync(offset_storage.get() + l.atoms + 1, rule_offsets.data(),
                          rule_offsets.size() * sizeof(uint64_t), cudaMemcpyHostToDevice,
                          stream.get()));
  } else {
    q::radial_kernel<<<q::blocks(l.atoms * spec.radial_points), 128, 0, stream.get()>>>(
        l.atoms, spec.radial_points, data + 3 * l.atoms, data + l.rules, data + l.rules + 512,
        data + l.radial);
    check(cudaGetLastError());
    q::polar_kernel<<<q::blocks(polar_entries), 128, 0, stream.get()>>>(
        polar_entries, data + l.rules + 1024, data + l.rules + 1280, data + l.polar);
    check(cudaGetLastError());
    for (const auto& [count, offset] : azimuth_rules) {
      q::azimuth_kernel<<<q::blocks(count), 128, 0, stream.get()>>>(count,
                                                                    data + l.azimuth + 3 * offset);
      check(cudaGetLastError());
    }
  }
  q::geometry_kernel<<<q::blocks(l.atoms * l.atoms), 128, 0, stream.get()>>>(
      data, l.atoms, spec.coincident_tolerance, data + l.geometry);
  check(cudaGetLastError());
  for (std::size_t begin = 0; begin < l.points; begin += l.tile) {
    const auto count = std::min(l.tile, l.points - begin);
    if (treutler) {
      q::atomic_points_kernel<<<q::blocks(count), 128, 0, stream.get()>>>(
          begin, count, l.atoms, offset_storage.get(), offset_storage.get() + l.atoms + 1,
          rule_storage.get(), data, data + l.xyz, data + l.weights);
    } else if (spec.version == 3) {
      q::pruned_points_kernel<<<q::blocks(count), 128, 0, stream.get()>>>(
          begin, count, spec.radial_points, pruned, data, data + l.radial, data + l.polar,
          data + l.azimuth, data + l.xyz, data + l.weights);
    } else {
      q::points_kernel<<<q::blocks(count), 128, 0, stream.get()>>>(
          begin, count, spec.radial_points, spec.angular_polar, spec.angular_azimuth, data,
          data + l.radial, data + l.polar, data + l.azimuth, data + l.xyz, data + l.weights);
    }
    check(cudaGetLastError());
    q::distances_kernel<<<q::atom_point_grid(count, l.atoms), 128, 0, stream.get()>>>(
        count, l.atoms, data + l.xyz, data, data + l.distances);
    check(cudaGetLastError());
    q::launch_partition(spec.partition_iterations, count, l.atoms, data + l.distances,
                        data + l.geometry, data + l.logs, stream.get(), adjustment_storage.get());
    check(cudaGetLastError());
    // points_kernel publishes the owner-local radial/angular measure before
    // Becke normalization overwrites data+l.weights. Preserve that exact raw
    // measure in the immutable resident owner for moving-grid force consumers.
    if (resident)
      check(cudaMemcpyAsync(resident_atomic_weights + begin, data + l.weights,
                            count * sizeof(double), cudaMemcpyDeviceToDevice, stream.get()));
    q::normalize_kernel<<<q::blocks(count), 128, 0, stream.get()>>>(
        begin, count, per_atom, l.atoms, data + l.logs, data + l.weights, invalid.get(),
        offset_storage.get());
    check(cudaGetLastError());
    // Preserve host exports for current derivative/reference APIs while also
    // publishing the exact generated values into one immutable full-grid
    // device owner. Downstream CUDA consumers borrow this owner directly and
    // therefore never re-upload the host copies.
    if (resident) {
      check(cudaMemcpyAsync(resident_points + 3 * begin, data + l.xyz, 3 * count * sizeof(double),
                            cudaMemcpyDeviceToDevice, stream.get()));
      check(cudaMemcpyAsync(resident_weights + begin, data + l.weights, count * sizeof(double),
                            cudaMemcpyDeviceToDevice, stream.get()));
    }
    check(cudaMemcpyAsync(result.points_.data() + 3 * begin, data + l.xyz,
                          3 * count * sizeof(double), cudaMemcpyDeviceToHost, stream.get()));
    check(cudaMemcpyAsync(result.weights_.data() + begin, data + l.weights, count * sizeof(double),
                          cudaMemcpyDeviceToHost, stream.get()));
  }
  int bad = 0;
  check(cudaMemcpyAsync(&bad, invalid.get(), sizeof(int), cudaMemcpyDeviceToHost, stream.get()));
  stream.synchronize();
  if (bad) throw std::runtime_error("invalid CUDA Becke partition normalization");
  if (resident) {
    const auto owner = next_grid_owner();
    if (!owner) throw std::overflow_error("CUDA molecular-grid owner identity exhausted");
    result.cuda_storage_ = resident;
    result.cuda_points_ = resident_points;
    result.cuda_weights_ = resident_weights;
    result.cuda_atomic_weights_ = resident_atomic_weights;
    result.cuda_device_bytes_ = cuda_resident_grid_bytes(l.points, treutler ? l.atoms : 0);
    result.cuda_owner_ = owner;
    result.cuda_device_ = device;
  }
  return result;
}
}  // namespace generativeqc::dft
