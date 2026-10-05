// Runtime-shaped generated map and strict provenance tests. Dense equations
// below are independent test oracles, never a production response provider.
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "generated_rhf_frame_response_cpu.hpp"
#include "hf/rhf_frame_response.hpp"

namespace {
using namespace generativeqc;
void require(bool value, const char* reason) {
  if (!value) throw std::runtime_error(reason);
}

void identity() {
  core::System system;
  system.electron_count = 2;
  system.atoms = {{2, {0., 0., 0.}, 0}};
  system.shells = {{0, 0, {{1., 1.}}}};
  hf::PhysicalReference ref;
  ref.nbf = 2;
  ref.nocc = 1;
  ref.energy = -1.;
  ref.coefficients = {1., 0., 0., 1.};
  ref.overlap = ref.hcore = ref.fock = ref.density = ref.weighted_density = ref.coefficients;
  ref.orbital_energies = {-1., 1.};
  hf::RHFFrameIdentity id(system, ref);
  require(id.matches(system, ref), "identical owned reference rejected");
  require(id.storage_bytes() == hf::RHFFrameIdentity::required_storage_bytes(system, ref),
          "identity payload not completely counted");
  for (auto member : {&hf::PhysicalReference::coefficients, &hf::PhysicalReference::overlap,
                      &hf::PhysicalReference::hcore, &hf::PhysicalReference::fock,
                      &hf::PhysicalReference::density, &hf::PhysicalReference::weighted_density,
                      &hf::PhysicalReference::orbital_energies}) {
    auto changed = ref;
    (changed.*member)[0] = std::nextafter((changed.*member)[0], 100.);
    require(!id.matches(system, changed), "one-ULP reference change admitted");
    (changed.*member).push_back(0.);
    require(!id.matches(system, changed), "reference length change admitted");
  }
  auto changed = ref;
  changed.energy = std::nextafter(ref.energy, 100.);
  require(!id.matches(system, changed), "energy change admitted");
  changed = ref;
  changed.coefficients[1] = -0.;
  require(!id.matches(system, changed), "signed-zero frame change admitted");
  changed = ref;
  changed.nocc = 2;
  require(!id.matches(system, changed), "occupation change admitted");
  auto molecule = system;
  molecule.atoms[0].position[0] = std::nextafter(0., 1.);
  require(!id.matches(molecule, ref), "one-ULP geometry change admitted");
  molecule = system;
  molecule.shells[0].primitives[0].exponent = std::nextafter(1., 2.);
  require(!id.matches(molecule, ref), "one-ULP basis change admitted");
  molecule = system;
  molecule.charge = 1;
  require(!id.matches(molecule, ref), "charge change admitted");
}

void generated(std::size_t o, std::size_t v, std::size_t q) {
  namespace maps = scf::generated::rhf_frame;
  std::vector<double> eo(o), ev(v), boo(q * o * o), bov(q * o * v), bvv(q * v * v);
  for (std::size_t i = 0; i < o; ++i) eo[i] = -1. - .1 * i;
  for (std::size_t a = 0; a < v; ++a) ev[a] = .2 + .1 * a;
  for (auto* array : {&boo, &bov, &bvv})
    for (std::size_t k = 0; k < array->size(); ++k) (*array)[k] = .03 * std::sin(double(k + 1));
  maps::PreconditionerInputs input;
  input.eps_o = eo.data();
  input.eps_v = ev.data();
  input.boo = boo.data();
  input.bov = bov.data();
  input.bvv = bvv.data();
  std::vector<double> arena(maps::preconditioner_arena_elements(o, v, q));
  const auto result = maps::run_preconditioner_cpu(o, v, q, input, arena.data(), arena.size());
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t a = 0; a < v; ++a) {
      long double expected = static_cast<long double>(ev[a]) - eo[i];
      for (std::size_t k = 0; k < q; ++k) {
        expected -= static_cast<long double>(boo[(k * o + i) * o + i]) * bvv[(k * v + a) * v + a];
        expected -= static_cast<long double>(bov[(k * o + i) * v + a]) * bov[(k * o + i) * v + a];
      }
      require(std::abs(result.diagonal[i * v + a] - expected) < 3e-14,
              "generated runtime diagonal differs from dense oracle");
    }
  for (std::size_t k = 0; k < bov.size(); ++k)
    require(result.low_rank[k] == 2 * bov[k], "generated low-rank layout mismatch");
  require(maps::preconditioner_contraction_terms(o, v, q) == 2 * o * v * q,
          "generated semantic contraction count mismatch");
  core::System system;
  hf::PhysicalReference ref;
  ref.nocc = o;
  ref.nbf = o + v;
  ref.orbital_energies = eo;
  ref.orbital_energies.insert(ref.orbital_energies.end(), ev.begin(), ev.end());
  auto rejected = hf::prepare_rhf_frame_df_preconditioner(system, ref, q, boo, bov, bvv, 1, 0);
  require(!rejected.data && rejected.contraction_terms == 0,
          "short preparation budget executed the map");
  const auto bound = hf::RHFFrameIdentity::required_storage_bytes(system, ref) +
                     8 * (arena.size() + o * v + bov.size());
  auto accepted = hf::prepare_rhf_frame_df_preconditioner(system, ref, q, boo, bov, bvv, 1, bound);
  require(accepted.data && accepted.numeric_capacity_bytes == bound,
          "exact preparation capacity rejected or miscounted");
  rejected = hf::prepare_rhf_frame_df_preconditioner(system, ref, q, boo, bov, bvv, 1, bound - 1);
  require(!rejected.data, "one-byte-short preparation budget admitted");
}
}  // namespace

int main() {
  try {
    identity();
    for (std::size_t o : {1, 2, 4})
      for (std::size_t v : {1, 3})
        for (std::size_t q : {1, 5}) generated(o, v, q);
    std::cout << "RHF generated preconditioner, exact identity and budgets passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
