#include <algorithm>
#include <exception>
#include <stdexcept>
#include <string>

#include "core/types.hpp"
#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"

namespace {
thread_local std::string error;
}

extern "C" const char* spherical_eri_error() { return error.c_str(); }

/** Export actual production values for independent full-tensor comparisons.
 * Cases select all-f or contracted mixed s/p/d/f records and either shell order.
 * These are explicit test inputs, not a production basis/chemistry selector.
 */
extern "C" int spherical_eri_values(unsigned scenario, double* output, std::size_t count) {
  try {
    const std::size_t basis = scenario & 1U ? 16 : 28;
    if (scenario > 3 || !output || count != basis * basis * basis * basis)
      throw std::invalid_argument("invalid independent spherical ERI output");
    generativeqc::core::System system;
    system.atoms = {{2, {0.1, -0.3, 0.2}},
                    {2, {-0.7, 0.8, 0.5}},
                    {2, {0.9, 0.2, -0.6}},
                    {2, {-0.4, -0.6, 0.7}}};
    if (scenario & 1U)
      system.shells = {{0, 0, {{0.9, 0.7}, {0.25, 0.3}}},
                       {1, 1, {{0.65, 0.8}, {0.18, -0.15}}},
                       {2, 2, {{0.4, 0.9}, {0.12, -0.1}}},
                       {3, 3, {{0.55, 1.0}, {0.16, -0.2}}}};
    else
      for (std::uint32_t atom = 0; atom < 4; ++atom)
        system.shells.push_back({atom, 3, {{0.35 + 0.1 * atom, 1.0}}});
    if (scenario & 2U) std::reverse(system.shells.begin(), system.shells.end());
    system.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
    system.multiplicity = 1;
    if (generativeqc::molecule::validate_and_normalize(system, error) !=
        GENERATIVEQC_STATUS_SUCCESS)
      return -1;
    const auto values = generativeqc::integrals::build_integrals(system, false);
    if (values.nbf != basis || values.eri.size() != count)
      throw std::runtime_error("production spherical ERI shape differs");
    std::copy(values.eri.begin(), values.eri.end(), output);
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
