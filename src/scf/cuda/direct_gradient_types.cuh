#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

// Direct-force result-layout contract shared by native runtime and generated math.
// This header contains data layout only; scientific derivative equations live in generated owners.
namespace vibeqc::scf::cuda_execution {

/** Density-weighted psss derivatives for the first three canonical centers. */
struct PsssWeightedGradient {
  double center[3][3];
};

/** Cartesian derivatives of one contracted quartet, indexed by input slot. */
struct CartesianQuartetGradient {
  double center[4][3];
};

}  // namespace vibeqc::scf::cuda_execution
