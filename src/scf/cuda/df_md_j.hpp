#pragma once

#include <cstddef>

#include "scf/cuda/cartesian_angular.cuh"
#include "scf/cuda/packed_basis.hpp"

namespace generativeqc::scf::cuda_execution {

/** Public AO expansion including component normalization, but not the radial
 * primitive weight. Auxiliary g needs six terms; never use the legacy Direct
 * topology's three-term stride for this view. The owner validates all indices. */
struct DfMdAo {
  unsigned count{};
  Angular angular[6]{};
  double coefficients[6]{};
};

/** One ordered primitive product in an unordered shell pair, or one auxiliary
 * primitive represented by (constant, Gaussian). No primitive multiplicity is
 * hidden in this descriptor. Only off-diagonal shell density folding doubles
 * orientations; all ordered same-shell primitive products remain explicit.
 * Geometry and E are prepared on the GPU. Offsets address disjoint intervals. */
struct DfMdPrimitive {
  std::size_t first_ao{}, second_ao{}, transform_offset{}, hermite_offset{}, response_offset{};
  std::size_t first_atom{}, second_atom{};
  unsigned first_count{}, second_count{}, first_angular{}, second_angular{};
  bool same_shell{}, density_active{};
  double alpha{}, beta{}, weight{};
  Vec3<double> first{}, second{};
  double exponent{};
  Vec3<double> product{};
};

/** Borrowed prepared primitive interval. An auxiliary view uses a unit first
 * AO, alpha=0, first_angular=0 and first_count=1; second AO indices address its
 * public auxiliary vector. Orbital views address a row-major public matrix.
 * All storage belongs to a stream-ordered, budgeted caller; these kernels do
 * not allocate, screen, apply the metric, or admit a production backend. */
struct DfMdView {
  DfMdPrimitive* primitives{};
  const DfMdAo *first_aos{}, *second_aos{};
  double *transforms{}, *density{}, *potential{}, *response_potential{};
  std::size_t count{}, nbf{};
  bool auxiliary{};
  /** Optional exact-zero source compaction, refreshed after each density
   * precontraction. Null preserves dense traversal for transform-only views.
   * Angular offsets are immutable; counts and indices belong to this stream. */
  std::size_t *active_indices{}, *active_offsets{}, *active_counts{};
};

}  // namespace generativeqc::scf::cuda_execution
