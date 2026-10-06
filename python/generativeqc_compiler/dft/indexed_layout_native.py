"""Emit the immutable native-owner binding of the indexed AO/grid contract."""

from __future__ import annotations


def emit_native_ao_grid_binding() -> str:
    """Bind existing CSR storage without another map or per-tile inventory.

    The native plan freezes basis and geometry for its entire lifetime, so its
    owner token replaces the Python adapter's separately rebindable epochs.
    Map capability is still independent of the currently evaluated jet order.
    Admission validates CSR labels once; this binding checks the current span.
    A full sorted unique physical map is the identity and uses the dense route.
    """
    return r"""
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace generativeqc::dft {
/** Native immutable-owner specialization of AoGridBlockLayout. The descriptor
 * borrows its already validated CSR span; it owns no scientific arrays. The
 * plan token binds the immutable basis/quadrature owner, not merely its shape. */
struct NativeAoGridBlockLayout {
  std::size_t nao{}, nactive{}, npoint{}, point_start{};
  unsigned derivative_order{};
  int map_derivative_order{-1};
  bool indexed{};
  const std::size_t* ao_ids{};
  const void* owner_identity{};
};

template <class Layout>
NativeAoGridBlockLayout bind_native_ao_grid_block(
    const Layout& owner, const std::vector<std::size_t>& offsets,
    const std::size_t* indices, std::size_t begin) {
  if (!owner.nao || !owner.tile_points || begin >= owner.npoint ||
      begin % owner.tile_points)
    throw std::invalid_argument("indexed AO/grid point domain mismatch");
  unsigned order = 0;
  constexpr unsigned jets[] = {1, 4, 10, 20};
  while (order < 4 && jets[order] != owner.jets) ++order;
  if (order == 4)
    throw std::invalid_argument("indexed AO/grid jet domain mismatch");
  const auto count = std::min(owner.tile_points, owner.npoint - begin);
  auto active = owner.nao;
  const std::size_t* ids = nullptr;
  if (owner.local_ao) {
    if (owner.map_derivative_order < static_cast<int>(order) ||
        owner.map_derivative_order > 3)
      throw std::invalid_argument("indexed AO/grid map derivative capability mismatch");
    const auto tile = begin / owner.tile_points;
    const auto tiles = 1 + (owner.npoint - 1) / owner.tile_points;
    if (offsets.size() != tiles + 1 || offsets[tile] > offsets[tile + 1] ||
        offsets[tile + 1] > owner.ao_map_entries)
      throw std::invalid_argument("indexed AO/grid CSR span mismatch");
    active = offsets[tile + 1] - offsets[tile];
    if (active > owner.nao || (active && !indices))
      throw std::invalid_argument("indexed AO/grid AO domain mismatch");
    if (active && active < owner.nao) ids = indices + offsets[tile];
  }
  return {owner.nao, active, count, begin, order,
          owner.local_ao ? owner.map_derivative_order : -1,
          owner.local_ao && active < owner.nao, ids, &owner};
}
}  // namespace generativeqc::dft
"""
