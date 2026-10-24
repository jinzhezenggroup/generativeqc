"""Explicit pairs preserve the borrowed foreign-POD representation contract."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def _assert_byte_inspection(source: str) -> None:
    body = source.split("Diagnostic validate_topology_host(", 1)[1].split(
        "if (topology.bucket_count != 0)", 1
    )[0]
    assert "static_cast<const AtomPairLayout*>" not in body
    assert "static_cast<const std::byte*>(topology.atom_pairs)" in body
    assert "static_cast<std::size_t>(pair) * sizeof(current)" in body
    assert "std::memcpy(&current, pair_bytes + byte_offset, sizeof(current))" in body


def test_explicit_pairs_do_not_form_foreign_typed_array_references() -> None:
    source = (ROOT / "src/runtime/ragged_topology.cpp").read_text()
    _assert_byte_inspection(source)
    # The tempting security-autofix regresses the documented foreign-POD ABI,
    # even when its pointer values happen to match on a particular compiler.
    broken = source.replace(
        "static_cast<const std::byte*>(topology.atom_pairs)",
        "static_cast<const AtomPairLayout*>(topology.atom_pairs)",
    )
    with pytest.raises(AssertionError):
        _assert_byte_inspection(broken)


@pytest.mark.parametrize("sanitize", [False, True])
def test_foreign_pair_arrays_preserve_diagnostics(
    tmp_path: Path, native_cxx: NativeCxx, sanitize: bool
) -> None:
    probe = tmp_path / "pairs.cpp"
    probe.write_text(
        r"""
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include "runtime/ragged_topology.hpp"
using namespace generativeqc::runtime::ragged;
struct ForeignPair { std::int64_t left; std::int64_t right; };
static_assert(!std::is_same_v<ForeignPair, AtomPairLayout>);
static_assert(sizeof(ForeignPair) == sizeof(AtomPairLayout));
static_assert(alignof(ForeignPair) == alignof(AtomPairLayout));
static_assert(offsetof(ForeignPair, right) == offsetof(AtomPairLayout, second));
int main() {
  std::int64_t atoms[]{0,3}, shells[]{0,3}, orbitals[]{0,3}, matrices[]{0,9};
  std::int64_t atom_shells[]{0,1,2,3}, shell_orbitals[]{0,1,2,3};
  std::int64_t shell_atoms[]{0,1,2}, orbital_shells[]{0,1,2}, orbital_atoms[]{0,1,2};
  std::int64_t offsets[]{0,3};
  ForeignPair foreign[]{{0,1},{0,2},{1,2}};
  AtomPairLayout native[]{{0,1},{0,2},{1,2}};
  TopologyView t{};
  t.plan_token=1; t.batch_size=1; t.total_atoms=3; t.total_shells=3;
  t.total_orbitals=3; t.total_matrix_elements=9; t.total_pairs=3;
  t.atom_offset_count=t.batch_shell_offset_count=t.batch_orbital_offset_count=2;
  t.matrix_offset_count=2; t.atom_shell_offset_count=t.shell_orbital_offset_count=4;
  t.shell_to_atom_count=t.orbital_to_shell_count=t.orbital_to_atom_count=3;
  t.pair_map_kind=PairMapKind::kExplicit; t.pair_offset_count=2; t.atom_pair_count=3;
  t.atom_offsets=atoms; t.batch_shell_offsets=shells; t.batch_orbital_offsets=orbitals;
  t.matrix_offsets=matrices; t.atom_shell_offsets=atom_shells;
  t.shell_orbital_offsets=shell_orbitals; t.shell_to_atom=shell_atoms;
  t.orbital_to_shell=orbital_shells; t.orbital_to_atom=orbital_atoms;
  t.pair_offsets=offsets;
  auto check=[&](const void* pairs, Error error, Field field, std::int64_t index) {
    t.atom_pairs=pairs; const auto d=validate_topology_host(t);
    return d.error==error && d.field==field && d.index==index;
  };
  if(!check(foreign,Error::kSuccess,Field::kNone,-1)) return 1;
  if(!check(native,Error::kSuccess,Field::kNone,-1)) return 2;
  foreign[1]={0,1};
  if(!check(foreign,Error::kInvalidPairMap,Field::kAtomPairs,1)) return 3;
  foreign[1]={0,3};
  if(!check(foreign,Error::kInvalidPairMap,Field::kAtomPairs,1)) return 4;
  foreign[1]={0,2}; foreign[2]={2,1};
  if(!check(foreign,Error::kInvalidPairMap,Field::kAtomPairs,2)) return 5;
  foreign[2]={1,2};
  if(!check(nullptr,Error::kNullPointer,Field::kAtomPairs,-1)) return 6;
  alignas(AtomPairLayout) std::array<std::byte,sizeof(foreign)+1> bytes{};
  std::memcpy(bytes.data()+1,foreign,sizeof(foreign));
  if(!check(bytes.data()+1,Error::kMisalignedPointer,Field::kAtomPairs,-1)) return 7;
  offsets[0]=1;
  if(!check(foreign,Error::kInvalidOffsets,Field::kPairOffsets,0)) return 8;
  offsets[0]=0; offsets[1]=4;
  if(!check(foreign,Error::kInvalidOffsets,Field::kPairOffsets,0)) return 9;
  offsets[1]=0; t.total_pairs=t.atom_pair_count=0;
  if(!check(nullptr,Error::kSuccess,Field::kNone,-1)) return 10;
  t.total_pairs=t.atom_pair_count=std::numeric_limits<std::int64_t>::max();
  if(!check(foreign,Error::kCountOverflow,Field::kAtomPairs,-1)) return 11;
  return 0;
}
"""
    )
    flags = ("-fsanitize=undefined", "-fno-sanitize-recover=all") if sanitize else ()
    output = tmp_path / "pairs"
    native_cxx.build_executable(
        [probe, ROOT / "src/runtime/ragged_topology.cpp"],
        output,
        compile_args=(
            "-std=c++20",
            "-O2",
            "-fstrict-aliasing",
            "-I",
            str(ROOT / "src"),
            *flags,
        ),
        link_args=flags,
    )
    subprocess.run([str(output)], check=True, capture_output=True, timeout=10)
