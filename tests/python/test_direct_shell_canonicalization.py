"""Exact shell-slot semantics for the generated low-order Direct force adapter."""

from __future__ import annotations

import itertools
import shutil
import subprocess
from pathlib import Path

import pytest
from vibeqc_compiler.integral.weighted_eri_cuda import (
    emit_direct_shell_canonicalization_helper,
    emit_low_order_weighted_header,
    emit_psss_weighted_header,
)

ROOT = Path(__file__).resolve().parents[2]

ShellSlots = tuple[int, ...]
ExecutedCase = tuple[ShellSlots, ShellSlots, ShellSlots]
ExecutedCases = list[ExecutedCase]


def _flatten_component(
    order: ShellSlots, counts: ShellSlots, component: ShellSlots
) -> int:
    index = 0
    for slot in order:
        index = index * counts[slot] + component[slot]
    return index


def _orbit() -> tuple[tuple[int, ...], ...]:
    """Enumerate the eight ERI symmetries, independently of the sorting code."""
    result = []
    for swap_pairs, swap_first, swap_second in itertools.product(
        (False, True), repeat=3
    ):
        first = (1, 0) if swap_first else (0, 1)
        second = (3, 2) if swap_second else (2, 3)
        result.append(second + first if swap_pairs else first + second)
    return tuple(result)


def _expected(angular: tuple[int, ...]) -> tuple[int, ...]:
    candidates = _orbit()
    best = max(tuple(angular[slot] for slot in order) for order in candidates)
    # Stable ties keep raw pair/slot order; shell IDs are not sorting keys.
    return min(
        order for order in candidates if tuple(angular[slot] for slot in order) == best
    )


@pytest.fixture(scope="module")
def compiled_slots(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile the actual emitted helper on the host, not a Python translation."""
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++ compiler is required for generated shell-slot execution")
    directory = tmp_path_factory.mktemp("direct-shell-slots")
    source = directory / "slots.cpp"
    executable = directory / "slots"
    source.write_text(
        "#include <cstdint>\n#include <iostream>\n"
        "#define __device__\n#define __forceinline__ inline\n"
        + emit_direct_shell_canonicalization_helper()
        + r"""
int main() {
  unsigned mode;
  while (std::cin >> mode) {
    std::int32_t shells[4];
    std::int64_t wide_shells[4];
    unsigned angular[16]{};
    std::uint8_t narrow_angular[16]{};
    for (unsigned slot = 0; slot < 4; ++slot) {
      std::cin >> shells[slot];
      wide_shells[slot] = shells[slot];
      if (shells[slot] < 0 || shells[slot] >= 16) return 2;
    }
    for (unsigned slot = 0; slot < 4; ++slot) {
      unsigned value;
      std::cin >> value;
      angular[shells[slot]] = value;
      narrow_angular[shells[slot]] = static_cast<std::uint8_t>(value);
    }
    if (!std::cin) return 3;
    unsigned slots[4] = {99U, 99U, 99U, 99U};
    if (mode == 0U) {
      canonicalize_direct_shell_slots(angular, shells, slots);
    } else {
      canonicalize_direct_shell_slots(narrow_angular, wide_shells, slots);
    }
    std::cout << slots[0] << ' ' << slots[1] << ' '
              << slots[2] << ' ' << slots[3] << '\n';
  }
}
""",
        encoding="utf-8",
    )
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return executable


@pytest.fixture(scope="module")
def executed_cases(
    compiled_slots: Path,
) -> ExecutedCases:
    layouts = ((0, 1, 2, 3), (7, 2, 9, 5), (7, 7, 9, 5), (2, 7, 2, 7), (5, 5, 5, 5))
    cases = []
    records = []
    for mode, shells, angular in itertools.product(
        (0, 1), layouts, itertools.product(range(4), repeat=4)
    ):
        # Aliased shell IDs necessarily have the same angular momentum.
        if any(
            shells[a] == shells[b] and angular[a] != angular[b]
            for a in range(4)
            for b in range(a)
        ):
            continue
        cases.append((shells, angular))
        records.append(" ".join(map(str, (mode, *shells, *angular))))
    result = subprocess.run(
        [str(compiled_slots)],
        input="\n".join(records) + "\n",
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    outputs = [tuple(map(int, line.split())) for line in result.stdout.splitlines()]
    assert len(outputs) == len(cases) == 1192
    return [
        (shells, angular, slots)
        for (shells, angular), slots in zip(cases, outputs, strict=True)
    ]


def test_emitted_slots_match_independent_eightfold_oracle(
    executed_cases: ExecutedCases,
) -> None:
    for shells, angular, slots in executed_cases:
        assert slots == _expected(angular), (shells, angular, slots)
        assert slots in _orbit()


def test_order2_preserves_historical_specialized_permutations(
    executed_cases: ExecutedCases,
) -> None:
    for _, angular, slots in executed_cases:
        if sum(angular) != 2:
            continue
        # Test-only legacy mapping is deliberately class-specific, unlike the
        # generated pair-symmetry helper. Cover every old low-order branch.
        if 2 in angular:
            d = angular.index(2)
            other = (2, 3) if d < 2 else (0, 1)
            expected = (d, d ^ 1, *other)
        elif sum(angular[:2]) == sum(angular[2:]) == 1:
            first = angular[:2].index(1)
            second = 2 + angular[2:].index(1)
            expected = (first, first ^ 1, second, second ^ 1)
        else:
            expected = (0, 1, 2, 3) if angular[0] == angular[1] == 1 else (2, 3, 0, 1)
        assert slots == expected, (angular, slots)


def test_equal_pair_classes_preserve_primitive_pair_ownership(
    executed_cases: ExecutedCases,
) -> None:
    for _, angular, slots in executed_cases:
        if sorted(angular[:2]) == sorted(angular[2:]):
            assert set(slots[:2]) == {0, 1}
            assert set(slots[2:]) == {2, 3}
        if len(set(angular)) == 1:
            assert slots == (0, 1, 2, 3)


def test_component_and_recovered_center_mapping_is_preserved(
    executed_cases: ExecutedCases,
) -> None:
    for shells, angular, slots in executed_cases:
        if sum(angular) not in (2, 3):
            continue
        expected = _expected(angular)
        counts = tuple((value + 1) * (value + 2) // 2 for value in angular)
        for component in itertools.product(*(range(count) for count in counts)):
            assert _flatten_component(slots, counts, component) == _flatten_component(
                expected, counts, component
            )
        # Distinct independent derivatives expose accidental inverse mapping.
        gradient = (1.25, -4.5, 8.0, -(1.25 - 4.5 + 8.0))
        raw = [0.0] * 4
        reference = [0.0] * 4
        for center, value in enumerate(gradient):
            raw[slots[center]] = value
            reference[expected[center]] = value
        assert raw == reference
        # Multiple shells may also share an atom; coalescing must not change.
        for atom_layout in ((0, 1, 2, 3), (0, 0, 1, 1), (0, 0, 0, 0)):
            actual_atoms = [
                sum(raw[i] for i in range(4) if atom_layout[i] == atom)
                for atom in range(4)
            ]
            expected_atoms = [
                sum(reference[i] for i in range(4) if atom_layout[i] == atom)
                for atom in range(4)
            ]
            assert actual_atoms == expected_atoms, shells


def test_native_force_adapters_use_one_generated_slot_owner() -> None:
    for name, calls in (("direct_force_order2.cuh", 3), ("direct_force_order3.cuh", 1)):
        source = (ROOT / "src/scf/cuda" / name).read_text(encoding="utf-8")
        assert (
            source.count("generated_weighted_eri::canonicalize_direct_shell_slots(")
            == calls
        )
        assert "first_p_slot" not in source
        assert "d_slot" not in source
        assert "order3_detail::canonicalize_shell_slots" not in source
        assert "inline unsigned pair_class(" not in source
    order2 = (ROOT / "src/scf/cuda/direct_force_order2.cuh").read_text(encoding="utf-8")
    assert (
        "shell_class != kPspsShellClass && shell_class != kPpssShellClass &&" in order2
    )
    assert "shell_class != kDsssShellClass" in order2
    assert "if (shell_class != TargetShellClass) return;" in order2
    assert "if (shell_class != kPspsShellClass) return;" in order2


def test_low_order_header_owns_helper_without_changing_standalone_interface() -> None:
    helper = emit_direct_shell_canonicalization_helper()
    assert helper == emit_direct_shell_canonicalization_helper()
    assert emit_low_order_weighted_header(inline_single_use=True).count(helper) == 1
    assert "canonicalize_direct_shell_slots" not in emit_psss_weighted_header(
        inline_single_use=True
    )
