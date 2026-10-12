"""Independent spin-channel, tiling and native-backend gates for UCCSD(T)."""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from generativeqc_compiler.cc.triples import triples_energy
from generativeqc_compiler.cc.uccsd_layout import UCCSDLayout
from generativeqc_compiler.cc.uccsd_triples import (
    CHANNELS,
    TriplesTileProgram,
    build_tile_program,
    channel_possible,
    channel_shape,
)
from generativeqc_compiler.tensor import (
    Program,
    add,
    execute,
    linearize,
    transpose_program,
)
from test_uccsd_equations import _amplitudes, _feeds, _molecular_case

_H6 = "; ".join(f"H 0 0 {position}" for position in range(6))


def test_pinned_upstream_formula_and_independent_oracle_identity() -> None:
    """Qualify upstream inputs without pinning generated/production function bodies."""
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (root / "tools/generativeqc_cc/uccsd_triples_source_manifest.json").read_text()
    )
    pyscf = pytest.importorskip("pyscf")
    assert pyscf.__version__ == manifest["version"]
    for entry in manifest["files"]:
        module = pytest.importorskip(
            entry["path"].removesuffix(".py").replace("/", ".")
        )
        assert (
            hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            == entry["sha256"]
        )


@lru_cache(maxsize=8)
def _case(atom: str = _H6, spin: int = 0, broken: bool = False) -> tuple:
    coupled_cluster, eris, layout, fock, integrals = _molecular_case(
        atom, spin, broken=broken
    )
    amplitudes = _amplitudes(layout)
    feeds = _feeds(layout, fock, integrals, amplitudes)
    for domain in "ov":
        for spin_label, energies in zip("ab", eris.mo_energy, strict=True):
            occupied = getattr(layout, "o" + spin_label)
            feeds[f"eps_{domain}_{spin_label}"] = (
                energies[:occupied] if domain == "o" else energies[occupied:]
            )
    return coupled_cluster, eris, layout, amplitudes, feeds


def _oracle_channels(coupled_cluster: Any, eris: Any, amplitudes: dict) -> dict:
    """Observe the four scientific scalar channel reductions of the pinned oracle.

    Only the independent PySCF module's NumPy namespace is substituted; no
    production implementation or global NumPy state is patched. Every tensor
    contraction still executes through NumPy and PySCF's own equation inventory.
    """
    reference = pytest.importorskip("pyscf.cc.uccsd_t_slow")
    contributions = []

    def contraction(equation: str, *operands: Any, **kwargs: Any) -> Any:
        result = np.einsum(equation, *operands, **kwargs)
        if np.ndim(result) == 0 and all(np.ndim(value) == 6 for value in operands):
            contributions.append(float(result) / 4)
        return result

    singles = (amplitudes["t1a"], amplitudes["t1b"])
    doubles = (amplitudes["t2aa"], amplitudes["t2ab"], amplitudes["t2bb"])
    with patch.object(
        reference, "numpy", SimpleNamespace(einsum=contraction, asarray=np.asarray)
    ):
        total = reference.kernel(coupled_cluster, eris, singles, doubles)
    np.testing.assert_allclose(
        coupled_cluster.ccsd_t(singles, doubles, eris), total, atol=3e-13, rtol=3e-13
    )
    return dict(zip(("aaa", "bbb", "aab", "abb"), contributions, strict=True))


def _tile_feeds(
    tile: TriplesTileProgram,
    sources: dict,
    coordinates: tuple[tuple[int, ...], ...],
) -> dict:
    """Execute only explicit source transpose/selection bindings, never equations."""
    positions = dict(zip("ijkabc", coordinates, strict=True))
    feeds = {}
    for binding in tile.operands:
        values = sources[binding.source].transpose(binding.axes)
        selection = tuple(
            positions.get(label, tuple(range(size)))
            for label, size in zip(binding.labels, values.shape, strict=True)
        )
        feeds[binding.name] = np.ascontiguousarray(values[np.ix_(*selection)])
    return feeds


@pytest.mark.parametrize(
    "atom,spin,broken",
    [
        ("Li 0 0 0", 1, False),
        ("O 0 0 0; H 0 0 .97", 1, False),
        (_H6, 0, False),
        ("; ".join(f"H 0 0 {1.6 * position}" for position in range(4)), 0, True),
    ],
)
def test_all_spin_channels_and_partitioned_tiles_match_pyscf(
    atom: str, spin: int, broken: bool
) -> None:
    coupled_cluster, eris, layout, amplitudes, sources = _case(atom, spin, broken)
    expected = _oracle_channels(coupled_cluster, eris, amplitudes)
    for channel in CHANNELS:
        if not channel_possible(layout, channel):
            assert abs(expected[channel]) < 2e-13
            continue
        full_shape = channel_shape(layout, channel)
        coordinates = tuple(tuple(range(size)) for size in full_shape)
        tile = build_tile_program(layout, channel, full_shape)
        full = execute(tile.program, _tile_feeds(tile, sources, coordinates)).outputs
        np.testing.assert_allclose(
            full["triples_energy"], expected[channel], atol=3e-13, rtol=3e-13
        )
        partitions = tuple(
            tuple(
                tuple(range(start, min(start + step, size)))
                for start in range(0, size, step)
            )
            for position, size in enumerate(full_shape)
            for step in (2 if position in (0, 3) else size,)
        )
        partial = 0.0
        for coordinates in product(*partitions):
            tile = build_tile_program(
                layout, channel, tuple(len(values) for values in coordinates)
            )
            partial += float(
                execute(tile.program, _tile_feeds(tile, sources, coordinates)).outputs[
                    "triples_energy"
                ]
            )
        np.testing.assert_allclose(partial, expected[channel], atol=3e-13, rtol=3e-13)


def test_restricted_limit_matches_existing_rccsd_triples() -> None:
    _, eris, layout, fock, integrals = _molecular_case(_H6, 0, restricted=True)
    generator = np.random.default_rng(51)
    singles = generator.normal(scale=0.03, size=(layout.oa, layout.va))
    doubles = generator.normal(
        scale=0.02, size=(layout.oa, layout.oa, layout.va, layout.va)
    )
    doubles += doubles.transpose(1, 0, 3, 2)
    same_spin = doubles - doubles.swapaxes(2, 3)
    amplitudes = {
        "t1a": singles,
        "t1b": singles,
        "t2aa": same_spin,
        "t2ab": doubles,
        "t2bb": same_spin,
    }
    sources = _feeds(layout, fock, integrals, amplitudes)
    for domain in "ov":
        for spin in "ab":
            sources[f"eps_{domain}_{spin}"] = (
                eris.mo_energy[: layout.oa]
                if domain == "o"
                else eris.mo_energy[layout.oa :]
            )
    actual = 0.0
    for channel in CHANNELS:
        shape = channel_shape(layout, channel)
        tile = build_tile_program(layout, channel, shape)
        feeds = _tile_feeds(tile, sources, tuple(tuple(range(size)) for size in shape))
        actual += float(execute(tile.program, feeds).outputs["triples_energy"])
    expected = triples_energy(
        layout.oa,
        layout.va,
        eris.get_ovvv(),
        eris.ovoo,
        eris.ovov,
        eris.fock[: layout.oa, layout.oa :],
        singles,
        doubles,
        eris.mo_energy[: layout.oa],
        eris.mo_energy[layout.oa :],
    )
    np.testing.assert_allclose(actual, expected, atol=3e-13, rtol=3e-13)


@pytest.mark.parametrize("channel", CHANNELS)
def test_native_cpu_tile_matches_independent_channel(
    channel: str, tmp_path: Path
) -> None:
    from tools.generativeqc_cc.native_tensor_cpu import NativeCCTensorExecutor

    coupled_cluster, eris, layout, amplitudes, sources = _case()
    tile = build_tile_program(layout, channel, channel_shape(layout, channel))
    coordinates = tuple(tuple(range(size)) for size in tile.shape)
    feeds = _tile_feeds(tile, sources, coordinates)
    expected = _oracle_channels(coupled_cluster, eris, amplitudes)[channel]
    executor = NativeCCTensorExecutor(max_bytes=256 << 20, cache=tmp_path)
    actual = executor.execute(tile.program, feeds)["triples_energy"]
    np.testing.assert_allclose(actual, expected, atol=3e-13, rtol=3e-13)
    partial = build_tile_program(layout, channel, (2, 1, 2, 2, 1, 2))
    selected = tuple(tuple(range(size)) for size in partial.shape)
    partial_feeds = _tile_feeds(partial, sources, selected)
    np.testing.assert_allclose(
        executor.execute(partial.program, partial_feeds)["triples_energy"],
        execute(partial.program, partial_feeds).outputs["triples_energy"],
        atol=3e-13,
        rtol=3e-13,
    )
    wrong_normalization = Program(
        {
            "triples_energy": add(
                tile.program.outputs["triples_energy"], coefficients=(2,)
            )
        }
    )
    mutant = execute(wrong_normalization, feeds).outputs["triples_energy"]
    assert not np.isclose(mutant, expected, atol=3e-13, rtol=3e-13)
    corrupted_sources = {
        name: -value if name.startswith("gooov_") else value
        for name, value in sources.items()
    }
    wrong_exchange = execute(
        tile.program, _tile_feeds(tile, corrupted_sources, coordinates)
    ).outputs["triples_energy"]
    assert not np.isclose(wrong_exchange, expected, atol=3e-13, rtol=3e-13)


@pytest.mark.parametrize("channel", CHANNELS)
def test_tile_jvp_vjp_transpose_and_finite_difference(channel: str) -> None:
    _, _, layout, _, sources = _case()
    shape = (2, 1, 2, 2, 1, 2)
    tile = build_tile_program(layout, channel, shape)
    coordinates = tuple(tuple(range(size)) for size in shape)
    feeds = _tile_feeds(tile, sources, coordinates)
    generator = np.random.default_rng(91)
    directions = {
        name: generator.normal(scale=0.05, size=value.shape)
        for name, value in feeds.items()
    }
    jvp = linearize(tile.program, tuple(feeds))
    vjp = transpose_program(tile.program, ("triples_energy",), inputs=tuple(feeds))
    forward = execute(
        jvp.program,
        {**feeds, **{f"d_{name}": value for name, value in directions.items()}},
    ).outputs["d_triples_energy"]
    reverse = execute(
        vjp.program, {**feeds, "bar_triples_energy": np.array(1.0)}
    ).outputs
    np.testing.assert_allclose(
        forward,
        sum(np.vdot(directions[name], reverse[f"bar_{name}"]) for name in feeds),
        atol=3e-12,
        rtol=3e-12,
    )
    step = 1e-5
    plus, minus = (
        execute(
            tile.program,
            {
                name: value + sign * step * directions[name]
                for name, value in feeds.items()
            },
        ).outputs["triples_energy"]
        for sign in (1, -1)
    )
    np.testing.assert_allclose(forward, (plus - minus) / (2 * step), atol=3e-10)


@pytest.mark.parametrize("shape", [(), (1,) * 5, (True,) * 6, (0,) * 6, (4,) * 6])
def test_invalid_tile_admission(shape: tuple) -> None:
    with pytest.raises(ValueError, match="tile shape"):
        build_tile_program(UCCSDLayout(3, 3, 3, 3), "aab", shape)
    with pytest.raises(ValueError, match="spin channel"):
        channel_shape(UCCSDLayout(3, 3, 3, 3), "aba")


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_TENSOR_CUDA_TEST") != "1",
    reason="requires explicit allocated-GPU opt-in",
)
@pytest.mark.parametrize("channel", CHANNELS)
def test_real_cuda_tile_matches_independent_channel(
    channel: str, tmp_path: Path
) -> None:
    from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from generativeqc_compiler.common.cuda_target import cuda_target_info

    from tools.generativeqc_cc.native_tensor_cuda import CudaCCTensorExecutor

    assert os.environ.get("SLURM_JOB_ID"), (
        "CUDA tests require a finite Slurm allocation"
    )
    coupled_cluster, eris, layout, amplitudes, sources = _case()
    tile = build_tile_program(layout, channel, channel_shape(layout, channel))
    coordinates = tuple(tuple(range(size)) for size in tile.shape)
    compiler = CudaCompilerAdapter(
        Path(os.environ["GENERATIVEQC_NVCC"]),
        cuda_target_info(os.environ.get("GENERATIVEQC_TENSOR_ARCH", "sm_120")),
    )
    executor = CudaCCTensorExecutor(
        max_bytes=256 << 20, compiler=compiler, cache=tmp_path
    )
    actual = executor.execute(tile.program, _tile_feeds(tile, sources, coordinates))[
        "triples_energy"
    ]
    expected = _oracle_channels(coupled_cluster, eris, amplitudes)[channel]
    np.testing.assert_allclose(actual, expected, atol=3e-13, rtol=3e-13)
    partial = build_tile_program(layout, channel, (2, 1, 2, 2, 1, 2))
    selected = tuple(tuple(range(size)) for size in partial.shape)
    partial_feeds = _tile_feeds(partial, sources, selected)
    np.testing.assert_allclose(
        executor.execute(partial.program, partial_feeds)["triples_energy"],
        execute(partial.program, partial_feeds).outputs["triples_energy"],
        atol=3e-13,
        rtol=3e-13,
    )
