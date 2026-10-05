"""Independent full-coordinate orbit/metric tests for restricted storage."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.cc.pair_coordinates import (
    RestrictedPairCoordinates,
    cpp_coordinates,
)


@pytest.mark.parametrize("o,v", [(1, 1), (1, 7), (2, 3), (4, 1), (3, 5)])
def test_complete_orbits_roundtrip_and_full_metric(o: int, v: int) -> None:
    layout = RestrictedPairCoordinates(o, v)
    n = o * v
    rng = np.random.default_rng(1902 + n)
    x = rng.normal(size=(o, o, v, v))
    y = rng.normal(size=x.shape)
    x = (x + x.transpose(1, 0, 3, 2)) / 2
    y = (y + y.transpose(1, 0, 3, 2)) / 2
    px = np.empty(layout.packed_doubles)
    py = np.empty_like(px)
    weights = np.zeros(len(px), dtype=int)
    seen = set()
    for i in range(o):
        for j in range(o):
            for a in range(v):
                for b in range(v):
                    flat = np.ravel_multi_index((i, j, a, b), x.shape).item()
                    mate, slot, weight = layout.orbit(flat)
                    assert mate == np.ravel_multi_index((j, i, b, a), x.shape)
                    assert layout.orbit(mate) == (flat, slot, weight)
                    if flat <= mate:
                        assert slot not in seen
                        seen.add(slot)
                        px[slot] = x.flat[flat]
                        py[slot] = y.flat[flat]
                        weights[slot] = weight
    assert seen == set(range(n * (n + 1) // 2))
    assert np.count_nonzero(weights == 1) == n and sum(weights) == n * n
    for flat in range(n * n):
        assert px[layout.orbit(flat)[1]] == x.flat[flat]
    full = np.sum(x.astype(np.longdouble) * y.astype(np.longdouble))
    packed = np.sum(weights * px.astype(np.longdouble) * py.astype(np.longdouble))
    np.testing.assert_allclose(full, packed, atol=1e-16, rtol=1e-16)


def test_invalid_pair_dimensions_and_indices() -> None:
    for dimensions in ((0, 1), (1, -1), (1, True)):
        with pytest.raises(ValueError):
            RestrictedPairCoordinates(*dimensions)
    for index in (-1, 36, 1.0):
        with pytest.raises(ValueError):
            RestrictedPairCoordinates(2, 3).orbit(index)


def test_generated_host_coordinate_adapter(tmp_path: Path) -> None:
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if not compiler or not cache:
        pytest.skip("requires compiler and ccache")
    source = tmp_path / "coordinates.cpp"
    source.write_text(
        cpp_coordinates()
        + r"""
#include <iostream>
#include <limits>
int main() {
  using generativeqc::cc::generated::RestrictedPairCoordinates;
  for (const auto shape : {std::pair{1,1},std::pair{2,3},std::pair{3,5}}) {
    RestrictedPairCoordinates map{std::size_t(shape.first),std::size_t(shape.second)};
    const auto n=map.o*map.v;
    for(std::size_t k=0;k<n*n;++k) {
      const auto orbit=map(k);
      std::cout<<orbit.partner<<' '<<orbit.slot<<' '<<int(orbit.weight)<<'\n';
    }
    const auto tiny=std::numeric_limits<double>::denorm_min();
    if(map.project(tiny,tiny)!=tiny || map.project(1e308,1e308)!=1e308) return 2;
    if(map.project(2.,4.)!=3.)return 3;
  }
}
"""
    )
    obj, executable = tmp_path / "coordinates.o", tmp_path / "coordinates"
    subprocess.run(
        [cache, compiler, "-std=c++20", "-O2", "-c", str(source), "-o", str(obj)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [compiler, str(obj), "-o", str(executable)], check=True, capture_output=True
    )
    output = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    expected = []
    for o, v in ((1, 1), (2, 3), (3, 5)):
        layout = RestrictedPairCoordinates(o, v)
        expected.extend(layout.orbit(k) for k in range((o * v) ** 2))
    assert [tuple(map(int, line.split())) for line in output] == expected
