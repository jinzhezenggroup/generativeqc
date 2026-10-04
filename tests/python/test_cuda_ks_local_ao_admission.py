"""Compile the actual native opt-in guard, without claiming CUDA execution."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def admission_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Exercise the production decision with synthetic, explicit owner facts."""
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("requires c++ and ccache")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    source = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    start = source.index("    const char* ao_selection = std::getenv(")
    end = source.index("    constexpr std::size_t ao_map_host_budget", start)
    directory = tmp_path_factory.mktemp("ks-local-ao-admission")
    unit, executable = directory / "probe.cpp", directory / "probe"
    unit.write_text(
        r"""
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
enum class SemilocalFamily { Pbe = 1, Wb97mv = 4 };
bool is_semilocal_family(unsigned code, SemilocalFamily family) {
  return code == static_cast<unsigned>(family);
}
struct Options { double semilocal_exchange_scale{0.75}, semilocal_correlation_scale{1}; };
struct Provider {
  struct System { std::vector<int> ecp_terms; } value;
  const System& system() const { return value; }
};
struct Precision { bool mixed{}; bool any_mixed() const { return mixed; } };
int main(int argc, char** argv) {
  if (argc != 3) return 2;
  if (std::string(argv[1]) == "unset") unsetenv("GENERATIVEQC_CUDA_KS_ACTIVE_AO");
  else setenv("GENERATIVEQC_CUDA_KS_ACTIVE_AO", argv[1], 1);
  const std::string mode = argv[2];
  Options options;
  Provider provider;
  Precision precision_schedule;
  unsigned functional = 1, spins = 1;
  bool has_exchange = true, has_range_correction = false, fitted_coulomb = false;
  bool fitted_exchange = false, nonlocal_correlation = false, host_unfused = false;
  double exchange_coefficient = -0.125;
  if (mode == "wb97mv") functional = 4;
  else if (mode == "family") functional = 2;
  else if (mode == "uks") spins = 2;
  else if (mode == "no-exchange") has_exchange = false;
  else if (mode == "range") has_range_correction = true;
  else if (mode == "df-j") fitted_coulomb = true;
  else if (mode == "df-k") fitted_exchange = true;
  else if (mode == "nonlocal") nonlocal_correlation = true;
  else if (mode == "ecp") provider.value.ecp_terms.push_back(1);
  else if (mode == "host") host_unfused = true;
  else if (mode == "mixed") precision_schedule.mixed = true;
  else if (mode == "x-scale") options.semilocal_exchange_scale = 1.0;
  else if (mode == "c-scale") options.semilocal_correlation_scale = 0.5;
  else if (mode == "k-scale") exchange_coefficient = -0.25;
  else if (mode == "near-k") exchange_coefficient = std::nextafter(-0.125, 0.0);
  else if (mode == "nan") options.semilocal_exchange_scale = std::nan("");
  else if (mode != "pbe0") return 3;
  try {
"""
        + source[start:end]
        + '\n    std::cout << "accepted";\n'
        '  } catch (const std::invalid_argument&) { std::cout << "rejected"; }\n}\n'
    )
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-fsanitize=undefined",
            str(unit),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
    )
    return executable


@pytest.mark.parametrize("mode", ["pbe0", "wb97mv"])
@pytest.mark.parametrize("selection", ["unset", "0", "1"])
def test_qualified_opt_in_and_unchanged_default(
    admission_probe: Path, mode: str, selection: str
) -> None:
    result = subprocess.run(
        [str(admission_probe), selection, mode],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout == "accepted"


@pytest.mark.parametrize(
    "mode",
    [
        "family",
        "uks",
        "no-exchange",
        "range",
        "df-j",
        "df-k",
        "nonlocal",
        "ecp",
        "host",
        "mixed",
        "x-scale",
        "c-scale",
        "k-scale",
        "near-k",
        "nan",
    ],
)
@pytest.mark.parametrize("selection", ["unset", "0", "1"])
def test_unqualified_compositions_stay_out(
    admission_probe: Path, mode: str, selection: str
) -> None:
    result = subprocess.run(
        [str(admission_probe), selection, mode],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout == ("rejected" if selection == "1" else "accepted")


@pytest.mark.parametrize("selection", ["", "yes", "2", "-1", "01"])
def test_unknown_selection_fails_closed(admission_probe: Path, selection: str) -> None:
    result = subprocess.run(
        [str(admission_probe), selection, "pbe0"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout == "rejected"
