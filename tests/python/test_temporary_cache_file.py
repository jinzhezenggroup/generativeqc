"""Real local-file reservation, bounded I/O and automatic cleanup contracts."""

import subprocess
import sys
from pathlib import Path

import pytest
from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="Linux local-file spill contract"
)


def test_temporary_cache_file_lifetime_and_io(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Exercise a short read without leaking a named or open temporary file."""
    source = tmp_path / "file.cpp"
    source.write_text(r"""
#include "runtime/temporary_cache_file.hpp"
#include <array>
#include <iostream>
#include <cstdlib>
using generativeqc::runtime::TemporaryCacheFile;
void require(bool value) { if (!value) std::abort(); }
int main(int argc, char** argv) {
  require(argc == 2);
  const std::filesystem::path directory = argv[1];
  if (TemporaryCacheFile::allowance(directory) < 16384) return 77;
  auto names = [&] { std::size_t n=0; for (const auto& p: std::filesystem::directory_iterator(directory)) ++n; return n; };
  const auto before = names();
  {
    TemporaryCacheFile refused;
    require(!refused.reserve(directory, 0));
    require(!refused.reserve(directory / "absent", 16384));
    require(!refused.reserve(directory, std::numeric_limits<std::size_t>::max()));
    require(!refused.finish_writes());
  }
  {
    TemporaryCacheFile file;
    require(file.reserve(directory, 16384));
    require(!file.reserve(directory, 16384));
    require(names() == before); // no visible path even while the owner is live
    std::array<unsigned char, 4096> expected{}, actual{};
    for(std::size_t i=0;i<expected.size();++i) expected[i] = (i*71+9)%251;
    require(file.write(3, expected.data(), expected.size()));
    require(file.write(16384-expected.size(), expected.data(), expected.size()));
    require(!file.write(16383, expected.data(), 2));
    require(file.finish_writes());
    require(file.read(3, actual.data(), actual.size()) && actual == expected);
    require(file.read(16384-actual.size(), actual.data(), actual.size()) && actual == expected);
    require(!file.read(16384, actual.data(), 1));
    // Fault injection acts through the real OS descriptor, not a mocked read.
    bool truncated = false;
    for(const auto& fd: std::filesystem::directory_iterator("/proc/self/fd")) {
      std::error_code error;
      auto target = std::filesystem::read_symlink(fd.path(), error).string();
      if(!error && target.starts_with(directory.string()+"/generativeqc-eri-")) {
        require(::ftruncate(std::stoi(fd.path().filename()), 0) == 0);
        truncated = true;
      }
    }
    require(truncated && !file.read(0, actual.data(), actual.size()));
  }
  require(names() == before);
  std::cout << "temporary file contracts passed\n";
}
""")
    executable = tmp_path / "probe"
    native_cxx.build_executable(
        [source], executable, compile_args=("-std=c++20", f"-I{ROOT / 'src'}")
    )
    result = subprocess.run(
        [str(executable), str(tmp_path)], check=False, capture_output=True, text=True
    )
    if result.returncode == 77:
        pytest.skip("requires local disk-backed temporary space")
    assert result.returncode == 0, result.stderr
    assert "contracts passed" in result.stdout
