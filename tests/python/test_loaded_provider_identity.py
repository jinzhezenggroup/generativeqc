"""Observe real small loaded ELF objects; no CUDA library or device is used."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="Linux ELF loader contract"
)

PROBE = r"""
#include <dlfcn.h>
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "runtime/loaded_provider_identity.hpp"
using namespace generativeqc::runtime;
void require(bool ok, const char* message) {if(!ok)throw std::runtime_error(message);}
int main(int argc,char** argv) {
 try {
  require(argc==6,"fixture arguments");
  std::array<void*,4> handles{};
  for(unsigned i=0;i<3;++i) {
   handles[i]=dlopen(argv[i+1],RTLD_NOW|RTLD_LOCAL);
   require(handles[i]!=nullptr,"could not load HOST fixture DSO");
  }
  const auto first=loaded_provider_identity(argv[1],"fixture_identity");
  require(first.status==LoadedIdentityStatus::available && first.build_id=="01020304",
          "real loaded GNU build-ID not observed");
  const auto second=loaded_provider_identity(argv[2],"fixture_identity");
  require(second.status==LoadedIdentityStatus::available && second.build_id=="aabbccdd",
          "symbol owner/build identity was pooled across loaded DSOs");
  require(loaded_provider_identity(argv[1],"fixture_identity",dlsym(handles[1],"fixture_identity")).status
              ==LoadedIdentityStatus::wrong_symbol_owner,
          "execution symbol from another object was accepted");
  handles[3]=dlopen(argv[5],RTLD_NOW|RTLD_LOCAL);
  require(handles[3]!=nullptr,"could not load dependent HOST fixture");
  require(loaded_provider_identity(argv[5],"fixture_identity").status==LoadedIdentityStatus::wrong_symbol_owner,
          "dependency symbol was assigned the enclosing object's build-ID");
  require(loaded_provider_identity(argv[3],"fixture_identity").status==LoadedIdentityStatus::missing_build_id,
          "missing GNU build-ID was accepted");
  require(loaded_provider_identity(argv[1],"absent_function").status==LoadedIdentityStatus::missing_symbol,
          "missing symbol was accepted");
  require(loaded_provider_identity(argv[4],"fixture_identity").status==LoadedIdentityStatus::not_loaded,
          "observer loaded an absent DSO");
  require(dlopen(argv[4],RTLD_NOW|RTLD_LOCAL|RTLD_NOLOAD)==nullptr,
          "identity query changed library loading");
#ifdef LM_ID_NEWLM
  void* other_namespace=dlmopen(LM_ID_NEWLM,argv[4],RTLD_NOW|RTLD_LOCAL);
  require(other_namespace!=nullptr,"could not load isolated HOST fixture namespace");
  require(loaded_provider_identity(argv[4],"fixture_identity").status==LoadedIdentityStatus::not_loaded,
          "an isolated loader namespace was silently treated as the execution namespace");
  dlclose(other_namespace);
#endif
  // These are independent hand-authored ELF note bytes, not output from the parser.
  const unsigned char valid[]{4,0,0,0,4,0,0,0,3,0,0,0,'G','N','U',0,1,2,3,4};
  std::vector<std::byte> note(sizeof(valid));std::memcpy(note.data(),valid,sizeof(valid));
  require(parse_loaded_build_id_notes(note).build_id=="01020304","valid note changed");
  auto bad=note;bad[4]=std::byte{255};bad[5]=bad[6]=bad[7]=std::byte{255};
  require(parse_loaded_build_id_notes(bad).status==LoadedIdentityStatus::malformed_notes,
          "overflow/truncated note accepted");
  bad=note;bad.pop_back();
  require(parse_loaded_build_id_notes(bad).status==LoadedIdentityStatus::malformed_notes,
          "truncated descriptor accepted");
  bad=note;bad.insert(bad.end(),note.begin(),note.end());
  require(parse_loaded_build_id_notes(bad).status==LoadedIdentityStatus::ambiguous_build_id,
          "duplicate build-ID notes accepted");
  auto huge=std::vector<std::byte>(65537);
  require(parse_loaded_build_id_notes(huge).status==LoadedIdentityStatus::malformed_notes,
          "unbounded note input accepted");
  require(first.library_pin!=nullptr,"identity did not retain its loaded object");
  for(auto handle:handles)dlclose(handle);
  std::cout<<"HOST real ELF build-ID ownership/missing/bounds checks passed\n";
  return 0;
 } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
"""


def test_real_loaded_provider_identity(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    source = tmp_path / "fixture.cpp"
    source.write_text('extern "C" int fixture_identity() {return 7;}\n')
    libraries = []
    try:
        for name, build_id in (
            ("first", "0x01020304"),
            ("second", "0xaabbccdd"),
            ("no_id", "none"),
        ):
            library = tmp_path / f"{name}.so"
            required_native_cxx.build_shared(
                [source], library, link_args=[f"-Wl,--build-id={build_id}"]
            )
            libraries.append(library)
        absent = tmp_path / "not_loaded.so"
        required_native_cxx.build_shared([source], absent)
        outer = tmp_path / "outer.cpp"
        outer.write_text(
            'extern "C" int fixture_identity();\n'
            'extern "C" int fixture_outer() {return fixture_identity();}\n'
        )
        outer_library = tmp_path / "outer.so"
        required_native_cxx.build_shared(
            [outer], outer_library, link_args=[str(libraries[0])]
        )
        probe = tmp_path / "probe.cpp"
        probe.write_text(PROBE)
        binary = required_native_cxx.build_executable(
            [probe],
            tmp_path / "probe",
            compile_args=["-std=c++20", "-O2", f"-I{ROOT / 'src'}"],
            link_args=["-ldl"],
        )
    except subprocess.CalledProcessError as error:
        pytest.fail(error.stdout + error.stderr, pytrace=False)
    result = subprocess.run(
        [str(binary), *map(str, libraries), str(absent), str(outer_library)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
