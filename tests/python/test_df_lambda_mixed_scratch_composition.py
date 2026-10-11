"""Compose independent serial scratch and FP32 epochs using live owner code.

These host probes execute production admission and pointer bindings with synthetic
capacity queries. They do not model scientific CUDA kernels or GPU qualification.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from test_df_cc_lambda import case, run
from test_df_lambda_serial_scratch import PREFIX

ROOT = Path(__file__).resolve().parents[2]


def _compile(folder: Path, source: str) -> Path:
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if not compiler or not cache:
        pytest.skip("host C++ compiler and ccache required")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    unit, output = folder / "composition.cpp", folder / "composition"
    obj = folder / "composition.o"
    unit.write_text(source)
    result = subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O0",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            "-c",
            str(unit),
            "-o",
            str(obj),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    subprocess.run(
        [compiler, str(obj), "-o", str(output)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return output


@pytest.fixture(scope="module")
def selector_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    source = (ROOT / "tests/native/df_cc_lambda_probe.cpp").read_text()
    controls = source[
        source.index("    response.serial_scratch_reuse =") : source.index(
            "    response.df_audit_matrix_gemm ="
        )
    ]
    return _compile(
        tmp_path_factory.mktemp("lambda-composed-masks"),
        r"""
#include <cmath>
#include <limits>
#include <string>
#include "cc/lambda_response.hpp"
int main(int argc,char** argv) {
  if(argc!=7)return 1;
  int mode=std::stoi(argv[1]);
  generativeqc::cc::LambdaOptions response;
"""
        + controls
        + r"""
  const bool sharing=std::stoi(argv[2]), mixed=std::stoi(argv[3]),
             limit=std::stoi(argv[4]), invalid=std::stoi(argv[5]), resident=std::stoi(argv[6]);
  if(response.serial_scratch_reuse!=sharing || response.df_mixed_precision!=mixed)return 2;
  if(response.df_mixed_precision_max_iterations!=(invalid?0:limit?2:16))return 3;
  if(std::isnan(response.df_mixed_precision_warmup_tolerance)!=invalid)return 4;
  if(response.cuda_resident_krylov!=resident)return 5;
}
""",
    )


@pytest.mark.parametrize("mask", range(32))
def test_probe_controls_have_independent_bits(selector_probe: Path, mask: int) -> None:
    """The Python packer and C++ decoder keep scratch, warmup and residency independent."""
    sharing, mixed, limit, invalid, resident = (
        bool(mask & (1 << bit)) for bit in range(5)
    )
    _, _, arrays = case(1, 2)
    observed = []

    def capture(*args: object) -> int:
        observed.append(args[3])
        return 0

    run(
        capture,
        arrays,
        serial_scratch_reuse=sharing,
        mixed_precision=mixed,
        mixed_limit_two=limit,
        invalid_mixed_controls=invalid,
        resident_krylov=resident,
    )
    mode = observed[0]
    assert isinstance(mode, int)
    assert mode & 0x1F000 == mask << 12
    subprocess.run(
        [
            str(selector_probe),
            str(mode),
            *(str(int(value)) for value in (sharing, mixed, limit, invalid, resident)),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.fixture(scope="module")
def scratch_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    source = (ROOT / "src/cc/df_lambda_cuda.cu").read_text()
    start = source.index("    const std::array<const std::vector<double>*, 16> host")
    planner = source[
        start : source.index("    cuda_check(cudaStreamCreateWithFlags", start)
    ]
    start = source.index("    auto at =")
    binding = source[start : source.index("      if (metrics.df_matrix_gemm)", start)]
    # Preserve the live retained-frontier binding, excluding unrelated GPU work.
    retained = re.search(r"^\s*staged.core_reuse_arena = [^;]+;", source, re.MULTILINE)
    audit_binding = re.search(r"^\s*staged.audit_arena = [^;]+;", source, re.MULTILINE)
    assert retained is not None and audit_binding is not None
    start = source.index(
        "    if (options.df_mixed_precision && metrics.df_matrix_gemm)"
    )
    mixed = source[start : source.index("    metrics.cuda_actions = true;", start)]
    definitions = []
    for namespace in ("generated_core", "generated_response", "generated_virtual"):
        functions = []
        for name in sorted(set(re.findall(namespace + r"::(\w+)\(", planner + mixed))):
            if name == "prepare_mixed_contractions":
                continue
            value = "128+padding"
            if "staged_" in name:
                value = "256+padding"
            if "contraction_terms" in name:
                value = (
                    "1000"
                    if name
                    in (
                        "transpose_contraction_terms",
                        "virtual_amplitude_vjp_contraction_terms",
                    )
                    else "1"
                )
            if "host_bytes" in name:
                value = "64"
            if name == "staged_core_reuse_arena_elements":
                value = "64+padding"
            if name == "staged_core_reuse_scratch_arena_elements":
                value = "640+padding"
            if name == "mixed_auxiliary_arena_elements":
                value = "mixed_large?100000:256+padding"
            functions.append(
                f"template<class... T> std::size_t {name}(T...){{return {value};}}"
            )
        definitions.append("namespace " + namespace + "{" + "\n".join(functions) + "}")
    # Retained dynamic scratch deliberately exceeds the ordinary core high-water
    # mark, while another matrix stage supplies enough total admitted scratch.
    prefix = PREFIX.replace("auxiliary_stage{512}", "auxiliary_stage{768}")
    return _compile(
        tmp_path_factory.mktemp("lambda-mixed-scratch"),
        prefix
        + r"""
#include <cstring>
#include <limits>
bool mixed_large=false;
int descriptor_failure=0;
namespace tensor { struct ContractionPreparationUnavailable : std::exception {}; }
namespace generated_response {
struct CudaState {
  double *foo{},*fov{},*fvv{},*ovov{},*ovvo{},*oovv{},*ovvv{},*ovoo{},*oooo{},*vvvv{},
         *d1{},*d2{},*t1{},*t2{},*response_arena{},*replay_arena{},*audit_arena{},
         *bar_correlation_energy{},*bar_singles_residual{},*bar_doubles_residual{},
         *df_virtual_singles{},*df_virtual_doubles{};
  const double *df_Lvv{},*df_Wvoov{},*df_Wvovo{},*df_Xv{},*df_D05_vv_ladder{},
               *df_singles_residual{},*bar_df_Lvv{},*bar_df_Wvoov{},*bar_df_Wvovo{},
               *bar_df_Xv{},*bar_df_D05_vv_ladder{},*bar_df_singles_residual{},
               *df_tau{},*bar_df_tau{},*core_reuse_arena{};
  int* error{};
  std::size_t o{},v{};
  int stream{};
};
struct StagedCudaState : CudaState {};
struct MixedState {
  std::size_t o{},v{};int stream{};int* error{};
  float* response_arena{};
  const float *t1{},*df_tau{},*bar_df_Lvv{},*bar_df_Wvoov{},*bar_df_Wvovo{},
              *bar_df_Xv{},*bar_df_D05_vv_ladder{},*bar_df_singles_residual{};
  std::array<int,1> contractions{};
};
constexpr const char* mixed_auxiliary_program_hash="synthetic-query-only";
template<class... T> void prepare_mixed_contractions(T&&...) {
  if(descriptor_failure==1)throw tensor::ContractionPreparationUnavailable{};
  if(descriptor_failure==2)throw std::bad_alloc{};
  if(descriptor_failure==3)throw std::runtime_error("nonfinite descriptor failure");
}
}
struct Auxiliary { std::size_t o{},v{};int stream{};int* error{};
  double *response_arena{},*t1{},*t2{},*bar_df_virtual_singles{},*bar_df_virtual_doubles{}; };
void upload(double*,const double*,std::size_t){}
"""
        + "\n".join(definitions)
        + r"""
int main(int argc,char** argv) {
  if(argc!=8)return 1;
  const bool sharing=std::stoi(argv[1]), warming=std::stoi(argv[2]);
  const unsigned rollback=std::stoi(argv[3]);
  mixed_large=std::stoi(argv[4]); descriptor_failure=std::stoi(argv[5]);
  padding=std::stoul(argv[6]);
  const bool audit=std::stoi(argv[7]);
  generativeqc::cc::Problem p;
  generativeqc::cc::SolverResult cc;
  p.foo.resize(4);p.fov.resize(6);p.fvv.resize(9);p.ovov.resize(36);
  p.d1.resize(6);p.d2.resize(36);p.df_bov.resize(30);p.df_bvv.resize(45);
  cc.t1.resize(6);cc.t2.resize(36);
  generativeqc::cc::LambdaOptions options;
  options.serial_scratch_reuse=sharing;options.df_mixed_precision=warming;
  options.df_auxiliary_reduction=true;options.df_matrix_gemm=true;
  options.df_audit_matrix_gemm=audit;options.df_core_reuse=true;
  options.max_bytes=1ULL<<30;
  const std::size_t o=2,v=3,q=5,n1=o*v,n2=n1*n1,vv=v*v;
  bool with_source=true,with_parameters=true,parameters_admitted=false;
  std::array<double*,6> cuts{},cut_seeds{};
  std::array<std::size_t,6> cut_sizes{};
  generativeqc::cc::LambdaDiagnostic metrics;
"""
        + planner
        + r"""
  if(rollback==1 || rollback==2)drop_core_reuse();
  if(rollback==2)drop_audit();
  if(rollback==3)scalar_plan();
  if(rollback==4){scalar_plan();metrics.df_auxiliary_reduction=false;
    cursor=fallback_cursor;scratch_elements=scratch;capacity();}
  const auto allocated=allocation_bytes,owned=metrics.owned_device_bytes,
             admitted=metrics.numeric_capacity_bytes;
  std::vector<unsigned char> allocation(allocation_bytes,0);
  struct { unsigned char* base;int stream=0,contractions=0; } storage{allocation.data()};
  generated_response::CudaState state;
  generated_response::StagedCudaState staged;
  generated_response::MixedState mixed;
  Auxiliary auxiliary;
  const double *bov{},*bvv{};
  double *tau{},*tau_seed{},*sum1{},*sum2{};
  float *mixed_bov{},*mixed_bvv{};
  std::array<float*,6> mixed_cut_seeds{};
"""
        + binding
        + "\n    }\n    if(metrics.df_core_reuse){"
        + retained[0]
        + "}\n    if(metrics.df_audit_matrix_gemm){"
        + audit_binding[0]
        + r"""}
  auto offset=[&](const void* p){return static_cast<const unsigned char*>(p)-storage.base;};
  struct Range {std::size_t first,last;};
  std::vector<Range> persistent;
  auto add=[&](const void* p,std::size_t size){if(size){
    const auto first=static_cast<std::size_t>(offset(p));
    if(first+size>allocated)throw std::runtime_error("pointer beyond allocation");
    for(const auto r:persistent)if(first<r.last && r.first<first+size)
      throw std::runtime_error("simultaneously live persistent slots overlap");
    persistent.push_back({first,first+size});}};
  for(std::size_t x=0;x<host.size();++x)add(at(offsets[x]),bytes(host[x]->size()));
  add(sum1,bytes(n1));add(sum2,bytes(n2));add(state.bar_correlation_energy,8);
  add(state.bar_singles_residual,bytes(n1));add(state.bar_doubles_residual,bytes(n2));
  add(state.error,sizeof(int));
  if(metrics.df_auxiliary_reduction){
    add(tau,bytes(n2));add(tau_seed,bytes(n2));
    for(std::size_t x=0;x<cuts.size();++x){add(cuts[x],bytes(cut_sizes[x]));
      add(cut_seeds[x],bytes(cut_sizes[x]));}}
  if(metrics.df_core_reuse)add(staged.core_reuse_arena,metrics.df_core_reuse_bytes);
  auto scratch_safe=[&](std::size_t first,std::size_t size){
    if(first+size>allocated)throw std::runtime_error("scratch exceeds allocation");
    for(const auto r:persistent)if(first<r.last && r.first<first+size)
      throw std::runtime_error("scratch aliases persistent slot");};
  scratch_safe(arena,bytes(sharing?scratch_elements:scratch));
  if(metrics.df_matrix_gemm){
    scratch_safe(offset(staged.response_arena),
                 std::max(bytes(matrix_scratch_elements),metrics.df_audit_arena_bytes));
    if(metrics.df_audit_matrix_gemm && staged.audit_arena!=staged.response_arena)return 2;
    if(sharing && arena!=std::size_t(offset(staged.response_arena)))return 3;
  }
  bool propagated=false;
  try {
"""
        + mixed
        + r"""
  }catch(const std::runtime_error& e){
    if(std::string(e.what())!="nonfinite descriptor failure")throw;
    propagated=true;
  }
  const bool fits=warming && metrics.df_matrix_gemm && !mixed_large &&
                  (!metrics.df_core_reuse || (audit && rollback<2));
  if(propagated!=(fits && descriptor_failure==3))return 4;
  if(!propagated && metrics.df_mixed_precision!=(fits && !descriptor_failure))return 5;
  if(allocation_bytes!=allocated || metrics.owned_device_bytes!=owned)return 6;
  if(!metrics.df_mixed_precision && metrics.numeric_capacity_bytes!=admitted)return 7;
  if(metrics.df_mixed_precision){
    if(mixed.response_arena!=reinterpret_cast<float*>(staged.response_arena))return 8;
    const auto first=offset(mixed_bov), scratch_first=offset(staged.response_arena);
    if(first!=scratch_first+metrics.df_mixed_input_offset_bytes)return 9;
    const auto expected_high=metrics.df_core_reuse?bytes(640+padding):bytes(512+padding);
    if(metrics.df_mixed_input_offset_bytes!=expected_high)return 10;
    scratch_safe(first,metrics.df_mixed_storage_bytes-metrics.df_mixed_input_offset_bytes);
    if(metrics.df_mixed_storage_bytes>std::max(bytes(matrix_scratch_elements),metrics.df_audit_arena_bytes))return 11;
    if(metrics.numeric_capacity_bytes!=admitted+64+bytes(dim))return 12;
    // Fill the real borrowed input tail, then overwrite every live warmup
    // scratch byte. The cached epoch and all pinned values must survive.
    const auto end=scratch_first+metrics.df_mixed_storage_bytes;
    std::memset(storage.base+first,0x5a,end-first);
    std::memset(storage.base+scratch_first,0xa5,metrics.df_mixed_input_offset_bytes);
    for(auto i=first;i<end;++i)if(storage.base[i]!=0x5a)return 13;
    for(const auto r:persistent)for(auto i=r.first;i<r.last;++i)
      if(storage.base[i]!=0)return 14;
  }
}
""",
    )


@pytest.mark.parametrize("sharing", [False, True])
@pytest.mark.parametrize("warming", [False, True])
@pytest.mark.parametrize("rollback", range(5))
@pytest.mark.parametrize("audit", [False, True])
@pytest.mark.parametrize("padding", [0, 1, 31])
def test_real_planner_mixed_epoch_never_overlaps_live_storage(
    scratch_probe: Path,
    sharing: bool,
    warming: bool,
    rollback: int,
    audit: bool,
    padding: int,
) -> None:
    subprocess.run(
        [
            str(scratch_probe),
            str(int(sharing)),
            str(int(warming)),
            str(rollback),
            "0",
            "0",
            str(padding),
            str(int(audit)),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.mark.parametrize("sharing", [False, True])
@pytest.mark.parametrize(
    "large,failure", [(True, 0), (False, 1), (False, 2), (False, 3)]
)
def test_refusal_stays_bounded_and_numerical_failure_propagates(
    scratch_probe: Path,
    sharing: bool,
    large: bool,
    failure: int,
) -> None:
    subprocess.run(
        [
            str(scratch_probe),
            str(int(sharing)),
            "1",
            "0",
            str(int(large)),
            str(failure),
            "0",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.mark.parametrize("ccsd_mixed", [False, True])
@pytest.mark.parametrize("lambda_mixed", [False, True])
@pytest.mark.parametrize("triples_mixed", [False, True])
@pytest.mark.parametrize("interval", [None, 1])
def test_complete_endpoint_helper_keeps_selector_positions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ccsd_mixed: bool,
    lambda_mixed: bool,
    triples_mixed: bool,
    interval: int | None,
) -> None:
    """Capture the real GPU-test command without a process or numerical oracle."""
    import test_df_triples_mixed_endpoint as endpoint

    calls = []

    def capture(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append(command)
        Path(command[2]).write_text("{}")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setenv("SLURM_JOB_ID", "host-command-capture-only")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("GENERATIVEQC_DF_FORCE_ENDPOINT_BINARY", "not-executed")
    monkeypatch.setattr(endpoint, "write_input", lambda *_args: None)
    monkeypatch.setattr(endpoint.subprocess, "run", capture)
    assert (
        endpoint.run_endpoint(
            tmp_path,
            {},
            mixed=triples_mixed,
            ccsd_mixed=ccsd_mixed,
            lambda_mixed=lambda_mixed,
            lambda_interval=interval,
        )
        == {}
    )
    assert len(calls) == 1
    command = calls[0]
    assert command[23] == str(int(triples_mixed))
    if ccsd_mixed or lambda_mixed:
        assert command[24] == str(30 if interval is None else interval)
        assert command[25:30] == ["1", "1", "1", "0", "0"]
        assert command[30] == str(int(ccsd_mixed))
        assert command[31:] == (["1"] if lambda_mixed else [])
    else:
        assert command[24:] == ([] if interval is None else [str(interval)])
