"""Host-only admission regression; no GPU arithmetic is simulated."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_df_energy_promise_rejects_force_before_work(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    ROOT = tmp_path
    s = (Path(__file__).resolve().parents[2] / "src/methods/dft_method.cpp").read_text()

    def method(signature: str) -> str:
        start = s.index(signature)
        brace = s.index(" {\n", start) + 1
        depth = 1
        i = brace + 1
        while depth:
            depth += (s[i] == "{") - (s[i] == "}")
            i += 1
        return s[start:i]

    guard = method("  [[nodiscard]] bool has_energy_only_df_capacity_promise()")
    direct_ports = []
    for signature, sentinel in [
        ("  generativeqc_status prepared_integral_gradient(", "work"),
        ("  generativeqc_status native_cuda_stationary_gradient_once(", "resources"),
        ("  generativeqc_status density_fitted_integral_gradient(", "work"),
        ("  generativeqc_status mixed_rsh_df_integral_gradient(", "work"),
        ("  generativeqc_status cuda_integral_gradient(", "work"),
    ]:
        actual = method(signature)
        prefix = actual[
            actual.index(" {\n") + 2 : actual.index("    runtime::DeviceCapacityScope")
        ]
        assert "has_energy_only_df_capacity_promise()" in prefix
        # Compile exact entry checks; the unexecuted remainder is a work sentinel.
        name = signature.split("(")[0].split()[-1]
        direct_ports.append(
            "int "
            + name
            + "(std::vector<double>& output, unsigned& "
            + sentinel
            + ", std::string& detail) {"
            + prefix
            + " ++port_allocations; ++port_source_work; return 0; }"
        )
    # The public stationary entry now delegates to a bounded-retry wrapper. Compile
    # that exact wrapper too: an admission rejection must never retire a tile or
    # enter its second attempt.
    stationary_wrapper = method(
        "  generativeqc_status native_cuda_stationary_gradient("
    )
    memory_port = method("  generativeqc_status cuda_prepare_force_memory(")
    assert memory_port.index(
        "has_energy_only_df_capacity_promise()"
    ) < memory_port.index("runtime::DeviceCapacityScope")
    # Energy-only final-state reads are not blanket-disabled.
    assert "has_energy_only_df_capacity_promise" not in method(
        "  generativeqc_status read_final_state("
    )
    props = method("  [[nodiscard]] generativeqc_property_flags supported_properties()")
    start = s.index(
        "    const auto wants_forces =",
        s.index("  std::vector<BatchItemResult> execute_items"),
    )
    end = s.index("    if (!coordinates.empty()", start)
    admission = s[start:end]
    assert s.index(
        "if (has_energy_only_df_capacity_promise()) return false;",
        s.index("bool supports_native_gga_force"),
    ) < s.index(
        "admitted_cuda_stationary_binding();", s.index("bool supports_native_gga_force")
    )
    assert end < s.index("std::vector<BatchItemResult> results(size());", end)
    cpp = (
        r"""
    #include <cassert>
    #include <cstdint>
    #include <memory>
    #include <span>
    #include <stdexcept>
    #include <string>
    #include <vector>
    using generativeqc_property_flags=unsigned;
    using generativeqc_status=int;
    constexpr int GENERATIVEQC_STATUS_OUT_OF_MEMORY=6;
    namespace dft {struct CudaKsFinalStateToken {}; using CudaStationaryResources=unsigned;}
    namespace scf {struct DirectForcePrecisionRequest {}; }
    constexpr unsigned GENERATIVEQC_PROPERTY_FORCES=2, ENERGY=1;
    constexpr int GENERATIVEQC_BACKEND_CUDA=1, GENERATIVEQC_STATUS_NOT_IMPLEMENTED=5;
    namespace scf { enum class FockApproximation {Exact,DensityFitted}; }
    struct Strategy {struct {struct {scf::FockApproximation approximation;} coulomb;} spec;};
    struct Fock {Strategy v;const Strategy& strategy()const{return v;}};
    struct MethodError:std::runtime_error {int status;MethodError(int s,const std::string&m):std::runtime_error(m),status(s){}};
    namespace dft {bool semilocal_family_requires_tau(int){return false;}}
    struct Execution {bool automatic_program=false;int semilocal_family=0;};
    const char* ks_semilocal_family_name(const Execution&){return "PBE";}
    struct State {
     void* capacity_{};int backend_=1;Fock fock_{{{ {scf::FockApproximation::DensityFitted} }}};
     struct {unsigned supported_properties=ENERGY|GENERATIVEQC_PROPERTY_FORCES;} capabilities_;
     bool qualified=true;int port_allocations=0,port_source_work=0;
    """
        + guard
        + r"""
     bool supports_native_gga_force()const noexcept {
      if(has_energy_only_df_capacity_promise()) return false;
      return qualified; // Independent pre-existing qualification, not simulated GPU work.
     }
    """
        + "\n".join(direct_ports)
        + r"""
     unsigned retire_attempts=0;
     bool retire_automatic_force_tile(){++retire_attempts;return true;}
     generativeqc_status native_cuda_stationary_gradient_once(
       const dft::CudaKsFinalStateToken&,std::size_t,std::vector<double>& output,
       dft::CudaStationaryResources& resources,std::string& detail,
       scf::DirectForcePrecisionRequest*) {
       return native_cuda_stationary_gradient_once(output,resources,detail);
     }
    """
        + stationary_wrapper
        + props
        + r"""
    };
    struct Item {State* plan;unsigned prepared_properties;};
    struct Batch {
     std::vector<Item> items_;Execution execution_plan_;int allocations=0,source_work=0;
     explicit Batch(std::vector<Item> x):items_(std::move(x)){}
     std::size_t size()const{return items_.size();}
     void execute(bool compute_forces,std::span<const std::uint8_t> selected_forces={}) {
    """
        + admission
        + r"""
      ++allocations;++source_work; // Sentinels after the exact production preflight.
     }
    };
    int main(){
     State promised,ordinary,direct,cpu;
     promised.capacity_=&promised;
     direct.capacity_=&direct;direct.fock_.v.spec.coulomb.approximation=scf::FockApproximation::Exact;
     cpu.capacity_=&cpu;cpu.backend_=0;
     assert(promised.supported_properties()==ENERGY);
     assert(!promised.supports_native_gga_force());
     for(auto* p:{&ordinary,&direct,&cpu})assert(p->supported_properties()==(ENERGY|GENERATIVEQC_PROPERTY_FORCES));
     // Call owner ports directly, bypassing batch execution/capability metadata.
     for (auto port : {&State::prepared_integral_gradient, &State::density_fitted_integral_gradient, &State::mixed_rsh_df_integral_gradient, &State::cuda_integral_gradient}) {
       std::vector<double> out{42};unsigned receipt=7;std::string detail;
       assert((promised.*port)(out,receipt,detail)==GENERATIVEQC_STATUS_NOT_IMPLEMENTED);
       assert(out.empty()&&receipt==0&&detail.find("energy-only")!=std::string::npos);
       assert(promised.port_allocations==0&&promised.port_source_work==0);
       for (auto* accepted : {&ordinary,&direct,&cpu})
         assert((accepted->*port)(out,receipt,detail)==0);
     }
     // Exercise the actual outer entry, including its short-circuit retry policy.
     for(auto* state:{&promised,&ordinary,&direct,&cpu}) {
       std::vector<double> out{42};unsigned receipt=7;std::string detail;
       const auto status=state->native_cuda_stationary_gradient({},1024,out,receipt,detail);
       if(state==&promised) {
         assert(status==GENERATIVEQC_STATUS_NOT_IMPLEMENTED);
         assert(out.empty()&&receipt==0&&detail.find("energy-only")!=std::string::npos);
         assert(state->port_allocations==0&&state->port_source_work==0);
       } else assert(status==0&&state->port_source_work==5);
       assert(state->retire_attempts==0);
     }
     Batch b{{{&promised,promised.supported_properties()}}};
     b.execute(false);assert(b.allocations==1&&b.source_work==1);
     try {b.execute(true);assert(false);}catch(const MethodError&e){assert(e.status==GENERATIVEQC_STATUS_NOT_IMPLEMENTED);assert(std::string(e.what()).find("energy-only")!=std::string::npos);}
     assert(b.allocations==1&&b.source_work==1);
     // Even stale permissive method metadata cannot bypass the explicit prepared guard.
     b.items_[0].prepared_properties=ENERGY|GENERATIVEQC_PROPERTY_FORCES;
     try {b.execute(true);assert(false);}catch(const MethodError&){}
     assert(b.allocations==1&&b.source_work==1);
     Batch allowed{{{&ordinary,ordinary.supported_properties()}}};allowed.execute(true);assert(allowed.source_work==1);
     Batch mixed{{{&promised,promised.supported_properties()},{&ordinary,ordinary.supported_properties()}}};
     const std::uint8_t good[]={0,1},bad[]={1,0};
     mixed.execute(false,good);assert(mixed.source_work==1);
     try {mixed.execute(false,bad);assert(false);}catch(const MethodError&){}
     assert(mixed.source_work==1&&mixed.allocations==1);
     // No promise is created or destroyed by capability/preflight decisions.
     assert(promised.capacity_==&promised&&ordinary.capacity_==nullptr);
    }
    """
    )
    (ROOT / "probe.cpp").write_text(cpp)
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(ROOT / "probe.cpp"),
            "-o",
            str(ROOT / "probe"),
        ],
        check=True,
    )
    subprocess.run([str(ROOT / "probe")], check=True)
