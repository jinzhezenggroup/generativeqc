"""Execute the exact stationary owner return paths with a queued host CUDA double.

Scientific children are isolated: this checks failure lifetime and fence placement,
not native numerical or GPU qualification. No source hashes stand in for behavior.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx

_ROOT = Path(__file__).resolve().parents[2]

_RUNTIME = r"""
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>
using cudaStream_t = void*;
using cudaError_t = int;
constexpr int cudaSuccess=0, cudaErrorInvalidResourceHandle=1, cudaErrorUnknown=2, cudaErrorInvalidValue=3;
constexpr int cudaMemcpyDeviceToDevice=0, cudaMemcpyHostToDevice=1, cudaMemcpyDeviceToHost=2;
namespace q {
struct Task {const void* source; std::function<void()> run;};
std::vector<Task> pending;
int calls=0, fail=0, fences=0, launches=0, fail_launch=0, late_staging=0;
bool throw_compute=false, converge=true, reject=false;
void reset() {if(!pending.empty())throw std::runtime_error("undrained work");
calls=fail=fences=launches=fail_launch=late_staging=0;
throw_compute=reject=false;converge=true;}
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
}
int cudaMemsetAsync(void* p,int v,std::size_t n,cudaStream_t){
if(++q::calls==q::fail)return 17;
q::pending.push_back({nullptr,[=]{std::memset(p,v,n);}});return 0;}
int cudaMemcpyAsync(void* p,const void* v,std::size_t n,int,cudaStream_t){
if(++q::calls==q::fail)return 17;
q::pending.push_back({v,[=]{std::memcpy(p,v,n);}});return 0;}
int cudaStreamSynchronize(cudaStream_t){++q::fences;
auto tasks=std::move(q::pending);q::pending.clear();
for(auto& task:tasks)task.run();return ++q::calls==q::fail?17:0;}
int cudaGetLastError(){return ++q::launches==q::fail_launch?19:0;}
void check(int error){if(error)throw std::runtime_error("injected failure");}
std::size_t product(std::size_t a,std::size_t b){return a*b;}
// Instrument staging destruction without dereferencing storage after its lifetime.
template<class T>struct TrackedVector:std::vector<T>{
using std::vector<T>::vector;
~TrackedVector(){for(auto& task:q::pending)if(task.source==this->data()){
++q::late_staging;task.run=[]{};}}};
namespace maps {unsigned blocks(std::size_t n){return unsigned(n);}}
void unit(double* p,std::size_t n,std::size_t i){
q::pending.push_back({nullptr,[=]{std::fill(p,p+n,0);p[i]=1;}});}
namespace response {
enum class CudaGmresStatus {running, converged};
enum class CudaCpksStatus {running, converged, linear_solve_failed};
enum class CudaCpksResidentPrecision {fp64};
struct CudaGmresReceipt{
int code{};std::size_t iterations{},restarts{};
double rhs_norm{},residual_norm{},tolerance{};
std::size_t operator_actions{},preconditioner_actions{};CudaGmresStatus status{};};
struct CudaCpksDenseReceipt{
std::size_t iterations{};CudaGmresReceipt last;
CudaCpksStatus failure{};};
struct CudaCpksDenseReplay{CudaCpksDenseReceipt* receipt;int* column_status;struct{} solver;};
}
struct DenseAssemblyAudit{
std::size_t completed;int failed;response::CudaCpksDenseReceipt cpks;};
void store_column(const double* p,std::size_t n,std::size_t i,const int* status,
double* out,DenseAssemblyAudit* audit){
q::pending.push_back({p,[=]{if(*status){audit->failed=1;return;}
++audit->completed;for(std::size_t j=0;j<n;++j)out[j*n+i]=p[j];}});}
namespace core {constexpr int kCartesianBasis=1;}
void symmetric(const double*,std::size_t,double,double,int*){
q::pending.push_back({nullptr,[]{}});}
enum class CudaKsWb97mvHessianStatus{failed,response_not_converged,converged};
enum class CudaKsWb97mvHessianMode{dense_and_action};
struct CudaKsWb97mvHessianResult{
CudaKsWb97mvHessianStatus status=CudaKsWb97mvHessianStatus::failed;
const char* failure_check{};
std::size_t completed_directions{};
struct{response::CudaGmresReceipt gmres;response::CudaCpksStatus status{};}response;
bool converged()const{return status==CudaKsWb97mvHessianStatus::converged;}};
struct Atom{int atomic_number{1};std::array<double,3> position{};double ionic_charge()const{return 1;}};
struct Primary{struct System{std::vector<Atom> atoms{Atom{},Atom{}};
int basis_representation=core::kCartesianBasis;} value;
const System& system(){return value;}
struct Strategy{struct{struct{bool present{};}exchange;}spec;};
Strategy strategy(){return {};}};
enum class Correction {none, d3, d4};
struct D4Correction {int profile()const{return 0;}};
namespace dispersion {
inline const double table[]{1};
struct D4Tables {
const double *elements{table}, *references{table}, *reference_c6{table};
std::size_t element_count{1}, reference_count{1}, reference_c6_count{1};};
struct ChargeTables {const double *elements{table}, *charge_elements{table};std::size_t element_count{1};};
D4Tables eeq_d4_host_tables(int){return {};}
ChargeTables eeq2019_host_tables(){return {};}
}
struct Owner{
cudaStream_t stream=nullptr;double storage[48]{};void* arena=storage;
std::size_t bytes=sizeof(storage),direction=0,candidate=2,complete=4,status_offset=10,
coordinates=2,charges=12,pair_atoms=14,pairs=1,correction_geometry=16,
d4_numbers=22,d4_elements=24,d4_references=25,d4_c6=26,eeq_parameters=27,dense_audit_offset=32;
CudaKsWb97mvHessianMode mode{CudaKsWb97mvHessianMode::dense_and_action};
struct{struct{struct{bool nonlocal_grid{};}model;}identity;}expected;
struct Geometry{int admit_actions(std::size_t){return 0;}}geometry,vv10_geometry;
struct{bool d3{};std::optional<D4Correction> d4;}corrections;
dispersion::D4Tables d4_tables;
dispersion::ChargeTables eeq_tables;
bool has_df{},has_ecp{},operator_matrix{};
struct{unsigned spins{1};}frame;
struct{Primary* primary;void* correction{};}fock;Primary primary;
struct{void collect_dense(const response::CudaCpksDenseReceipt&) {}}cpks_operator;
struct Induced{
Owner* owner;
int collect_work(){
double receipt{};
const auto e=cudaMemcpyAsync(&receipt,owner->at(owner->candidate),sizeof(receipt),
cudaMemcpyDeviceToHost,owner->stream);
const auto completed=cudaStreamSynchronize(owner->stream);
return e?e:completed;}}
induced{this};
struct{double symmetry_absolute_tolerance=1e-7,symmetry_relative_tolerance=1e-7;
struct{response::CudaCpksResidentPrecision resident_precision{};
struct{std::size_t max_iterations{20};}gmres;}cpks;}options;
struct{unsigned published_hvps=0,published_hessians=0,completed_directions=0,cpks_strict_iterations=0;}receipt;
Owner(){fock.primary=&primary;}
template<class T=double>T* at(std::size_t p){return reinterpret_cast<T*>(storage+p);}
int admit(const double*,double*,bool,cudaStream_t){return q::reject?23:0;}
int prepare_frozen_exact(){return 0;}
int compute(CudaKsWb97mvHessianResult& r,bool=false,std::size_t=0,
response::CudaCpksDenseReplay* replay=nullptr){
if(q::throw_compute)throw std::runtime_error("child exception");
auto e=cudaMemsetAsync(at(candidate),0,2*sizeof(double),stream);if(e)return e;
if(replay){
q::pending.push_back({nullptr,[=]{
*replay->column_status=q::converge?0:1;
replay->receipt->last.code=int(response::CudaGmresStatus::converged);
if(!q::converge)replay->receipt->failure=response::CudaCpksStatus::linear_solve_failed;
}});r.status=CudaKsWb97mvHessianStatus::converged;return 0;}
if(!q::converge){r.status=CudaKsWb97mvHessianStatus::response_not_converged;return 0;}
e=cudaStreamSynchronize(stream);if(!e)r.status=CudaKsWb97mvHessianStatus::converged;
return e;}
int gate(){return cudaStreamSynchronize(stream);}int authenticate(){return 0;}
};
struct CudaKsWb97mvHessian{
std::unique_ptr<Owner> impl_=std::make_unique<Owner>();
int apply(const double*,double*,cudaStream_t,CudaKsWb97mvHessianResult&)noexcept;
int hessian(double*,cudaStream_t,CudaKsWb97mvHessianResult&)noexcept;
};
"""

_MAIN = r"""
int main(){try{
int controls=0;
for(auto stream:{static_cast<void*>(nullptr),reinterpret_cast<void*>(8)}){
for(int mode:{0,1,2}){
const bool full=mode!=0,deferred=mode==2;
q::reset();CudaKsWb97mvHessian owner;owner.impl_->stream=stream;owner.impl_->has_df=!deferred;
double input[2]{1,2},output[4]{27,27,27,27};CudaKsWb97mvHessianResult result;
auto run=[&]{return full?owner.hessian(output,stream,result):owner.apply(input,output,stream,result);};
q::require(run()==0&&result.converged()&&q::pending.empty(),"valid action");
q::require(q::fences==(full?(deferred?3:5):3),"extra successful-path fence");
const int calls=q::calls,launches=q::launches;
const auto prior_hvps=owner.impl_->receipt.published_hvps,
prior_hessians=owner.impl_->receipt.published_hessians;
for(int failure=1;failure<=calls;++failure){q::reset();q::fail=failure;
std::fill(output,output+4,27);auto e=run();
q::require(e!=0,"injected API failure was hidden");
q::require(q::pending.empty(),"failure retained borrowed work");
q::require(owner.impl_->receipt.published_hvps==prior_hvps&&
owner.impl_->receipt.published_hessians==prior_hessians,"failed publication counted");++controls;}
for(int failure=1;failure<=launches;++failure){q::reset();q::fail_launch=failure;
std::fill(output,output+4,27);q::require(run()==19,"launch error changed");
q::require(q::pending.empty(),"launch failure retained work");
q::require(std::all_of(output,output+4,[](double x){return x==27;}),"early output changed");++controls;}
for(bool exception:{false,true}){q::reset();q::throw_compute=exception;q::converge=false;
std::fill(output,output+4,27);const auto e=run();
q::require(e==(exception?cudaErrorUnknown:0)&&!result.converged(),"child status changed");
q::require(q::pending.empty(),"child failure retained work");
q::require(std::all_of(output,output+4,[](double x){return x==27;}),"rejected output changed");++controls;}
q::reset();q::reject=true;q::require(run()==23&&q::fences==0&&q::calls==0,"admission queued work");++controls;
}
for(auto correction:{Correction::none,Correction::d3,Correction::d4}){
q::reset();q::require(prepare_upload(stream,correction)==0,"valid upload");
q::require(q::pending.empty()&&q::late_staging==0,"successful upload retained work");
q::require(q::fences==(correction==Correction::none?1:2),"extra successful upload fence");
const int uploads=q::calls;
for(int failure=1;failure<=uploads;++failure){q::reset();q::fail=failure;
q::require(prepare_upload(stream,correction)!=0,"upload error was hidden");
q::require(q::pending.empty()&&q::late_staging==0,"staging died before drain");++controls;}
}
}
std::cout<<"PASS queued lifetime controls="<<controls<<"\n";
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
"""


def _translation_unit(source: str) -> str:
    guard = ""
    if "struct DrainOnFailure {" in source:
        start = source.index("struct DrainOnFailure {")
        guard = source[start : source.index("\n};", start) + 3]
    start = source.index("cudaError_t CudaKsWb97mvHessian::apply(")
    methods = source[start : source.index("CudaKsWb97mvHessianWork ", start)]
    # Only CUDA launch syntax is removed; integer arguments and method bodies stay.
    methods = re.sub(r"<<<.*?>>>", "", methods, flags=re.DOTALL)
    start = source.index("    std::vector<double> charges;")
    uploads = source[start : source.index("    check(owner->geometry.prepare", start)]
    uploads = uploads.replace("std::vector<double>", "TrackedVector<double>")
    uploads = uploads.replace("std::vector<std::size_t>", "TrackedVector<std::size_t>")
    uploads = uploads.replace(
        "std::vector<std::int32_t>", "TrackedVector<std::int32_t>"
    )
    prepare = (
        "int prepare_upload(cudaStream_t stream,Correction correction){Owner value;auto* owner=&value;"
        "owner->corrections.d3=correction==Correction::d3;"
        "if(correction==Correction::d4)owner->corrections.d4.emplace();"
        "owner->stream=stream;struct{std::size_t atoms=2;}shape;try{"
        + uploads
        + "return 0;}catch(...){return 17;}}\n"
    )
    return _RUNTIME + guard + methods + prepare + _MAIN


def test_stationary_failure_paths_drain_borrows(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    source = (_ROOT / "src/dft/cuda_ks_wb97mv_hessian.cu").read_text()
    fixture = tmp_path / "stationary_lifetime.cpp"
    fixture.write_text(_translation_unit(source))
    executable = native_cxx.build_executable(
        [fixture], tmp_path / "stationary_lifetime", compile_args=("-std=c++20", "-O0")
    )
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, check=False, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS queued lifetime controls=" in result.stdout
