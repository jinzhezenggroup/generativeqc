"""Execute the exact stationary owner return paths with a queued host CUDA double.

Scientific children are isolated: this checks failure lifetime and fence placement,
not native numerical or GPU qualification. No source hashes stand in for behavior.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]

_RUNTIME = r"""
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
using cudaStream_t = void*;
using cudaError_t = int;
constexpr int cudaSuccess=0, cudaErrorInvalidResourceHandle=1, cudaErrorUnknown=2;
constexpr int cudaMemcpyDeviceToDevice=0, cudaMemcpyHostToDevice=1;
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
int cudaMalloc(void** p,std::size_t){*p=nullptr;return 0;}
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
void store_column(const double* p,std::size_t n,std::size_t i,double* out){
q::pending.push_back({p,[=]{for(std::size_t j=0;j<n;++j)out[j*n+i]=p[j];}});}
void symmetric(const double*,std::size_t,double,double,int*){
q::pending.push_back({nullptr,[]{}});}
enum class CudaKsWb97mvHessianStatus{failed,response_not_converged,converged};
struct CudaKsWb97mvHessianResult{
CudaKsWb97mvHessianStatus status=CudaKsWb97mvHessianStatus::failed;
bool converged()const{return status==CudaKsWb97mvHessianStatus::converged;}};
struct Atom{double ionic_charge()const{return 1;}};
struct Primary{struct System{std::vector<Atom> atoms{Atom{},Atom{}};} value;
const System& system(){return value;}};
struct Owner{
cudaStream_t stream=nullptr;double storage[32]{};void* arena=storage;
std::size_t bytes=sizeof(storage),direction=0,candidate=2,complete=4,status_offset=10,
coordinates=2,charges=12,pair_atoms=14,pairs=1;
struct{Primary* primary;}fock;Primary primary;
struct{double symmetry_absolute_tolerance=1e-7,symmetry_relative_tolerance=1e-7;}options;
struct{unsigned published_hvps=0,published_hessians=0;}receipt;
Owner(){fock.primary=&primary;}
template<class T=double>T* at(std::size_t p){return reinterpret_cast<T*>(storage+p);}
int admit(const double*,double*,bool,cudaStream_t){return q::reject?23:0;}
int prepare_frozen_exact(){return 0;}
int compute(CudaKsWb97mvHessianResult& r,bool=false,std::size_t=0){
if(q::throw_compute)throw std::runtime_error("child exception");
auto e=cudaMemsetAsync(at(candidate),0,2*sizeof(double),stream);if(e)return e;
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
for(bool full:{false,true}){
q::reset();CudaKsWb97mvHessian owner;owner.impl_->stream=stream;
double input[2]{1,2},output[4]{27,27,27,27};CudaKsWb97mvHessianResult result;
auto run=[&]{return full?owner.hessian(output,stream,result):owner.apply(input,output,stream,result);};
q::require(run()==0&&result.converged()&&q::pending.empty(),"valid action");
q::require(q::fences==(full?4:2),"extra successful-path fence");
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
for(int failure=0;failure<=4;++failure){q::reset();q::fail=failure;
const auto e=prepare_upload(stream);
q::require((e!=0)==(failure!=0),"upload error changed");
q::require(q::pending.empty()&&q::late_staging==0,"staging died before drain");
if(!failure)q::require(q::fences==1,"extra successful upload fence");++controls;}
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
    start = source.index("    check(cudaMalloc(&owner->arena, owner->bytes));")
    uploads = source[start : source.index("    check(owner->geometry.prepare", start)]
    uploads = uploads.replace("std::vector<double>", "TrackedVector<double>")
    uploads = uploads.replace("std::vector<std::size_t>", "TrackedVector<std::size_t>")
    # The upload test borrows a fixed arena; cudaMalloc's stub must not replace it.
    uploads = uploads.replace(
        "check(cudaMalloc(&owner->arena, owner->bytes));", "/* preallocated arena */"
    )
    prepare = (
        "int prepare_upload(cudaStream_t stream){Owner value;auto* owner=&value;"
        "owner->stream=stream;struct{std::size_t atoms=2;}shape;try{"
        + uploads
        + "return 0;}catch(...){return 17;}}\n"
    )
    return _RUNTIME + guard + methods + prepare + _MAIN


def test_stationary_failure_paths_drain_borrows(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a C++20 compiler for the host lifetime fixture")
    source = (_ROOT / "src/dft/cuda_ks_wb97mv_hessian.cu").read_text()
    fixture = tmp_path / "stationary_lifetime.cpp"
    fixture.write_text(_translation_unit(source))
    executable = tmp_path / "stationary_lifetime"
    build = subprocess.run(
        [compiler, "-std=c++20", "-O0", str(fixture), "-o", str(executable)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert build.returncode == 0, build.stderr
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, check=False, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS queued lifetime controls=" in result.stdout
