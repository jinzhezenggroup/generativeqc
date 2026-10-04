"""Two-component FP64 arithmetic for cancellation-sensitive scalar consumers.

The expansion uses ordinary binary64 storage and operations, including exact
FMA product residuals. It is neither a wider native dtype nor arbitrary
precision. Callers retain their own finite/range and resource contracts.
"""


def emit_fp64_expansion() -> str:
    """Emit a namespace-local expansion; explicit rounding survives NVCC FMA.

    Addition and multiplication retain the leading rounding residual. Division
    and square root use one residual correction. The exponential is range
    reduced with a split ln(2), then evaluated by a bounded Taylor series.
    Only nonpositive exponential arguments are needed by Gaussian consumers.
    """
    return r"""
namespace fp64_expansion {
__device__ inline double add(double a,double b) {
#ifdef __CUDA_ARCH__
  return __dadd_rn(a,b);
#else
  volatile double r=a+b;return r;
#endif
}
__device__ inline double mul(double a,double b) {
#ifdef __CUDA_ARCH__
  return __dmul_rn(a,b);
#else
  volatile double r=a*b;return r;
#endif
}
struct Wide {
  double hi{},lo{};
  __device__ Wide()=default;
  __device__ Wide(double h,double l=0):hi(h),lo(l){}
  __device__ double value() const {return add(hi,lo);}
};
__device__ inline Wide operator+(Wide a,Wide b) {
  const double s=add(a.hi,b.hi),v=add(s,-a.hi);
  const double e=add(add(add(a.hi,-add(s,-v)),add(b.hi,-v)),add(a.lo,b.lo));
  const double h=add(s,e);return {h,add(e,-add(h,-s))};
}
__device__ inline Wide operator-(Wide a) {return {-a.hi,-a.lo};}
__device__ inline Wide operator-(Wide a,Wide b) {return a+(-b);}
__device__ inline Wide operator*(Wide a,Wide b) {
  const double p=mul(a.hi,b.hi);
  const double e=add(add(fma(a.hi,b.hi,-p),mul(a.hi,b.lo)),mul(a.lo,b.hi));
  const double h=add(p,e);return {h,add(e,-add(h,-p))};
}
__device__ inline Wide operator/(Wide a,Wide b) {
  const double q=a.hi/b.hi;
  const Wide r=a-Wide(q)*b;return Wide(q)+r.value()/b.hi;
}
__device__ inline Wide& operator+=(Wide& a,Wide b) {a=a+b;return a;}
__device__ inline Wide& operator*=(Wide& a,Wide b) {a=a*b;return a;}
__device__ inline bool operator<(Wide a,Wide b) {
  return a.hi<b.hi || (a.hi==b.hi && a.lo<b.lo);
}
__device__ inline Wide sqrt(Wide a) {
  if(a.hi==0) return 0;
  const double r=::sqrt(a.hi);return Wide(r)+(a-Wide(r)*r)/(2*r);
}
__device__ inline Wide exp(Wide x) {
  // Preserve Gaussian underflow without narrowing an unbounded exponent to int.
  if(x.hi < -746) return 0;
  const Wide ln2{0.6931471805599453,2.3190468138462996e-17};
  const int n=static_cast<int>(nearbyint(x.hi/ln2.hi));
  const Wide r=x-ln2*n;
  Wide term=1,sum=1;
  for(unsigned k=1;k<=28;++k) {term=term*r/k;sum+=term;}
  return {ldexp(sum.hi,n),ldexp(sum.lo,n)};
}
__device__ inline Wide erf(Wide x) {
  // This overload is used only by the large-T Boys branch (x >= sqrt(30)).
  // erfc preserves the tiny tail that would disappear in a rounded erf(x).
  const Wide correction=Wide(1.1283791670955126,1.533545961316588e-17)
      *exp(-x*x)*x.lo;
  return Wide(1)-Wide(::erfc(x.hi))+correction;
}
} // namespace fp64_expansion
"""
