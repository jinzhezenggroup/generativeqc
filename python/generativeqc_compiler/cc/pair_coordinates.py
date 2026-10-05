"""Storage coordinates for real restricted doubles, never new CC equations.

The involution (i,j,a,b)->(j,i,b,a) has ov fixed points and
(o²v²-ov)/2 two-element orbits. A stored representative therefore has metric
weight one or two. Flatten the pair labels x=iv+a,y=jv+b and retain x<=y.
Residual rounding asymmetry is a separate, explicitly gated projection policy;
arbitrary supplied initial amplitudes are not assumed symmetric.
"""

from dataclasses import dataclass
from fractions import Fraction

from generativeqc_compiler.tensor import Program, TensorSpec, add, input_tensor
from generativeqc_compiler.tensor.scalar_cpp import emit_scalar_cpp


@dataclass(frozen=True)
class RestrictedPairCoordinates:
    nocc: int
    nvir: int

    def __post_init__(self) -> None:
        if any(type(n) is not int or n < 1 for n in (self.nocc, self.nvir)):
            raise ValueError("restricted pair dimensions must be positive")

    @property
    def pairs(self) -> int:
        return self.nocc * self.nvir

    @property
    def packed_doubles(self) -> int:
        return self.pairs * (self.pairs + 1) // 2

    def orbit(self, flat: int) -> tuple[int, int, int]:
        """Return partner in ijab order, packed slot and full-metric weight."""
        o, v, n = self.nocc, self.nvir, self.pairs
        if type(flat) is not int or not 0 <= flat < n * n:
            raise ValueError("restricted doubles coordinate out of range")
        i, tail = divmod(flat, o * v * v)
        j, tail = divmod(tail, v * v)
        a, b = divmod(tail, v)
        x, y = sorted((i * v + a, j * v + b))
        slot = x * (2 * n - x + 1) // 2 + y - x
        return ((j * o + i) * v + b) * v + a, slot, 1 if x == y else 2


def pair_projection_program() -> Program:
    """Overflow-safe mean for an admitted two-element rounding orbit.

    Identical values and singleton orbits bypass this expression in the storage
    consumer, preserving exact symmetric values including subnormal numbers.
    """
    scalar = TensorSpec((), role="input")
    first, second = (input_tensor(name, scalar) for name in ("first", "second"))
    return Program(
        {"mean": add(first, second, coefficients=(Fraction(1, 2), Fraction(1, 2)))}
    )


def cpp_coordinates() -> str:
    """Emit one host/device coordinate contract for conventional and DF owners.

    Integer arithmetic is admitted by the owner's checked full/packed extents;
    no coordinate table, square-root inverse or full conversion arena is needed.
    """
    projection = emit_scalar_cpp(
        pair_projection_program(),
        function_name="pair_projection",
        input_order=("first", "second"),
        output_order=("mean",),
        caller_owned_checks=True,
        ordered_native_sums=True,
        output_dependency_order=True,
    ).replace("inline bool", "GENERATIVEQC_PAIR_HD inline bool")
    lines = [
        "#ifndef GENERATIVEQC_RESTRICTED_PAIR_COORDINATES_DEFINED",
        "#define GENERATIVEQC_RESTRICTED_PAIR_COORDINATES_DEFINED",
        "#include <cstddef>",
        "#include <cmath>",
        "#if defined(__CUDACC__)",
        "#define GENERATIVEQC_PAIR_HD __host__ __device__",
        "#else",
        "#define GENERATIVEQC_PAIR_HD",
        "#endif",
        "namespace generativeqc::cc::generated {",
        projection,
        "// Storage admission only; this is not a physical residual tolerance.",
        "inline constexpr double pair_rounding_tolerance=0x1p-46; // 64 FP64 eps",
        "struct RestrictedPairCoordinates {",
        "  std::size_t o{},v{};",
        "  struct Orbit { std::size_t partner,slot; unsigned char weight; };",
        "  GENERATIVEQC_PAIR_HD Orbit operator()(std::size_t flat) const {",
        "    const auto b=flat%v; flat/=v; const auto a=flat%v; flat/=v;",
        "    const auto j=flat%o,i=flat/o,n=o*v;",
        "    auto x=i*v+a,y=j*v+b; if(x>y){const auto t=x;x=y;y=t;}",
        "    return {((j*o+i)*v+b)*v+a,x*(2*n-x+1)/2+y-x,",
        "            static_cast<unsigned char>(x==y ? 1 : 2)};",
        "  }",
        "  GENERATIVEQC_PAIR_HD double project(double first,double second) const {",
        "    if(first==second)return first; double mean;pair_projection(first,second,mean);return mean;",
        "  }",
        "};",
        "} // namespace generativeqc::cc::generated",
        "#undef GENERATIVEQC_PAIR_HD",
        "#endif",
        "",
    ]
    return "\n".join(lines)
