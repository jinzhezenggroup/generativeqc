"""Bounded native XC matrix schedules generated from the compact bilinear.

Qualified shared-memory contractions reuse each density/AO load across a tile.
The conservative production fallback remains the already-qualified 16x16 shape.
Potential assembly packs the scalar/spatial coefficients into the dead density
work panels, then evaluates a symmetric cross product. The tiled path also
folds the deterministic three-channel point-total reduction into the potential
launch. No library handle, provider workspace, global allocation, screening or
precision change is needed.
"""

from dataclasses import dataclass
from typing import Any

from generativeqc_compiler.integral.cuda import CudaEmitter

from .xc_bilinear import ao_pair_bilinear, symmetric_density_element
from .xc_density_lowering import emit_density_binding
from .xc_density_provider import emit_density_provider
from .xc_potential_lowering import emit_potential_portfolio


@dataclass(frozen=True, slots=True)
class XcMatrixSchedule:
    """One compiler-owned CUDA matrix tile with explicit resource admission.

    Alternative tiles are compile/tuning candidates only. Production keeps the
    qualified 16x16 schedule unless a caller explicitly selects another legal
    candidate and independently qualifies it.
    """

    tile: int

    def __post_init__(self) -> None:
        if type(self.tile) is not int or self.tile < 1 or self.tile & (self.tile - 1):
            raise ValueError("XC matrix tile must be a positive power of two")
        # All three totals channels need an x lane; larger tiles can exceed
        # CUDA block/shared-memory limits. Direct emitters share this boundary.
        if self.tile not in (8, 16, 32):
            raise ValueError("XC matrix tile must be a supported candidate (8, 16, 32)")

    @property
    def threads(self) -> int:
        return self.tile * self.tile

    @property
    def density_shared_bytes(self) -> int:
        return 2 * self.tile * (self.tile + 1) * 8

    @property
    def potential_shared_bytes(self) -> int:
        # Four FP64 panels, exact-scaling flags and two shared tile indices.
        return (4 * 8 + 1) * self.tile * (self.tile + 1) + 16

    @property
    def shared_bytes(self) -> int:
        return max(self.density_shared_bytes, self.potential_shared_bytes)

    @property
    def additional_workspace_bytes(self) -> int:
        """The tiled candidates reuse already-borrowed AO/work panels."""

        return 0

    def admitted(
        self,
        n: int,
        count: int,
        *,
        spins: int = 1,
        work_jets: int = 1,
        maximum_threads_per_block: int = 1024,
        maximum_shared_bytes: int = 48 * 1024,
        maximum_grid_dimension: int = 65535,
    ) -> bool:
        """Check launch/resource legality without making a profitability claim."""

        for label, value in (
            ("AO count", n),
            ("point count", count),
            ("spin count", spins),
            ("work jet count", work_jets),
            ("maximum threads", maximum_threads_per_block),
            ("maximum shared bytes", maximum_shared_bytes),
            ("maximum grid dimension", maximum_grid_dimension),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        tiles = (n + self.tile - 1) // self.tile
        point_tiles = (count + self.tile - 1) // self.tile
        tile_pairs = tiles * (tiles + 1) // 2
        return (
            n >= self.tile
            and count >= self.tile
            and self.threads <= maximum_threads_per_block
            and self.shared_bytes <= maximum_shared_bytes
            and tiles <= maximum_grid_dimension
            and point_tiles <= maximum_grid_dimension
            and tile_pairs <= maximum_grid_dimension
            and spins * work_jets <= maximum_grid_dimension
        )


XC_MATRIX_SCHEDULE_CANDIDATES = tuple(XcMatrixSchedule(tile) for tile in (8, 16, 32))
DEFAULT_XC_MATRIX_SCHEDULE = XcMatrixSchedule(16)


def qualified_xc_matrix_schedules(
    n: int,
    count: int,
    *,
    spins: int = 1,
    work_jets: int = 1,
    maximum_threads_per_block: int = 1024,
    maximum_shared_bytes: int = 48 * 1024,
    maximum_grid_dimension: int = 65535,
) -> tuple[XcMatrixSchedule, ...]:
    """Return resource-legal candidates; ordering is not a performance ranking."""

    return tuple(
        schedule
        for schedule in XC_MATRIX_SCHEDULE_CANDIDATES
        if schedule.admitted(
            n,
            count,
            spins=spins,
            work_jets=work_jets,
            maximum_threads_per_block=maximum_threads_per_block,
            maximum_shared_bytes=maximum_shared_bytes,
            maximum_grid_dimension=maximum_grid_dimension,
        )
    )


def select_xc_matrix_schedule(
    n: int,
    count: int,
    *,
    preferred_tile: int | None = None,
    spins: int = 1,
    work_jets: int = 1,
    maximum_threads_per_block: int = 1024,
    maximum_shared_bytes: int = 48 * 1024,
    maximum_grid_dimension: int = 65535,
) -> XcMatrixSchedule | None:
    """Resolve an explicit qualified candidate or preserve the 16x16 fallback.

    With no profile/preference, this deliberately does *not* promote a new tile
    from a static heuristic. If the qualified 16x16 schedule is unavailable,
    callers retain the scalar fallback. Autotuning may request 8x8 or 32x32
    explicitly and benchmark it under the same scientific contract.
    """

    candidates = qualified_xc_matrix_schedules(
        n,
        count,
        spins=spins,
        work_jets=work_jets,
        maximum_threads_per_block=maximum_threads_per_block,
        maximum_shared_bytes=maximum_shared_bytes,
        maximum_grid_dimension=maximum_grid_dimension,
    )
    if preferred_tile is not None:
        if type(preferred_tile) is not int or preferred_tile < 1:
            raise ValueError("preferred XC matrix tile must be a positive integer")
        for schedule in candidates:
            if schedule.tile == preferred_tile:
                return schedule
        raise ValueError(f"XC matrix tile {preferred_tile} is not qualified")
    return (
        DEFAULT_XC_MATRIX_SCHEDULE if DEFAULT_XC_MATRIX_SCHEDULE in candidates else None
    )


def compact_panel_program(family: str) -> tuple:
    """Factor the symmetric bilinear into A*W^T + W*A^T.

    The value panel absorbs both gradient legs. The scalar and diagonal kinetic
    terms get a one-half because symmetric assembly visits each twice. Keep
    that preparation order even for the one-sided consumer: moving the factor
    of two into this graph changes its overflow and underflow domain.
    """
    graph, x, y, _coefficients, bilinear = ao_pair_bilinear(family)

    def entry(row: int, column: int) -> Any:
        return graph.differentiate(graph.differentiate(bilinear, x[row]), y[column])

    value = entry(0, 0) * x[0] / 2
    for j in range(1, len(x)):
        value += entry(j, 0) * x[j]
    roots = [value]
    if family == "mgga":
        roots.extend(entry(j, j) * x[j] / 2 for j in range(1, 4))
    return graph, tuple(roots)


def _emit_density_factor(indexed: bool = False) -> str:
    """Emit the same factor over a dense or validated immutable indexed view."""
    graph, root = symmetric_density_element()
    emitter = CudaEmitter(
        graph, {"left": "density[source]", "right": "density[transpose]"}
    )
    emitter.emit((root,))
    prologue = r"""
__global__ void materialize_density_factor(const double* density, I n, I batches,
                                          double* output, int* error) {
  for (I i=I(blockIdx.x)*blockDim.x+threadIdx.x; i<batches*n*n;
       i+=I(blockDim.x)*gridDim.x) {
    const I source=i;
    const I transpose=(i/(n*n)*n+i%n)*n+i/n%n;
"""
    if indexed:
        prologue = r"""
__global__ void gather_density_factor(const double* density, I global_n, I n, I batches,
    const std::size_t* ids, double* output, int* error) {
  for (I i=I(blockIdx.x)*blockDim.x+threadIdx.x; i<batches*n*n;
       i+=I(blockDim.x)*gridDim.x) {
    const I spin=i/(n*n);
    // A complete sorted CSR span is published as the null identity map.
    const I row=ids ? ids[i/n%n] : i/n%n, col=ids ? ids[i%n] : i%n;
    const I source=(spin*global_n+row)*global_n+col;
    const I transpose=(spin*global_n+col)*global_n+row;
"""
    return (
        prologue
        + "\n".join(emitter.lines)
        + f"\n    output[i]=finite({emitter.reference(root)},error,1);\n"
        + "  }\n}\n"
    )


def _emit_panels(*, diagonal_one_sided: bool = False) -> str:
    lines = [
        r"""
__device__ void compact_potential_panels_body(const double* ao, const double* coefficients,
    const double* weights, I n, I count, I spins, I terms, I work_jets, double* work, int* error) {
  const I panel = count*n;
  for (I i = I(blockIdx.x)*blockDim.x+threadIdx.x; i < spins*panel;
       i += I(blockDim.x)*gridDim.x) {
    const I spin = i/panel, index = i%panel, point = index/n;
"""
    ]
    for family, terms in (("lda", 1), ("gga", 4), ("mgga", 5)):
        graph, roots = compact_panel_program(family)
        bindings = {f"x{j}": f"ao[{j}*panel+index]" for j in range(4)}
        bindings.update(
            {f"c{j}": f"coefficients[(spin*terms+{j})*count+point]" for j in range(5)}
        )
        emitter = CudaEmitter(graph, bindings)
        emitter.emit(roots)
        lines.append(f"    if (terms == {terms}) {{")
        lines.extend("    " + line for line in emitter.lines)
        lines.extend(
            f"      work[(spin*work_jets+{j})*panel+index] = finite(weights[point]*({emitter.reference(root)}), error, 3);"
            for j, root in enumerate(roots)
        )
        lines.append("    }")
    lines.append("  }\n}\n")
    lines.append(r"""
__global__ void compact_potential_panels(const double* ao, const double* coefficients,
    const double* weights, I n, I count, I spins, I terms, I work_jets, double* work, int* error) {
  compact_potential_panels_body(ao, coefficients, weights, n, count, spins, terms, work_jets,
                                work, error);
}
""")
    source = "\n".join(lines)
    if diagonal_one_sided:
        source = source.replace(
            "compact_potential_panels", "compact_diagonal_potential_panels"
        )
    return source


_TILED_TEMPLATE = r"""
// Padding avoids bank conflicts when the AO lane reads a transposed panel.
// Partial tiles still participate in both barriers and load exact zero padding.
template <bool Mixed, bool ActiveRowsOnly = false, class Arithmetic = double>
__device__ void tiled_density_product_body(const double* density, const double* ao, I n, I count,
    I work_jets, double* work, int* error, const size_t* ao_ids, I full_n, I spin_jet,
    const double* occupied_frame = nullptr, I occupied_count = 0, I occupied_rank = 0) {
  __shared__ Arithmetic d[@TILE@][@PAD@], a[@TILE@][@PAD@];
  const I x = threadIdx.x, y = threadIdx.y;
  const I mu = I(blockIdx.x)*@TILE@+x, point = I(blockIdx.y)*@TILE@+y;
  const I spin = spin_jet/work_jets, jet = spin_jet%work_jets;
  const I panel = count*n;
  const I output_n = occupied_frame ? occupied_rank : n;
  const double* source = ao+jet*panel;
  const double* matrix = (occupied_frame ? occupied_frame : density)+spin*full_n*full_n;
  // All AO rows for one point/jet share the same binary scale. Restoring it
  // after the FP32 dot avoids selective tail underflow and needs no FP64 dot,
  // additional AO panel, or change to the strict specialization.
  int ao_exponent = 0;
  if constexpr (std::is_same_v<Arithmetic, float>) {
    double maximum = 0.0;
    if (point < count)
      for (I col = x; col < n; col += @TILE@)
        maximum = fmax(maximum, fabs(source[point*n+col]));
    for (I offset = @TILE@/2; offset; offset /= 2)
      maximum = fmax(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset, @TILE@));
    if (maximum > 0.0) frexp(maximum, &ao_exponent);
  }
  Arithmetic value = 0.0;
  for (I begin = 0; begin < n; begin += @TILE@) {
    const I row = I(blockIdx.x)*@TILE@+y, col = begin+x;
    // Gather directly into the existing shared tile: no full/local D copy and
    // no extra matrix allocation. Tail lanes must not dereference the AO map.
    const I global_row = row < n ? (ao_ids ? ao_ids[row] : row) : 0;
    const I global_col = col < n ? (ao_ids ? ao_ids[col] : col) : 0;
    if constexpr (Mixed) {
      if (row < n && col < n) {
        const float left = __double2float_rn(matrix[global_row*full_n+global_col]);
        const float right = __double2float_rn(matrix[global_col*full_n+global_row]);
        d[y][x] = static_cast<double>(
            __fadd_rn(__fmul_rn(0.5f, left), __fmul_rn(0.5f, right)));
      } else {
        d[y][x] = 0.0;
      }
      a[y][x] = point < count && col < n
                    ? static_cast<double>(__double2float_rn(source[point*n+col]))
                    : 0.0;
    } else {
      if (occupied_frame)
        d[y][x] = row < occupied_count && col < n
            ? matrix[row*full_n+global_col] : 0.0;
      else
        d[y][x] = row < n && col < n ? 0.5*matrix[global_row*full_n+global_col]+0.5*matrix[global_col*full_n+global_row] : 0.0;
      if constexpr (std::is_same_v<Arithmetic, float>)
        a[y][x] = point < count && col < n
                      ? scalbn(source[point*n+col], -ao_exponent) : 0.0;
      else
        a[y][x] = point < count && col < n ? source[point*n+col] : 0.0;
    }
    __syncthreads();
    // Only discarded outputs may skip arithmetic. Every lane still loads the
    // shared panels and reaches both barriers; valid sums keep their order.
    if (!ActiveRowsOnly || (mu < output_n && point < count))
      for (I k = 0; k < @TILE@; ++k) {
      if constexpr (Mixed)
        value = __dadd_rn(
            value, static_cast<double>(__fmul_rn(__double2float_rn(d[x][k]),
                                                  __double2float_rn(a[y][k]))));
      else if constexpr (std::is_same_v<Arithmetic, double>)
        // Explicit FP64 fusion keeps the k order and single accumulator, but
        // avoids a separately rounded multiply under global --fmad=false.
        value = fma(d[x][k], a[y][k], value);
      else
        value += d[x][k]*a[y][k];
    }
    __syncthreads();
  }
  if (mu < output_n && point < count) {
    // Normalize only the FP32 operand; restore its exact power-of-two scale
    // in FP64 storage so tiny physical AO tails do not lose their sign/domain.
    if constexpr (std::is_same_v<Arithmetic, float>)
      work[(spin*work_jets+jet)*count*output_n+point*output_n+mu] = finite(scalbn(static_cast<double>(value), ao_exponent), error, 1);
    else
      work[(spin*work_jets+jet)*count*output_n+point*output_n+mu] = finite(value, error, 1);
  }
}

template <bool Mixed, class Arithmetic = double>
__global__ void tiled_density_product(const double* density, const double* ao, I n, I count,
    I work_jets, double* work, int* error, const size_t* ao_ids, I full_n) {
  tiled_density_product_body<Mixed, false, Arithmetic>(density, ao, n, count, work_jets, work, error, ao_ids,
                                     full_n, blockIdx.z);
}

// One triangle is authoritative, including on diagonal and partial blocks.
// A compact linear block domain enumerates only tile_mu <= tile_nu instead of
// launching the unused lower half of a square grid. One lane decodes the tile
// pair; all @THREADS@ lanes then execute the unchanged symmetric contraction. The
// first block of spin zero also performs the historical serial-per-channel
// point-total reduction after its matrix work, preserving the exact arithmetic
// order while avoiding a separate kernel launch for every point tile. The first
// point tile owns output initialization, so admitted tiled execution needs no
// matrix-sized or totals memset before every XC evaluation.
template <bool ActivePairsOnly = false, class Arithmetic = double, bool DiagonalJets = false>
__device__ void tiled_potential_body(const double* ao, const double* work, I n, I count,
                                I work_jets, const double* point_totals, double* potential,
    double* totals, bool accumulate, int* error, const size_t* ao_ids, I full_n, I spin,
    const double* diagonal_weights = nullptr) {
  __shared__ I tile_mu, tile_nu;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    const I pair = blockIdx.x;
    I low = 0, high = (n+@TILE_MINUS_ONE@)/@TILE@;
    while (low+1 < high) {
      const I mid = (low+high)/2;
      if (mid*(mid+1)/2 <= pair)
        low = mid;
      else
        high = mid;
    }
    tile_nu = low;
    tile_mu = pair-low*(low+1)/2;
  }
  __syncthreads();
  __shared__ Arithmetic am[@TILE@][@PAD@], an[@TILE@][@PAD@], wm[@TILE@][@PAD@], wn[@TILE@][@PAD@];
  __shared__ unsigned char scaled_wn[@TILE@][@PAD@];
  const I x = threadIdx.x, y = threadIdx.y;
  const I mu = tile_mu*@TILE@+x, nu = tile_nu*@TILE@+y;
  const I nu_load = tile_nu*@TILE@+x, panel = count*n;
  Arithmetic value = 0.0;
  for (I jet = 0; jet < work_jets; ++jet) {
    // Additional panels are diagonal bilinears derived by compact_panel_program.
    // Preserve canonical half-panel preparation. Only bounded operand tiles
    // reuse a doubled right panel; unsafe tiles keep both original products.
    // The upper bound prevents scaling/product overflow and amplification of
    // preparation underflow. It is independent of the method and AO count.
    const bool cross = !DiagonalJets || jet == 0;
    const double* a = ao+jet*panel;
    const double* w = work+(spin*work_jets+jet)*panel;
    for (I begin = 0; begin < count; begin += @TILE@) {
      const I p = begin+y;
      am[y][x] = p < count && mu < n ? a[p*n+mu] : 0.0;
      an[y][x] = p < count && nu_load < n ? a[p*n+nu_load] : 0.0;
      wm[y][x] = p < count && mu < n ? w[p*n+mu] : 0.0;
      const Arithmetic right = p < count && nu_load < n ? w[p*n+nu_load] : 0.0;
      bool reuse_diagonal = false;
      if (DiagonalJets && !cross) {
        // Keep the qualified FP64 range unchanged. FP32 needs its own bound:
        // a finite right panel can overflow when doubled before multiplication.
        constexpr double diagonal_limit = std::is_same_v<Arithmetic, float> ? 0x1p30 : 0x1p128;
        const bool scaled = fabs(right) <= diagonal_limit;
        scaled_wn[y][x] = scaled;
        wn[y][x] = scaled ? right+right : right;
        const bool unsafe_weight = !diagonal_weights ||
            (p < count && fabs(diagonal_weights[p]) > diagonal_limit);
        const bool unsafe = unsafe_weight || !scaled || fabs(am[y][x]) > diagonal_limit ||
            fabs(an[y][x]) > diagonal_limit || fabs(wm[y][x]) > diagonal_limit;
        reuse_diagonal = !__syncthreads_or(unsafe);
      } else {
        wn[y][x] = right;
        __syncthreads();
      }
      if (!ActivePairsOnly || (mu < n && nu < n && mu <= nu)) {
        if (cross)
          for (I k = 0; k < @TILE@; ++k) {
            if constexpr (std::is_same_v<Arithmetic, double>) {
              const double prior = value;
              value = fma(wm[k][x], an[k][y], fma(am[k][x], wn[k][y], value));
              // Fusion must not lose a finite grouped cross term when its
              // first addition overflows before the opposite leg cancels it.
              if (!(fabs(value) <= 0x1.fffffffffffffp1023))
                value = prior + (am[k][x]*wn[k][y]+wm[k][x]*an[k][y]);
            } else
              value += am[k][x]*wn[k][y]+wm[k][x]*an[k][y];
          }
        else if (reuse_diagonal)
          for (I k = 0; k < @TILE@; ++k) {
            if constexpr (std::is_same_v<Arithmetic, double>)
              value = fma(am[k][x], wn[k][y], value);
            else
              value += am[k][x]*wn[k][y];
          }
        else
          for (I k = 0; k < @TILE@; ++k) {
            const Arithmetic original_right = scaled_wn[k][y] ? wn[k][y]*0.5 : wn[k][y];
            value += am[k][x]*original_right+wm[k][x]*an[k][y];
          }
      }
      __syncthreads();
    }
  }
  if (mu < n && nu < n && mu <= nu) {
    const I row = ao_ids ? ao_ids[mu] : mu, col = ao_ids ? ao_ids[nu] : nu;
    const I index = (spin*full_n+row)*full_n+col;
    const double prior = accumulate ? potential[index] : 0.0;
    const double combined = finite(prior+static_cast<double>(value), error, 3);
    potential[index] = combined;
    potential[(spin*full_n+col)*full_n+row] = combined;
  }
  if (point_totals && blockIdx.x == 0 && spin == 0 && threadIdx.y == 0 && threadIdx.x < 3) {
    const I channel = threadIdx.x;
    Arithmetic sum = 0.0;
    for (I p = 0; p < count; ++p) sum += static_cast<Arithmetic>(point_totals[channel*count+p]);
    totals[channel] = finite((accumulate ? totals[channel] : 0.0)+static_cast<double>(sum),error,3);
  }
}

// The compiler owns schedule admission; native only supplies borrowed buffers.
// Tiny shapes and dimensions outside the two-dimensional launch domain retain
// the bounded grid-stride scalar schedule. Both cover the same scientific work.
template <class Arithmetic = double>
__global__ void tiled_potential(const double* ao, const double* work, I n, I count,
    I work_jets, const double* point_totals, double* potential, double* totals,
    bool accumulate, int* error, const size_t* ao_ids, I full_n) {
  tiled_potential_body<false, Arithmetic>(ao, work, n, count, work_jets, point_totals, potential, totals,
                        accumulate, error, ao_ids, full_n, blockIdx.z);
}

template <class Arithmetic = double>
__global__ void tiled_diagonal_potential(const double* ao, const double* work, I n, I count,
    I work_jets, const double* point_totals, double* potential, double* totals,
    bool accumulate, int* error, const size_t* ao_ids, I full_n, const double* weights) {
  tiled_potential_body<false, Arithmetic, true>(ao, work, n, count, work_jets, point_totals,
      potential, totals, accumulate, error, ao_ids, full_n, blockIdx.z, weights);
}

inline bool tiled_xc_admitted(I n, I count, I spins, I work_jets) {
  if (n < @TILE@ || count < @TILE@ || spins < 1 || work_jets < 1 ||
      work_jets > 65535 || spins > 65535/work_jets) return false;
  // Avoid overflow before rejecting an out-of-domain launch shape.
  const I tiles = 1+(n-1)/@TILE@;
  const I point_tiles = 1+(count-1)/@TILE@;
  if (tiles > 65535 || point_tiles > 65535) return false;
  const I tile_pairs = tiles*(tiles+1)/2;
  return tile_pairs <= 65535;
}
/** FP32 also uses the existing masked tiled body for small/tail domains.
 * Rounding dimensions up only for admission gives the same one-tile launch
 * bounds without changing selected AOs, points or strict-FP64 scheduling. */
inline bool fp32_tiled_xc_admitted(I ao_count, I count, I spins, I work_jets) {
  return ao_count > 0 && count > 0 &&
      tiled_xc_admitted(ao_count < @TILE@ ? @TILE@ : ao_count, count < @TILE@ ? @TILE@ : count,
                       spins, work_jets);
}
template <bool Mixed, bool Tiled, class Arithmetic = double>
void launch_density_product(cudaStream_t stream, const double* density,
    const double* ao, I n, I count, I spins, I work_jets,
    double* work, int* error, const size_t* ao_ids, I full_n) {
  if (!n) return;
  if constexpr (Tiled) {
    const dim3 grid((n+@TILE_MINUS_ONE@)/@TILE@, (count+@TILE_MINUS_ONE@)/@TILE@,
                    spins*work_jets), block(@TILE@,@TILE@);
    tiled_density_product<Mixed, Arithmetic><<<grid, block, 0, stream>>>(
        density, ao, n, count, work_jets, work, error, ao_ids, full_n);
  } else {
    density_product<Mixed><<<generativeqc_tensor::blocks(spins*work_jets*count*n,128),128,0,stream>>>(
        density,ao,n,count,spins,work_jets,work,error,ao_ids,full_n);
  }
}
template <class Arithmetic = double>
inline void generated_potential(cudaStream_t stream, const double* ao,
    const double* coefficients, const double* weights, I n, I count, I spins,
    I terms, I work_jets, double* work, const double* point_totals,
    double* potential, double* totals, bool accumulate, int* error,
    const size_t* ao_ids = nullptr, I full_n = 0) {
  if (!full_n) full_n = n;
  // Sparse accumulation requires initialization of absent global entries by
  // the owner, once per complete evaluation. Empty tiles still publish totals.
  if (ao_ids && !accumulate)
    throw std::invalid_argument("mapped XC potential requires initialized global output");
  if (!n) {
    if (point_totals) accumulate_totals<<<1,32,0,stream>>>(point_totals,count,totals,error);
    return;
  }
  if (tiled_xc_admitted(n, count, spins, work_jets) ||
      (std::is_same_v<Arithmetic, float> && fp32_tiled_xc_admitted(n, count, spins, work_jets))) {
    if (work_jets == 4) {
      compact_diagonal_potential_panels<<<generativeqc_tensor::blocks(spins*count*n,128),128,0,stream>>>(
          ao,coefficients,weights,n,count,spins,terms,work_jets,work,error);
      generativeqc_tensor::cuda_check(cudaGetLastError());
      const I tiles = (n+@TILE_MINUS_ONE@)/@TILE@, tile_pairs = tiles*(tiles+1)/2;
      tiled_diagonal_potential<Arithmetic><<<dim3(tile_pairs,1,spins),dim3(@TILE@,@TILE@),0,stream>>>(
          ao,work,n,count,work_jets,point_totals,potential,totals,accumulate,error,ao_ids,full_n,weights);
      return;
    }
    compact_potential_panels<<<generativeqc_tensor::blocks(spins*count*n,128),128,0,stream>>>(
        ao,coefficients,weights,n,count,spins,terms,work_jets,work,error);
    generativeqc_tensor::cuda_check(cudaGetLastError());
    const I tiles = (n+@TILE_MINUS_ONE@)/@TILE@, tile_pairs = tiles*(tiles+1)/2;
    tiled_potential<Arithmetic><<<dim3(tile_pairs,1,spins),dim3(@TILE@,@TILE@),0,stream>>>(
        ao,work,n,count,work_jets,point_totals,potential,totals,accumulate,error,ao_ids,full_n);
  } else {
    if (!accumulate) {
      generativeqc_tensor::cuda_check(cudaMemsetAsync(potential,0,spins*n*n*sizeof(double),stream));
      generativeqc_tensor::cuda_check(cudaMemsetAsync(totals,0,3*sizeof(double),stream));
    }
    assemble_potential<<<generativeqc_tensor::blocks(spins*n*n,128),128,0,stream>>>(
        ao,coefficients,weights,n,count,spins,terms,potential,error,ao_ids,full_n);
    generativeqc_tensor::cuda_check(cudaGetLastError());
    if (point_totals) accumulate_totals<<<1,32,0,stream>>>(point_totals,count,totals,error);
  }
}

// This adapter only supplies semantic operands and the compiler-owned packing
// and epilogue. Provider selection/handles remain in the shared tensor owner.
struct PotentialCall {
  cudaStream_t stream;
  const double *ao, *coefficients, *weights;
  I n, count, spins, terms, work_jets;
  double* work;
  const double* point_totals;
  double *potential, *totals;
  bool accumulate;
  int* error;
  const size_t* ids;
  I full_n;
};
inline void scheduled_potential(cudaStream_t stream, const double* ao,
    const double* coefficients, const double* weights, I n, I count, I spins,
    I terms, I work_jets, double* work, const double* point_totals,
    double* potential, double* totals, bool accumulate, int* error,
    const size_t* ao_ids = nullptr, I full_n = 0,
    const tensor::PreparedSymmetricProduct* binding = nullptr, bool fp32 = false) {
  if (fp32) {
    generated_potential<float>(stream,ao,coefficients,weights,n,count,spins,terms,
        work_jets,work,point_totals,potential,totals,accumulate,error,ao_ids,full_n);
    return;
  }
  PotentialCall state{stream,ao,coefficients,weights,n,count,spins,terms,work_jets,
      work,point_totals,potential,totals,accumulate,error,ao_ids,full_n};
  const auto generated=[](void* opaque) {
    const auto& s=*static_cast<PotentialCall*>(opaque);
    generated_potential(s.stream,s.ao,s.coefficients,s.weights,s.n,s.count,s.spins,s.terms,
        s.work_jets,s.work,s.point_totals,s.potential,s.totals,s.accumulate,s.error,s.ids,s.full_n);
  };
  if (!binding) {generated(&state);return;}
  const auto materialize=[](void* opaque) {
    const auto& s=*static_cast<PotentialCall*>(opaque);
    compact_potential_panels<<<generativeqc_tensor::blocks(s.spins*s.count*s.n,128),128,0,s.stream>>>(
        s.ao,s.coefficients,s.weights,s.n,s.count,s.spins,s.terms,s.work_jets,s.work,s.error);
    generativeqc_tensor::cuda_check(cudaGetLastError());
  };
  const auto epilogue=[](void* opaque) {
    const auto& s=*static_cast<PotentialCall*>(opaque);
    if (!s.point_totals) return;
    if (!s.accumulate)
      generativeqc_tensor::cuda_check(cudaMemsetAsync(s.totals,0,3*sizeof(double),s.stream));
    accumulate_totals<<<1,32,0,s.stream>>>(s.point_totals,s.count,s.totals,s.error);
    generativeqc_tensor::cuda_check(cudaGetLastError());
  };
  binding->execute(stream,{std::size_t(n),std::size_t(work_jets*count),std::size_t(spins),
      ao,work,potential,error,accumulate,ao_ids!=nullptr || (full_n && full_n!=n),
      &state,generated,materialize,epilogue,ao_ids,std::size_t(full_n)});
}
"""


def _emit_tiled(schedule: XcMatrixSchedule) -> str:
    if not isinstance(schedule, XcMatrixSchedule):
        raise TypeError("XC matrix emission requires XcMatrixSchedule")
    replacements = {
        "@TILE@": str(schedule.tile),
        "@PAD@": str(schedule.tile + 1),
        "@TILE_MINUS_ONE@": str(schedule.tile - 1),
        "@THREADS@": str(schedule.threads),
    }
    source = _TILED_TEMPLATE
    for token, value in replacements.items():
        source = source.replace(token, value)
    return source


def emit_native_xc_matrix_schedule(
    schedule: XcMatrixSchedule = DEFAULT_XC_MATRIX_SCHEDULE,
    *,
    density_source: str = "",
) -> str:
    """Emit compact graph lowering for one explicitly qualified tile candidate."""
    panels = _emit_panels() + _emit_panels(diagonal_one_sided=True)
    return (
        emit_potential_portfolio(panels + _emit_tiled(schedule))
        + '\n#include "tensor/cuda_panel_product.cuh"\n#include <type_traits>\n'
        + "namespace generativeqc::dft::cuda_xc_detail {\nnamespace {\n"
        + panels
        + _emit_tiled(schedule)
        + _emit_occupied_features(schedule)
        + _emit_density_factor()
        + _emit_density_factor(indexed=True)
        + "\n} // namespace\n"
        + emit_density_binding(
            schedule.tile,
            density_source
            + _emit_tiled(schedule)
            + _emit_density_factor()
            + _emit_density_factor(indexed=True),
        )
        + emit_density_provider()
        + "\n} // namespace generativeqc::dft::cuda_xc_detail\n"
    )


def _emit_occupied_features(schedule: XcMatrixSchedule) -> str:
    """Project an exact density witness into bounded dead density panels.

    All four orbital jets are required. Potential construction retains the
    original four AO jets; the native owner checks density and FP64 provenance.
    """
    return _OCCUPIED_TEMPLATE.replace("@TILE@", str(schedule.tile)).replace(
        "@PAD@", str(schedule.tile + 1)
    )


_OCCUPIED_TEMPLATE = r"""
__device__ void tiled_occupied_product_body(CudaXcOccupiedDensity occupied,
    const double* ao, I active, I count, double* work, int* error,
    const size_t* ids, I spin_jet) {
  // Both scientific sources use the same shared panels and AO traversal.
  // A compact batch does not pay for a second statically allocated panel.
  tiled_density_product_body<false, true>(occupied.coefficients, ao, active, count, 4,
      work, error, ids, occupied.nao, spin_jet, occupied.coefficients,
      occupied.occupations[spin_jet / 4], occupied.rank());
}

__global__ void occupied_density_products(CudaXcOccupiedDensity occupied,
    const double* ao, I active, I count, double* work, int* error, const size_t* ids) {
  tiled_occupied_product_body(occupied, ao, active, count, work, error, ids, blockIdx.z);
}

template<bool Cooperative = true>
__device__ void occupied_density_features_body(const double* work, I rank, I count,
    I spins, I terms, double* features, int* error) {
  constexpr I width = Cooperative ? 32 : 1;
  const I lane = threadIdx.x % width, stride = count * rank;
  for (I item = (I(blockIdx.x) * blockDim.x + threadIdx.x) / width;
       item < spins * count; item += I(blockDim.x) * gridDim.x / width) {
    const I spin = item / count, point = item % count;
    const auto* source = work + spin * 4 * stride;
    double accum[5]{};
    for (I orbital = lane; orbital < rank; orbital += width) {
      const I index = point * rank + orbital;
      double panel[4]{source[index], source[stride + index],
                      source[2 * stride + index], source[3 * stride + index]};
      double derivatives[3]{panel[1], panel[2], panel[3]};
      generativeqc_grid_policy::add_features(panel[0], derivatives, panel, accum, 15U);
    }
    for (I term = 0; term < terms; ++term) {
      if constexpr (Cooperative)
        for (int offset = 16; offset > 0; offset >>= 1)
          accum[term] += __shfl_down_sync(0xffffffffu, accum[term], offset);
      if (lane == 0)
        features[(spin * terms + term) * count + point] =
            finite((spins == 1 ? 2.0 : 1.0) * accum[term], error, 1);
    }
  }
}

__global__ void occupied_density_features(const double* work, I rank, I count,
    I spins, I terms, double* features, int* error) {
  occupied_density_features_body(work, rank, count, spins, terms, features, error);
}

inline void launch_occupied_density(cudaStream_t stream, CudaXcOccupiedDensity occupied,
    const double* ao, I active, I count, double* work, double* features,
    I terms, int* error, const size_t* ids) {
  occupied_density_products<<<dim3((occupied.rank() + @TILE@ - 1) / @TILE@,
      (count + @TILE@ - 1) / @TILE@, occupied.spins * 4), dim3(@TILE@, @TILE@), 0, stream>>>(
      occupied, ao, active, count, work, error, ids);
  generativeqc_tensor::cuda_check(cudaGetLastError());
  occupied_density_features<<<generativeqc_tensor::blocks(occupied.spins * count * 32, 128),
      128, 0, stream>>>(work, occupied.rank(), count, occupied.spins, terms, features, error);
  generativeqc_tensor::cuda_check(cudaGetLastError());
}
"""
