// Included by cuda_grid_geometry.cu after the shared partition evaluator.
namespace becke_fold {
constexpr int tile = 96;
struct Vector {
  double xyz[3];
};
struct Radius {
  Vector unit;
  double distance, inverse;
};
struct Edge {
  double factor, slope, curvature;
};
struct Product {
  double value, nonzero, seed;
  int zero, first, second, exponent;
};
__device__ Radius radius(Vector first, Vector second) {
  Vector gap{};
  double square{};
  for (int axis = 0; axis < 3; ++axis) {
    gap.xyz[axis] = first.xyz[axis] - second.xyz[axis];
    square += gap.xyz[axis] * gap.xyz[axis];
  }
  const double distance = sqrt(square);
  const double inverse = 1 / distance;
  for (double& coordinate : gap.xyz) coordinate *= inverse;
  return {gap, distance, inverse};
}
__device__ int pair_index(int first, int second) {
  const auto high = max(first, second), low = min(first, second);
  return high * (high - 1) / 2 + low;
}
__global__ void tile_radii(int atoms, const Vector* centers, const Vector* points, Radius* radii,
                           int* status) {
  const int atom = threadIdx.x, point = blockIdx.x;
  if (atom < atoms) {
    const auto radial = radius(points[point], centers[atom]);
    if (!(radial.distance >= 0x1p-32 && radial.distance <= 0x1p32)) atomicOr(status, 1);
    radii[point * atoms + atom] = radial;
  }
}

__global__ void tile_edges(int atoms, const Vector* centers, const Radius* radii, Edge* edges,
                           CudaGridGeometryOptions options, const double* atomic_radii,
                           int* status) {
  const int point = blockIdx.x, first = threadIdx.x;
  if (first >= atoms) return;
  auto* out = edges + point * (atoms * (atoms - 1) / 2);
  radii += point * atoms;
  for (int second = 0; second < first; ++second) {
    const auto separation = radius(centers[first], centers[second]);
    const bool adjusted = atomic_radii != nullptr;
    const double ratio = adjusted ? sqrt(atomic_radii[second] / atomic_radii[first]) : 1;
    const double adjustment = fmax(-0.5, fmin(0.5, 0.25 * (ratio - 1 / ratio)));
    const double coordinate = (radii[first].distance - radii[second].distance) * separation.inverse;
    const auto factor =
        partition_switch({coordinate, 1, 1, 0}, adjusted, adjustment, options.partition_iterations);
    if (!(separation.distance > options.coincident_tolerance && separation.distance >= 0x1p-32 &&
          separation.distance <= 0x1p32 && finite(factor) && maximum(factor) <= 0x1p64 &&
          factor.value >= 0 && factor.value <= 1))
      atomicOr(status, 2);
    out[pair_index(first, second)] = {factor.value, factor.left, factor.mixed};
  }
}

__device__ Edge ordered(const Edge* edges, int atom, int other) {
  const auto edge = edges[pair_index(atom, other)];
  return atom > other ? edge : Edge{1 - edge.factor, -edge.slope, -edge.curvature};
}

__device__ Vector coordinate_gradient(const Radius* radii, Radius separation, int first, int second,
                                      int center) {
  Vector out{};
  const double coordinate = (radii[first].distance - radii[second].distance) * separation.inverse;
  const double sign = center == first ? -1 : 1;
  for (int axis = 0; axis < 3; ++axis)
    out.xyz[axis] = sign * separation.inverse *
                    (radii[center].unit.xyz[axis] + coordinate * separation.unit.xyz[axis]);
  return out;
}

__device__ void place_pair(double* target, int atom, int other, Vector self, Vector neighbor,
                           double scale) {
  for (int axis = 0; axis < 3; ++axis) {
    target[3 * atom + axis] += scale * self.xyz[axis];
    target[3 * other + axis] += scale * neighbor.xyz[axis];
  }
}

__global__ void tile_rank(int atoms, int owner, const Vector* centers, const Radius* radii,
                          const Edge* edges, const double* measure, const double* weights,
                          const double* energy, Product* products, double* left, double* right,
                          double* gradient, std::size_t gradient_stride,
                          CudaGridGeometryOptions options, int* status) {
  const int atom = threadIdx.x, point = blockIdx.x;
  const int coordinates = 3 * atoms;
  const int pitch = coordinates * (2 * atoms + 2);
  __shared__ double pvalues[96], denominator, fraction;
  radii += point * atoms;
  edges += point * (atoms * (atoms - 1) / 2);
  products += point * atoms;
  double* left_panel = left + point * pitch;
  double* right_panel = right + point * pitch;
  Product product{0, 1, 0, 0, -1, -1, 0};
  if (atom < atoms) {
    for (int other = 0; other < atoms; ++other) {
      if (atom == other) continue;
      const auto edge = ordered(edges, atom, other);
      if (edge.factor != 0) {
        if (!(edge.factor >= 0x1p-100)) atomicOr(status, 4);
        product.nonzero *= edge.factor;
        // Rebase the nonzero-factor product before a subsequent multiply
        // could make it subnormal. The exponent is not differentiated.
        if (product.nonzero < 0x1p-448) {
          product.nonzero *= 0x1p512;
          product.exponent -= 512;
        }
      } else {
        if (product.zero++ == 0)
          product.first = other;
        else
          product.second = other;
        if (edge.slope == 0 && edge.curvature == 0) {
          product.zero = 3;
          break;
        }
      }
    }
    if (!product.zero) product.value = ldexp(product.nonzero, product.exponent);
    pvalues[atom] = product.value;
  }
  __syncthreads();
  if (!atom) {
    double value{};
    for (int center = 0; center < atoms; ++center) value += pvalues[center];
    denominator = value;
    fraction = pvalues[owner] / value;
    if (!(value >= 0x1p-48 && fabs(measure[point]) <= 0x1p32 && fabs(energy[point]) <= 0x1p32 &&
          matches_partition({fraction, 0, 0, 0}, weights[point], measure[point], options)))
      atomicOr(status, 8);
  }
  __syncthreads();
  if (atom < atoms) {
    auto* first_rank = right_panel + 2 * atom * coordinates;
    auto* second_rank = first_rank + coordinates;
    for (int coordinate = 0; coordinate < coordinates; ++coordinate)
      first_rank[coordinate] = second_rank[coordinate] = 0;
    const double normalized_mantissa = product.nonzero / denominator;
    product.seed = ldexp(
        normalized_mantissa * (double(atom == owner) - fraction) * (energy[point] * measure[point]),
        product.exponent);
    product.nonzero = ldexp(normalized_mantissa, product.exponent);
    if (product.zero < 3) {
      for (int other = 0; other < atoms; ++other) {
        if (other == atom) continue;
        const auto edge = ordered(edges, atom, other);
        if (product.zero == 2 && other != product.first && other != product.second) continue;
        const int first = max(atom, other), second = min(atom, other);
        const auto separation = radius(centers[first], centers[second]);
        const auto self = coordinate_gradient(radii, separation, first, second, atom);
        const auto neighbor = coordinate_gradient(radii, separation, first, second, other);
        if (edge.factor) {
          place_pair(first_rank, atom, other, self, neighbor, edge.slope / edge.factor);
        } else if (product.zero == 1 || other == product.second) {
          place_pair(second_rank, atom, other, self, neighbor, edge.slope);
        } else {
          place_pair(first_rank, atom, other, self, neighbor, edge.slope);
        }
      }
    }
    for (auto* rank : {first_rank, second_rank}) {
      for (int axis = 0; axis < 3; ++axis) {
        double translated{};
        for (int index = 0; index < atoms; ++index)
          if (index != owner) translated -= rank[3 * index + axis];
        rank[3 * owner + axis] = translated;
      }
    }
    products[atom] = product;
  }
  __syncthreads();
  bool valid = true;
  for (int coordinate = atom; coordinate < coordinates; coordinate += blockDim.x) {
    double total_derivative{}, owner_derivative{};
    for (int center = 0; center < atoms; ++center) {
      const auto row = products[center];
      const int index = (2 * center) * coordinates + coordinate;
      const double first_value = right_panel[index],
                   second_value = right_panel[index + coordinates];
      if (row.zero == 0) {
        const double derivative = row.nonzero * first_value;
        total_derivative += derivative;
        if (center == owner) owner_derivative = derivative;
        left_panel[index] = row.seed * first_value;
        left_panel[index + coordinates] = 0;
      } else if (row.zero < 3) {
        if (row.zero == 1) {
          const double derivative = row.nonzero * second_value;
          total_derivative += derivative;
          if (center == owner) owner_derivative = derivative;
        }
        left_panel[index] = row.seed * second_value;
        left_panel[index + coordinates] = row.seed * first_value;
      } else {
        left_panel[index] = left_panel[index + coordinates] = 0;
      }
      valid = valid && isfinite(left_panel[index] * first_value) &&
              isfinite(left_panel[index + coordinates] * second_value);
    }
    const double weight_gradient =
        measure[point] * (owner_derivative - fraction * total_derivative);
    gradient[coordinate * gradient_stride + point] = weight_gradient;
    const int rank = 2 * atoms * coordinates + coordinate;
    left_panel[rank] = -energy[point] * weight_gradient;
    right_panel[rank] = total_derivative;
    left_panel[rank + coordinates] = -energy[point] * total_derivative;
    right_panel[rank + coordinates] = weight_gradient;
    valid = valid && isfinite(weight_gradient) && isfinite(left_panel[rank] * right_panel[rank]) &&
            isfinite(left_panel[rank + coordinates] * right_panel[rank + coordinates]);
  }
  if (!valid) atomicOr(status, 16);
}

__device__ void accumulate_pair_coefficients(Edge edge, Product product, int other, double& slope,
                                             double& curvature) {
  if (product.zero == 0) {
    const double ratio = edge.slope / edge.factor;
    slope += product.seed * ratio;
    curvature += product.seed * (edge.curvature / edge.factor - ratio * ratio);
  } else if (product.zero == 1 && product.first == other) {
    slope += product.seed * edge.slope;
    curvature += product.seed * edge.curvature;
  }
}
__device__ double warp_sum(double value) {
  for (int offset = 16; offset != 0; offset /= 2)
    value += __shfl_down_sync(0xffffffff, value, offset);
  return value;
}
__global__ void tile_pair_curvature(int atoms, int count, int owner, const Vector* centers,
                                    const Radius* radii, const Edge* edges, const Product* products,
                                    double* hessian) {
  const int first = blockIdx.x, second = blockIdx.y;
  if (first <= second) return;
  const int pairs = atoms * (atoms - 1) / 2, edge_index = pair_index(first, second);
  const auto separation = radius(centers[first], centers[second]);
  double entries[6][6]{};
  for (int point = threadIdx.x; point < count; point += blockDim.x) {
    const auto* rad = radii + point * atoms;
    const auto* factors = edges + point * pairs;
    const auto* row = products + point * atoms;
    double slope{}, curvature{};
    const auto edge = factors[edge_index];
    accumulate_pair_coefficients(edge, row[first], second, slope, curvature);
    accumulate_pair_coefficients({1 - edge.factor, -edge.slope, -edge.curvature}, row[second],
                                 first, slope, curvature);
    const int centers_idx[2]{first, second};
    const Vector gradient[2]{coordinate_gradient(rad, separation, first, second, first),
                             coordinate_gradient(rad, separation, first, second, second)};
    const double gap = rad[first].distance - rad[second].distance;
    const double inv2 = separation.inverse * separation.inverse;
    for (int row_index = 0; row_index < 6; ++row_index) {
      const int axis_i = row_index % 3, atom_i = row_index / 3;
      const auto ri = rad[centers_idx[atom_i]];
      const double sign_i = atom_i == 0 ? 1 : -1;
      const double ni = -sign_i * ri.unit.xyz[axis_i];
      const double di = -sign_i * separation.unit.xyz[axis_i] * inv2;
      for (int column_index = 0; column_index < 6; ++column_index) {
        const int axis_j = column_index % 3, atom_j = column_index / 3;
        const auto rj = rad[centers_idx[atom_j]];
        const double sign_j = atom_j == 0 ? 1 : -1;
        const double nj = -sign_j * rj.unit.xyz[axis_j];
        const double dj = -sign_j * separation.unit.xyz[axis_j] * inv2;
        const double same = double(axis_i == axis_j);
        const double dn = atom_i == atom_j ? sign_i * ri.inverse *
                                                 (same - ri.unit.xyz[axis_i] * ri.unit.xyz[axis_j])
                                           : 0;
        const double dd = sign_i * sign_j * inv2 * separation.inverse *
                          (3 * separation.unit.xyz[axis_i] * separation.unit.xyz[axis_j] - same);
        const double second_derivative = dn * separation.inverse + ni * dj + di * nj + gap * dd;
        entries[row_index][column_index] +=
            slope * second_derivative +
            curvature * gradient[atom_i].xyz[axis_i] * gradient[atom_j].xyz[axis_j];
      }
    }
  }
  // A tile has one owner. Reduce an edge over its points *before* scattering
  // the translated blocks, so global atomics do not scale with point count.
  for (int row_index = 0; row_index < 6; ++row_index) {
    for (int column_index = 0; column_index < 6; ++column_index) {
      const double term = warp_sum(entries[row_index][column_index]);
      if (threadIdx.x & 31) continue;
      const int atom_i = row_index < 3 ? first : second, atom_j = column_index < 3 ? first : second;
      if (atom_i == owner || atom_j == owner) continue;
      const int coordinate_i = atom_i * 3 + row_index % 3;
      const int coordinate_j = atom_j * 3 + column_index % 3;
      const int owner_i = owner * 3 + row_index % 3, owner_j = owner * 3 + column_index % 3;
      const int coordinates = 3 * atoms;
      atomicAdd(hessian + coordinate_i + coordinates * coordinate_j, term);
      atomicAdd(hessian + owner_i + coordinates * coordinate_j, -term);
      atomicAdd(hessian + coordinate_i + coordinates * owner_j, -term);
      atomicAdd(hessian + owner_i + coordinates * owner_j, term);
    }
  }
}
}  // namespace becke_fold
