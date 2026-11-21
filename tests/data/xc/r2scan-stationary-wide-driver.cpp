real limit_floor(real value, real threshold) {
  const real fraction = fmaxq(0, fminq(1, value / threshold - 0.5Q));
  return fmaxq(value, threshold * (1 + 0.5Q * fraction * fraction * fraction * (2 - fraction)));
}

real gate(real value, real threshold) {
  const real fraction = fmaxq(0, fminq(1, value / threshold - 1));
  return fraction * fraction * fraction * (10 - 15 * fraction + 6 * fraction * fraction);
}

real oracle_energy(const real* point) {
  info_type info;
  mgga_c_r2scan_params correlation{0.001Q};
  mgga_x_r2scan_params exchange{0.667Q, 0.8Q, 1.24Q, 0.065Q, 0.001Q, 0.361Q};
  constexpr real cutoff_x = static_cast<real>(1e-11), cutoff_c = static_cast<real>(1e-15);
  constexpr real tau_floor = static_cast<real>(1e-20);
  real sigma[3]{}, kinetic[] = {limit_floor(point[8], tau_floor), limit_floor(point[9], tau_floor)};
  for (unsigned axis = 0; axis < 3; ++axis) {
    const real alpha = point[2 + axis], beta = point[5 + axis];
    sigma[0] += alpha * alpha;
    sigma[1] += alpha * beta;
    sigma[2] += beta * beta;
  }
  real energy = 0;
  for (unsigned spin = 0; spin < 2; ++spin) {
    if (point[spin] <= cutoff_x) continue;
    real rho[] = {point[spin], cutoff_x};
    real gradient[] = {sigma[2 * spin], 0, 0};
    real tau[] = {kinetic[spin], tau_floor};
    real values[8]{};
    xc_mgga_out_params out{values, values + 1, values + 3, nullptr, values + 6};
    xc_func_type state{&exchange, cutoff_x, 0x1p-52Q, &info, {}};
    x_vxc(&state, 0, rho, gradient, nullptr, tau, &out);
    energy += gate(point[spin], cutoff_x) * values[0] * (point[spin] + cutoff_x);
  }
  const real total = point[0] + point[1];
  if (total > cutoff_c) {
    real rho[] = {limit_floor(point[0], cutoff_c), limit_floor(point[1], cutoff_c)};
    real values[8]{};
    xc_mgga_out_params out{values, values + 1, values + 3, nullptr, values + 6};
    xc_func_type state{&correlation, cutoff_c, 0x1p-52Q, &info, {}};
    c_vxc(&state, 0, rho, sigma, nullptr, kinetic, &out);
    energy += gate(total, cutoff_c) * values[0] * total;
  }
  return energy;
}

// A fourth-order stencil of upstream energies is independent of the point
// compiler's differentiation and does not subtract nearly equal FP64 values.
void wide(const double* point, const double* direction, double* output) {
  real center[10];
  for (unsigned channel = 0; channel < 10; ++channel) center[channel] = point[channel];
  const real energy = oracle_energy(center);
  real first[2]{}, second[2]{};
  for (unsigned index = 0; index < 2; ++index) {
    const real step = index == 0 ? 0x1p-18Q : 0x1p-19Q;
    real above[10], below[10];
    for (unsigned channel = 0; channel < 10; ++channel) {
      above[channel] = center[channel] + step * direction[channel];
      below[channel] = center[channel] - step * direction[channel];
    }
    const real plus = oracle_energy(above), minus = oracle_energy(below);
    first[index] = (plus - minus) / (2 * step);
    second[index] = (plus + minus - 2 * energy) / (step * step);
  }
  output[0] = static_cast<double>(energy);
  output[1] = static_cast<double>((4 * first[1] - first[0]) / 3);
  output[2] = static_cast<double>((4 * second[1] - second[0]) / 3);
}

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  FILE* file = fopen(argv[1], "rb");
  uint64_t count;
  if (!file || fread(&count, sizeof(count), 1, file) != 1 || count > 100000) return 3;
  std::vector<double> inputs(20 * count), outputs(3 * count);
  if (fread(inputs.data(), sizeof(double), inputs.size(), file) != inputs.size()) return 4;
  fclose(file);
  for (size_t index = 0; index < count; ++index)
    wide(inputs.data() + 20 * index, inputs.data() + 20 * index + 10, outputs.data() + 3 * index);
  file = fopen(argv[2], "wb");
  if (!file || fwrite(&count, sizeof(count), 1, file) != 1 ||
      fwrite(outputs.data(), sizeof(double), outputs.size(), file) != outputs.size())
    return 5;
  return fclose(file) == 0 ? 0 : 6;
}
