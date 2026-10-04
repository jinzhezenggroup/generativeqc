#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>
struct Shell {
  int angular, atom;
  std::vector<long double> a, w;
};
static std::vector<Shell> shells(int count, const int* info, const double* coeff) {
  std::vector<Shell> out(count);
  for (int i = 0; i < count; ++i) {
    auto& sh = out[i];
    sh.angular = info[4 * i];
    sh.atom = info[4 * i + 1];
    int offset = info[4 * i + 2], n = info[4 * i + 3];
    sh.a.resize(n);
    sh.w.resize(n);
    long double norm = 0;
    for (int k = 0; k < n; ++k) {
      sh.a[k] = coeff[2 * (offset + k)];
      sh.w[k] = coeff[2 * (offset + k) + 1];
    }
    for (int k = 0; k < n; ++k)
      for (int l = 0; l < n; ++l)
        norm += sh.w[k] * sh.w[l] *
                powl(2 * sqrtl(sh.a[k] * sh.a[l]) / (sh.a[k] + sh.a[l]), sh.angular + 1.5L);
    for (int k = 0; k < n; ++k)
      sh.w[k] *= powl(2 * sh.a[k] / std::numbers::pi_v<long double>, .75L) *
                 powl(4 * sh.a[k], .5L * sh.angular) / sqrtl(norm);
  }
  return out;
}
// Independent s/s/s expression and its first/second Gaussian center derivatives.
// No fitted roots and no native compiler table/recurrence is used here.
extern "C" void analytic_low_degree(int n, int q, int ns, int qs, const int* si, const int* qi,
                                    const double* sc, const double* qc, const int* ao,
                                    const int* aux, const double* xyz, long double* raw) {
  auto s = shells(ns, si, sc), z = shells(qs, qi, qc);
#pragma omp parallel for
  for (int mu = 0; mu < n; ++mu)
    for (int nu = 0; nu < n; ++nu)
      for (int P = 0; P < q; ++P) {
        const auto &A = s[ao[2 * mu]], &B = s[ao[2 * nu]], &C = z[aux[2 * P]];
        if (A.angular + B.angular + C.angular > 2) continue;
        long double ans = 0;
        for (std::size_t ia = 0; ia < A.a.size(); ++ia)
          for (std::size_t ib = 0; ib < B.a.size(); ++ib)
            for (std::size_t ic = 0; ic < C.a.size(); ++ic) {
              long double a = A.a[ia], b = B.a[ib], g = C.a[ic], p = a + b, rho = p * g / (p + g);
              long double ab[3], d[3], ab2 = 0, d2 = 0;
              for (int k = 0; k < 3; ++k) {
                long double ac =
                    (long double)xyz[3 * A.atom + k] - (long double)xyz[3 * C.atom + k];
                ab[k] = (long double)xyz[3 * A.atom + k] - (long double)xyz[3 * B.atom + k];
                d[k] = ac - b / p * ab[k];
                ab2 += ab[k] * ab[k];
                d2 += d[k] * d[k];
              }
              long double t = rho * d2, f0 = 0, f1 = 0, f2 = 0;
              if (t < .5L) {
                long double term = 1;
                for (int k = 0; k < 40; ++k) {
                  f0 += term / (2 * k + 1);
                  f1 += term / (2 * k + 3);
                  f2 += term / (2 * k + 5);
                  term *= -t / (k + 1);
                }
              } else {
                f0 = sqrtl(std::numbers::pi_v<long double>) / (2 * sqrtl(t)) * erfl(sqrtl(t));
                f1 = (f0 - expl(-t)) / (2 * t);
                f2 = (3 * f1 - expl(-t)) / (2 * t);
              }
              long double v = f0;
              if (A.angular == 1) {
                int k = ao[2 * mu + 1];
                v = -b / p * ab[k] * f0 - g / (p + g) * d[k] * f1;
              }
              if (B.angular == 1) {
                int k = ao[2 * nu + 1];
                v = a / p * ab[k] * f0 - g / (p + g) * d[k] * f1;
              }
              if (C.angular == 1) {
                int k = aux[2 * P + 1];
                v = p / (p + g) * d[k] * f1;
              }
              if (A.angular + B.angular + C.angular == 2 && A.angular < 2 && B.angular < 2 &&
                  C.angular < 2) {
                const long double sx = g / (p + g), sy = p / (p + g);
                if (A.angular && B.angular) {
                  int i = ao[2 * mu + 1], j = ao[2 * nu + 1];
                  long double pa = -b / p * ab[i], pb = a / p * ab[j];
                  v = (pa * pb + (i == j ? .5L / p : 0)) * f0 -
                      sx * (pa * d[j] + pb * d[i] + (i == j ? .5L / p : 0)) * f1 +
                      sx * sx * d[i] * d[j] * f2;
                } else {
                  int i = A.angular ? ao[2 * mu + 1] : ao[2 * nu + 1], j = aux[2 * P + 1];
                  long double pa = (A.angular ? -b : a) / p * ab[i];
                  v = (pa * sy * d[j] + (i == j ? .5L / (p + g) : 0)) * f1 -
                      sx * sy * d[i] * d[j] * f2;
                }
              }
              if (A.angular == 2 || B.angular == 2 || C.angular == 2) {
                int slot = A.angular == 2 ? 0 : (B.angular == 2 ? 1 : 2);
                int component =
                    slot == 0 ? ao[2 * mu + 1] : (slot == 1 ? ao[2 * nu + 1] : aux[2 * P + 1]);
                auto moment = [&](int i, int j) {
                  long double mi = slot == 2 ? 0 : (slot == 0 ? -b : a) / p * ab[i];
                  long double mj = slot == 2 ? 0 : (slot == 0 ? -b : a) / p * ab[j];
                  long double slope = slot == 2 ? p / (p + g) : -g / (p + g);
                  long double vi = slot == 2 ? .5L / g : .5L / p;
                  long double vc = slot == 2 ? -p / (p + g) * vi : -g / (p + g) * vi;
                  return (mi * mj + (i == j ? vi : 0)) * f0 +
                         (slope * (mi * d[j] + mj * d[i]) + (i == j ? vc : 0)) * f1 +
                         slope * slope * d[i] * d[j] * f2;
                };
                // Independently normalized real l=2 harmonics, libcint m=-2..2.
                if (component == 0) v = moment(0, 1);
                if (component == 1) v = moment(1, 2);
                if (component == 2)
                  v = (moment(2, 2) - .5L * moment(0, 0) - .5L * moment(1, 1)) / sqrtl(3);
                if (component == 3) v = moment(0, 2);
                if (component == 4) v = .5L * (moment(0, 0) - moment(1, 1));
              }
              ans += A.w[ia] * B.w[ib] * C.w[ic] *
                     (2 * powl(std::numbers::pi_v<long double>, 2.5L)) / (p * g * sqrtl(p + g)) *
                     expl(-a * b / p * ab2) * v;
            }
        raw[((std::size_t)mu * n + nu) * q + P] = ans;
      }
}
#include <cstddef>
#include <vector>
// Diagnostic accumulation oracle: FP64 input arrays, 64-bit mantissa products
// and sums. Independent auxiliary pages bound scratch to 2*n*n long doubles
// per OpenMP thread; preserve every raw pair without symmetrization.
template <class T>
void transform(std::size_t n, std::size_t q, const T* raw, const double* c, double* mo,
               bool round_first = false) {
#pragma omp parallel for
  for (std::size_t p = 0; p < q; ++p) {
    std::vector<long double> first(n * n), page(n * n);
    for (std::size_t m = 0; m < n; ++m)
      for (std::size_t k = 0; k < n; ++k) page[m * n + k] = raw[(m * n + k) * q + p];
    for (std::size_t m = 0; m < n; ++m)
      for (std::size_t b = 0; b < n; ++b) {
        long double sum = 0;
        for (std::size_t k = 0; k < n; ++k) sum += page[m * n + k] * (long double)c[k * n + b];
        first[m * n + b] = round_first ? (long double)(double)sum : sum;
      }
    for (std::size_t a = 0; a < n; ++a)
      for (std::size_t b = 0; b < n; ++b) {
        long double sum = 0;
        for (std::size_t k = 0; k < n; ++k) sum += (long double)c[k * n + a] * first[k * n + b];
        mo[(p * n + a) * n + b] = (double)sum;
      }
  }
}
extern "C" void extended_mo(std::size_t n, std::size_t q, const double* r, const double* c,
                            double* m) {
  transform(n, q, r, c, m);
}
extern "C" void extended_mo_ld(std::size_t n, std::size_t q, const long double* r, const double* c,
                               double* m) {
  transform(n, q, r, c, m);
}
extern "C" void extended_mo_round(std::size_t n, std::size_t q, const double* r, const double* c,
                                  double* m) {
  transform(n, q, r, c, m, true);
}
