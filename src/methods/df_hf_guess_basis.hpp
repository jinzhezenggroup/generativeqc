#pragma once

#include <array>

namespace generativeqc::methods::detail {
/** Offline aug-cc-pVTZ-JKFIT H/C records from PySCF 2.14.0.
 * Exponents are raw; native normalization owns all AO conventions.
 * This basis is used only for a provisional density, never the accepted reference.
 * Reproduce with benchmarks/df_hf_preconvergence.py prepare-jk; provenance and
 * the frozen metadata SHA-256 are retained in the default-policy Agent Note. */
struct DFHFGuessBasisShell {
  int atomic_number;
  unsigned angular_momentum;
  double exponent;
};
inline constexpr std::array<DFHFGuessBasisShell, 44> df_hf_guess_basis{{
    {6, 0, 1113.9867718999999},
    {6, 0, 369.16234179999998},
    {6, 0, 121.79275232000001},
    {6, 0, 48.127114540000001},
    {6, 0, 20.365074},
    {6, 0, 8.0883596900000008},
    {6, 0, 2.5068656599999999},
    {6, 0, 1.24385374},
    {6, 0, 0.48449900000000001},
    {6, 0, 0.19185160000000001},
    {6, 0, 0.075969270000000005},
    {6, 1, 102.99176249},
    {6, 1, 28.132594009999998},
    {6, 1, 9.8364318199999996},
    {6, 1, 3.3490544999999998},
    {6, 1, 1.4947618600000001},
    {6, 1, 0.57690109000000001},
    {6, 1, 0.20320062999999999},
    {6, 1, 0.071572919999999998},
    {6, 2, 10.59406836},
    {6, 2, 3.5997195400000002},
    {6, 2, 1.33556911},
    {6, 2, 0.51949765000000003},
    {6, 2, 0.19954125},
    {6, 2, 0.07664464},
    {6, 3, 1.1948663399999999},
    {6, 3, 0.41586634},
    {6, 3, 0.14473987999999999},
    {6, 4, 0.85886633999999995},
    {6, 4, 0.34354654000000001},
    {1, 0, 9.5302493300000002},
    {1, 0, 1.9174506200000001},
    {1, 0, 0.68424048999999998},
    {1, 0, 0.28413255999999998},
    {1, 0, 0.11798675},
    {1, 1, 2.9133231999999998},
    {1, 1, 1.26212054},
    {1, 1, 0.50199775999999996},
    {1, 1, 0.19966535999999999},
    {1, 2, 2.3135329100000002},
    {1, 2, 0.71290724000000005},
    {1, 2, 0.21967992},
    {1, 3, 1.65657261},
    {1, 3, 0.66262904},
}};
}  // namespace generativeqc::methods::detail
