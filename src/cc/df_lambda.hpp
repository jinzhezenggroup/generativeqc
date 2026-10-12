#pragma once

#include <memory>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "cc/lambda_response.hpp"

namespace generativeqc::cc::detail {

/** Borrowed inputs/error flag on the DF action owner's existing stream.
 * The action owner outlives all queued consumers; this view conveys no new
 * scientific generation or independent storage ownership.
 */
struct DFLambdaResidentState {
  void* stream{};
  double* seed_one{};
  double* seed_two{};
  int* error{};
};

/** Resident native DF scientific actions for the existing Lambda solve.
 * Retained-core and every auxiliary derivative execute on one stream. Inputs
 * are staged once; outputs detach only after the complete sticky-error audit.
 * The full numeric bound includes borrowed problem/CC state, host Krylov and
 * publication storage, and the device arena. No ovvv/vvvv is reconstructed.
 */
class DFLambdaActions {
 public:
  DFLambdaActions(const Problem&, const SolverResult&, const LambdaOptions&, int device,
                  bool with_source, bool with_parameters);
  ~DFLambdaActions();
  DFLambdaActions(const DFLambdaActions&) = delete;
  DFLambdaActions& operator=(const DFLambdaActions&) = delete;
  void replay(double& energy, std::vector<double>& r1, std::vector<double>& r2);
  void rhs(bool independent, std::vector<double>& one, std::vector<double>& two);
  /** One bounded warmup epoch borrows otherwise idle strict-audit scratch.
   * Begin only after the fresh FP64 primal/RHS checks. Finishing invalidates
   * every approximate buffer permanently before strict solve/audit/VJP work.
   */
  bool begin_mixed_precision();
  void finish_mixed_precision();
  void transpose(bool independent, std::span<const double> one, std::span<const double> two,
                 std::vector<double>& out_one, std::vector<double>& out_two);
  [[nodiscard]] DFLambdaResidentState resident_state() const noexcept;
  /** Enqueue the existing physical transpose using already resident seeds.
   * Outputs are borrowed until the next action. Enqueue their consumers on
   * resident_state().stream and audit its sticky error before publication.
   * This entry point neither stages dimension-sized data nor fences success.
   */
  [[nodiscard]] std::pair<const double*, const double*> transpose_device(bool independent);
  void seeds(std::span<const double> one, std::span<const double> two);
  std::vector<double> parameter(std::string_view name, std::size_t count);
  std::pair<std::vector<double>, std::vector<double>> virtual_factors();
  const LambdaDiagnostic& diagnostic() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace generativeqc::cc::detail
