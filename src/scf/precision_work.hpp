#ifndef GENERATIVEQC_SCF_PRECISION_WORK_HPP
#define GENERATIVEQC_SCF_PRECISION_WORK_HPP

#include <cstdint>
#include <vector>

#include "generativeqc/generativeqc.h"

namespace generativeqc::scf {

enum class PrecisionWorkEventKind : std::int32_t {
  MixedFock = GENERATIVEQC_PRECISION_EVENT_MIXED_FOCK,
  StrictFock = GENERATIVEQC_PRECISION_EVENT_STRICT_FOCK,
  PostScfFock = GENERATIVEQC_PRECISION_EVENT_POST_SCF_FOCK,
  FinalAudit = GENERATIVEQC_PRECISION_EVENT_FINAL_AUDIT,
  Retry = GENERATIVEQC_PRECISION_EVENT_RETRY,
  Fallback = GENERATIVEQC_PRECISION_EVENT_FALLBACK,
  Conversion = GENERATIVEQC_PRECISION_EVENT_CONVERSION,
};

enum class PrecisionWorkPhase : std::int32_t {
  Scf = GENERATIVEQC_PRECISION_PHASE_SCF,
  Refinement = GENERATIVEQC_PRECISION_PHASE_REFINEMENT,
  Finalization = GENERATIVEQC_PRECISION_PHASE_FINALIZATION,
  Retry = GENERATIVEQC_PRECISION_PHASE_RETRY,
};

enum class PrecisionOperatorKind : std::int32_t {
  CoulombJ = GENERATIVEQC_PRECISION_OPERATOR_COULOMB_J,
  ExchangeK = GENERATIVEQC_PRECISION_OPERATOR_EXCHANGE_K,
  Xc = GENERATIVEQC_PRECISION_OPERATOR_XC,
  FockAssembly = GENERATIVEQC_PRECISION_OPERATOR_FOCK_ASSEMBLY,
  PhysicalResidual = GENERATIVEQC_PRECISION_OPERATOR_PHYSICAL_RESIDUAL,
  Eigensolver = GENERATIVEQC_PRECISION_OPERATOR_EIGENSOLVER,
  DensityBuild = GENERATIVEQC_PRECISION_OPERATOR_DENSITY_BUILD,
  Diis = GENERATIVEQC_PRECISION_OPERATOR_DIIS,
  MatrixProduct = GENERATIVEQC_PRECISION_OPERATOR_MATRIX_PRODUCT,
  Diagnostics = GENERATIVEQC_PRECISION_OPERATOR_DIAGNOSTICS,
  OccupationStabilization = GENERATIVEQC_PRECISION_OPERATOR_OCCUPATION_STABILIZATION,
  CoulombRecurrence = GENERATIVEQC_PRECISION_OPERATOR_COULOMB_RECURRENCE,
  ExchangeRecurrence = GENERATIVEQC_PRECISION_OPERATOR_EXCHANGE_RECURRENCE,
};

enum class PrecisionDtype : std::int32_t {
  Unknown = GENERATIVEQC_PRECISION_DTYPE_UNKNOWN,
  Fp64 = GENERATIVEQC_PRECISION_DTYPE_FP64,
  Fp32 = GENERATIVEQC_PRECISION_DTYPE_FP32,
  Tf32 = GENERATIVEQC_PRECISION_DTYPE_TF32,
  Fp16 = GENERATIVEQC_PRECISION_DTYPE_FP16,
  Bf16 = GENERATIVEQC_PRECISION_DTYPE_BF16,
};

enum class PrecisionArithmeticMode : std::int32_t {
  Strict = GENERATIVEQC_PRECISION_ARITHMETIC_STRICT,
  Mixed = GENERATIVEQC_PRECISION_ARITHMETIC_MIXED,
  Tf32 = GENERATIVEQC_PRECISION_ARITHMETIC_TF32,
  Fp16 = GENERATIVEQC_PRECISION_ARITHMETIC_FP16,
  Bf16 = GENERATIVEQC_PRECISION_ARITHMETIC_BF16,
};

struct PrecisionWorkEvent {
  PrecisionWorkEventKind kind{};
  PrecisionWorkPhase phase{};
  std::uint64_t sequence{};
  std::uint32_t iteration{};
  std::uint64_t owner_id{};
  std::uint64_t solve_epoch{};
  std::uint64_t state_generation{};
};

struct PrecisionOperatorRecord {
  PrecisionOperatorKind kind{};
  PrecisionDtype storage{PrecisionDtype::Unknown};
  PrecisionDtype compute{PrecisionDtype::Unknown};
  PrecisionDtype accumulation{PrecisionDtype::Unknown};
  PrecisionDtype reduction{PrecisionDtype::Unknown};
  PrecisionArithmeticMode arithmetic_mode{};
  std::uint64_t count{};
};

/** Execution-owned variable-length provenance.
 *
 * The default is intentionally a versioned but uncertified empty record. A
 * producer may retain partial rows after a numerical failure, but it must not
 * set either completeness flag unless its full returned-attempt inventory is
 * present. API adapters copy this record verbatim and never derive rows from
 * aggregate PrecisionProvenance counters. */
struct PrecisionWork {
  std::uint32_t detail_version{GENERATIVEQC_PRECISION_WORK_DETAIL_VERSION};
  bool complete{};
  bool operator_inventory_complete{};
  std::uint64_t conversion_count{};
  std::uint64_t fallback_count{};
  std::uint64_t owner_id{};
  std::uint64_t returned_solve_epoch{};
  std::uint64_t returned_state_generation{};
  std::vector<PrecisionWorkEvent> events;
  std::vector<PrecisionOperatorRecord> operators;
};

}  // namespace generativeqc::scf

#endif
