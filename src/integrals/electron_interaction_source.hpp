#ifndef GENERATIVEQC_INTEGRALS_ELECTRON_INTERACTION_SOURCE_HPP
#define GENERATIVEQC_INTEGRALS_ELECTRON_INTERACTION_SOURCE_HPP

#include <array>
#include <cstddef>
#include <stdexcept>

#include "core/types.hpp"

namespace generativeqc::integrals {

/** Method-neutral AO interaction operators exposed by a bounded source.
 *
 * The source contract deliberately stops below SCF/DFT/CC semantics: consumers
 * ask only for mathematically defined AO-space tiles. Fock assembly, orbital
 * transforms, correlation equations and response remain owned by their method
 * layers.
 */
enum class ElectronInteractionOperator { overlap, hcore, eri, metric, three_center };

/** Caller-owned device destination for an optional zero-host-staging read.
 *
 * The source does not own or extend the lifetime of values/stream. A successful
 * read_device() call only guarantees that work was enqueued in-order on this
 * stream; the caller owns completion/publication. Opaque stream spelling keeps
 * this method-neutral contract usable by CPU-only translation units.
 */
struct DeviceInteractionTarget {
  int device{-1};
  void* stream{};
  double* values{};
  std::size_t capacity{};
};

/** Read-only AO interaction source shared by mean-field and post-HF adapters.
 *
 * Implementations own their normalized geometry/basis state and may generate,
 * cache or stream values internally. A consumer must check supports() before
 * requesting an optional operator. read() is transactional with respect to the
 * caller-owned output buffer: validation must complete before publishing values.
 */
class ElectronInteractionSource {
 public:
  using Operator = ElectronInteractionOperator;

  virtual ~ElectronInteractionSource() = default;
  virtual const core::System& orbital() const = 0;
  virtual std::size_t nbf() const = 0;
  virtual std::size_t naux() const = 0;
  /** Numeric bytes retained by this source while a consumer borrows it.
   * Consumers use this value for endpoint memory admission; it must not omit
   * resident value tensors merely because they are immutable or shared.
   */
  virtual std::size_t retained_numeric_bytes() const = 0;
  virtual bool supports(Operator op) const noexcept = 0;

  /** Optional device-resident value path. False means callers must use read().
   * This capability never changes operator semantics or authorizes screening.
   */
  virtual bool supports_device_read(Operator op, int device) const noexcept {
    (void)op;
    (void)device;
    return false;
  }

  virtual void read(Operator op, const std::array<std::size_t, 4>& begin,
                    const std::array<std::size_t, 4>& count, double* out,
                    std::size_t elements) const = 0;

  /** Enqueue one exact row-major tile directly into caller-owned device storage.
   * Implementations must validate the same bounds/operator contract as read(),
   * write exactly elements values, and use target.stream on target.device.
   * No synchronization or host publication is implied.
   */
  virtual void read_device(Operator op, const std::array<std::size_t, 4>& begin,
                           const std::array<std::size_t, 4>& count,
                           DeviceInteractionTarget target,
                           std::size_t elements) const {
    (void)op;
    (void)begin;
    (void)count;
    (void)target;
    (void)elements;
    throw std::invalid_argument("interaction source has no device-read capability");
  }
};

}  // namespace generativeqc::integrals
#endif
