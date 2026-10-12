#ifndef GENERATIVEQC_SCF_TARGET_FOCK_WORK_HPP
#define GENERATIVEQC_SCF_TARGET_FOCK_WORK_HPP

#include <cstdint>
#include <limits>
#include <optional>

namespace generativeqc::scf {

/** Execution-local physical target Fock census, independent of returned history.
 * Every started attempt must finish with an independently certified count.
 * An exception, missing inventory, overlap, or overflow permanently revokes
 * this execution's total; a successful cold retry cannot repair missing work.
 * Preliminary-method work has its own diagnostic and is not target work.
 */
class TargetFockWork {
 public:
  void begin_attempt() noexcept {
    if (pending_) valid_ = false;
    started_ = pending_ = true;
  }

  void finish_attempt(std::optional<std::uint64_t> builds) noexcept {
    if (!pending_ || !builds || *builds > std::numeric_limits<std::uint64_t>::max() - builds_) {
      valid_ = false;
    } else if (valid_) {
      builds_ += *builds;
    }
    pending_ = false;
  }

  /** No iterations-derived estimate or partial total is ever returned. */
  std::optional<std::uint64_t> total() const noexcept {
    if (!started_ || pending_ || !valid_) return std::nullopt;
    return builds_;
  }

 private:
  std::uint64_t builds_{};
  bool started_{}, pending_{}, valid_{true};
};

}  // namespace generativeqc::scf

#endif
