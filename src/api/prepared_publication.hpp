#pragma once

namespace generativeqc::api {
/** The method's numerical return is not yet a successful ABI publication.
 * Argument rejection and copy-out failure must revoke the original #929 leases,
 * including failures before the native endpoint was entered. */
template <class Owner>
class PreparedPublication {
 public:
  explicit PreparedPublication(Owner& owner) noexcept : owner_(owner) {}
  PreparedPublication(const PreparedPublication&) = delete;
  PreparedPublication& operator=(const PreparedPublication&) = delete;
  ~PreparedPublication() {
    if (!committed_) owner_.reject_execution();
  }
  void commit() noexcept { committed_ = true; }

 private:
  Owner& owner_;
  bool committed_{};
};
}  // namespace generativeqc::api
