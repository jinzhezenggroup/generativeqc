#ifndef GENERATIVEQC_GENERATIVEQC_HPP
#define GENERATIVEQC_GENERATIVEQC_HPP

#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "generativeqc/generativeqc.h"

namespace generativeqc {

/** Exception retaining the native status associated with a failed call. */
/** @native-contract generativeqc::Error
 * @behavior Native failure exception derived from std::runtime_error.
 * @outputs what() retains the supplied message; status() returns its associated
 * generativeqc_status.
 * @lifetime Owns a copied message and status independently of native handles.
 * @errors Constructing the exception can allocate and throw a standard allocation exception.
 */
class Error : public std::runtime_error {
 public:
  /** @native-contract generativeqc::Error::Error
   * @behavior Construct an exception with a native status and copied diagnostic message.
   * @inputs status is preserved verbatim; message is a std::string and may be empty.
   * @lifetime Copies the message, so the source string need not outlive the exception.
   * @errors std::runtime_error construction may throw std::bad_alloc.
   */
  Error(generativeqc_status status, const std::string& message)
      : std::runtime_error(message), status_(status) {}
  /** @native-contract generativeqc::Error::status
   * @behavior Read the native status stored in this exception.
   * @outputs Returns the exact construction status by value.
   * @errors noexcept; does not throw.
   * @execution Const synchronous read; no native handle access.
   */
  [[nodiscard]] generativeqc_status status() const noexcept { return status_; }

 private:
  generativeqc_status status_;
};

/** @native-contract generativeqc::check
 * @behavior Convert a failed C status into Error.
 * @inputs Any native status code; SUCCESS is the only nonthrowing value.
 * @errors Throws Error for non-SUCCESS, using generativeqc_status_message; exception
 * construction may throw std::bad_alloc.
 * @execution Synchronous scalar check; does not execute or query a native owner.
 */
inline void check(generativeqc_status status) {
  if (status != GENERATIVEQC_STATUS_SUCCESS) {
    throw Error(status, generativeqc_status_message(status));
  }
}

/** @native-contract generativeqc::MethodCapabilities
 * @behavior Value snapshot of native method-registry capabilities.
 * @outputs method and family identify the registry entry; supported_properties is a bitmask,
 * available and supports_batch are registry booleans. They do not establish every
 * device/basis/precision combination.
 * @lifetime Self-contained value; default initialization yields zero/false fields and is not a
 * successful query.
 */
struct MethodCapabilities {
  generativeqc_method method{};
  generativeqc_method_family family{};
  generativeqc_property_flags supported_properties{};
  bool available{};
  bool supports_batch{};
};

/** Query the native registry without preparing a system or calculation. */
/** @native-contract generativeqc::method_capabilities
 * @behavior Return an owned registry capability value for a method.
 * @inputs A generated generativeqc_method identifier; no context is required.
 * @outputs Returns MethodCapabilities after internally initializing/copying the C descriptor.
 * @lifetime Returned value owns its scalars; no native resource is retained.
 * @errors Throws Error for unknown method or failed query; ordinary allocation exceptions may
 * propagate.
 * @execution Synchronous registry query; no numerical execution.
 */
inline MethodCapabilities method_capabilities(generativeqc_method method) {
  generativeqc_method_capabilities_descriptor native{
      sizeof(generativeqc_method_capabilities_descriptor), GENERATIVEQC_ABI_VERSION, 0, 0, 0, 0, 0};
  check(generativeqc_method_get_capabilities(method, &native));
  return {native.method, native.family, native.supported_properties, native.available != 0,
          native.supports_batch != 0};
}

/** Move-only context owner; dependent native objects must be destroyed first. */
/** @native-contract generativeqc::Context
 * @behavior Move-constructible RAII context owner; copying is deleted.
 * @lifetime Owns one native context; dependent calculation/batch/correction plans must be
 * destroyed first. Moving transfers the handle; the source has a null handle and is only
 * supported for destruction. No move assignment is provided.
 * @errors Constructors/operations can throw Error for native failure and standard allocation
 * exceptions. Destruction and the move constructor are nonthrowing.
 * @execution Serialize operations and lifetime changes involving the same object/context. RAII
 * does not provide independent thread-safe execution.
 */
class Context {
 public:
  /** @native-contract generativeqc::Context::Context
   * @behavior Construct and own a prepared native context.
   * @inputs Supply an initialized generativeqc_context_descriptor. C descriptor
   * initialization/admission rules apply.
   * @lifetime Copies descriptor options and owns the created context.
   * @errors Throws Error on native validation/preparation failure; no live wrapper is returned.
   * Allocation failures may throw standard exceptions.
   * @execution Synchronous construction; serialize with operations on the same Context.
   * @units Uses C descriptor atomic units: Bohr geometry, Hartree energies and Hartree/Bohr
   * forces; memory limits are bytes.
   */
  explicit Context(const generativeqc_context_descriptor& descriptor) {
    check(generativeqc_context_create(&descriptor, &handle_));
  }
  /** @native-contract generativeqc::Context::~Context
   * @behavior Destroy the owned native context and release resources.
   * @lifetime Owns one native context; dependent calculation/batch/correction plans must be
   * destroyed first. A moved-from object holds NULL and destruction is harmless.
   * @errors Nonthrowing destructor; calls the matching C destroy function.
   * @execution Synchronous; never race destruction with an operation or borrowed-handle use.
   */
  ~Context() { generativeqc_context_destroy(handle_); }
  /** @native-contract generativeqc::Context::Context-copy
   * @behavior Deleted copy operation: native ownership cannot be duplicated.
   * @lifetime Use move construction for exclusive ownership transfer; this declaration cannot
   * be called.
   * @errors Attempting this operation is a compile-time error, not a runtime Error.
   */
  Context(const Context&) = delete;
  /** @native-contract generativeqc::Context::operator=-copy
   * @behavior Deleted copy operation: native ownership cannot be duplicated.
   * @lifetime Use move construction for exclusive ownership transfer; this declaration cannot
   * be called.
   * @errors Attempting this operation is a compile-time error, not a runtime Error.
   */
  Context& operator=(const Context&) = delete;
  /** @native-contract generativeqc::Context::Context-move
   * @behavior Transfer exclusive ownership from other.
   * @inputs An rvalue Context; caller prevents concurrent use or destruction during transfer.
   * @lifetime Destination acquires the native handle and associated metadata; other retains no
   * owning handle. Existing dependent native objects still require their context to remain
   * alive.
   * @errors noexcept; no native preparation is repeated.
   * @execution Synchronous ownership transfer; moved-from operations other than destruction are
   * outside the wrapper contract.
   */
  Context(Context&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
  /** @native-contract generativeqc::Context::get
   * @behavior Borrow the underlying native handle.
   * @outputs Returns the held pointer, or NULL after move construction.
   * @lifetime Does not transfer ownership; never destroy the returned handle independently.
   * Valid only while the owning wrapper/native resource remains alive.
   * @errors noexcept; does not validate the native handle.
   * @execution Synchronous read; serialize against moves/destruction.
   */
  [[nodiscard]] generativeqc_context* get() const noexcept { return handle_; }

 private:
  generativeqc_context* handle_ = nullptr;
};

/** Move-only owner of copied system data; preparation consumers copy it again. */
/** @native-contract generativeqc::System
 * @behavior Move-constructible RAII system owner; copying is deleted.
 * @lifetime Owns a copied native system. Successful prepared consumers have their own system
 * data. Moving transfers the handle; the source has a null handle and is only supported for
 * destruction. No move assignment is provided.
 * @errors Constructors/operations can throw Error for native failure and standard allocation
 * exceptions. Destruction and the move constructor are nonthrowing.
 * @execution Serialize operations and lifetime changes involving the same object/context. RAII
 * does not provide independent thread-safe execution.
 */
class System {
 public:
  /** @native-contract generativeqc::System::System
   * @behavior Construct and own a prepared native system.
   * @inputs Supply a live Context and initialized generativeqc_system_descriptor. C descriptor
   * initialization/admission rules apply.
   * @lifetime Copies the system buffers; the descriptor may be released after construction.
   * @errors Throws Error on native validation/preparation failure; no live wrapper is returned.
   * Allocation failures may throw standard exceptions.
   * @execution Synchronous construction; serialize with operations on the same Context.
   * @units Uses C descriptor atomic units: Bohr geometry, Hartree energies and Hartree/Bohr
   * forces; memory limits are bytes.
   */
  System(Context& context, const generativeqc_system_descriptor& descriptor)
      : atom_count_(descriptor.atom_count) {
    check(generativeqc_system_create(context.get(), &descriptor, &handle_));
  }
  /** @native-contract generativeqc::System::~System
   * @behavior Destroy the owned native system and release resources.
   * @lifetime Owns a copied native system. Successful prepared consumers have their own system
   * data. A moved-from object holds NULL and destruction is harmless.
   * @errors Nonthrowing destructor; calls the matching C destroy function.
   * @execution Synchronous; never race destruction with an operation or borrowed-handle use.
   */
  ~System() { generativeqc_system_destroy(handle_); }
  /** @native-contract generativeqc::System::System-copy
   * @behavior Deleted copy operation: native ownership cannot be duplicated.
   * @lifetime Use move construction for exclusive ownership transfer; this declaration cannot
   * be called.
   * @errors Attempting this operation is a compile-time error, not a runtime Error.
   */
  System(const System&) = delete;
  /** @native-contract generativeqc::System::operator=-copy
   * @behavior Deleted copy operation: native ownership cannot be duplicated.
   * @lifetime Use move construction for exclusive ownership transfer; this declaration cannot
   * be called.
   * @errors Attempting this operation is a compile-time error, not a runtime Error.
   */
  System& operator=(const System&) = delete;
  /** @native-contract generativeqc::System::System-move
   * @behavior Transfer exclusive ownership from other.
   * @inputs An rvalue System; caller prevents concurrent use or destruction during transfer.
   * @lifetime Destination acquires the native handle and associated metadata; other retains no
   * owning handle. Existing dependent native objects still require their context to remain
   * alive.
   * @errors noexcept; no native preparation is repeated.
   * @execution Synchronous ownership transfer; moved-from operations other than destruction are
   * outside the wrapper contract.
   */
  System(System&& other) noexcept
      : handle_(std::exchange(other.handle_, nullptr)), atom_count_(other.atom_count_) {}
  /** @native-contract generativeqc::System::get
   * @behavior Borrow the underlying native handle.
   * @outputs Returns the held pointer, or NULL after move construction.
   * @lifetime Does not transfer ownership; never destroy the returned handle independently.
   * Valid only while the owning wrapper/native resource remains alive.
   * @errors noexcept; does not validate the native handle.
   * @execution Synchronous read; serialize against moves/destruction.
   */
  [[nodiscard]] generativeqc_system* get() const noexcept { return handle_; }
  /** @native-contract generativeqc::System::atom_count
   * @behavior Return the copied input atom count.
   * @outputs Returns the construction descriptor atom_count by value; it does not query a
   * native owner.
   * @errors noexcept; no allocation.
   * @execution Const synchronous metadata read; serialize against lifetime changes.
   */
  [[nodiscard]] std::uint32_t atom_count() const noexcept { return atom_count_; }

 private:
  generativeqc_system* handle_ = nullptr;
  std::uint32_t atom_count_{};
};

/** Native calculation values; analytic forces are optional and in Hartree/Bohr. */
/** @native-contract generativeqc::CalculationResult
 * @behavior Self-contained result of a successful Calculation::execute.
 * @outputs energy is total energy; forces is absent unless requested, otherwise flat atom/xyz
 * -dE/dR. iterations and executed_backend report actual work. density_rms and legacy
 * reference_residual alias the native method convergence aggregate. physical_residual_rms is
 * optional separate FDS-SDF RMS; correlation is optional method-specific diagnostics.
 * @lifetime Owns force vectors and copied diagnostics; remains valid after calculation
 * destruction. Default zero initialization alone does not mean a solve succeeded.
 * @units Energy is Hartree and forces Hartree/Bohr; residual meanings remain method-specific,
 * with SCF density-update and physical measures separate.
 */
struct CalculationResult {
  double energy{};
  std::optional<std::vector<double>> forces;
  std::uint32_t iterations{};
  double reference_residual{};
  generativeqc_backend executed_backend{};
  std::optional<generativeqc_correlation_diagnostic> correlation;
  /** Native method convergence aggregate; reference_residual retains its legacy alias. */
  double density_rms{};
  /** Separate physical commutator, absent when the method does not report it. */
  std::optional<double> physical_residual_rms;
};

/** Native single-system plan; context must outlive it. Unsupported properties
 * fail before
 * execution, and absent forces are represented explicitly. */
/** @native-contract generativeqc::Calculation
 * @behavior Move-constructible RAII calculation owner; copying is deleted.
 * @lifetime Owns one native calculation; associated Context must outlive execution and
 * destruction. Moving transfers the handle; the source has a null handle and is only supported
 * for destruction. No move assignment is provided.
 * @errors Constructors/operations can throw Error for native failure and standard allocation
 * exceptions. Destruction and the move constructor are nonthrowing.
 * @execution Serialize operations and lifetime changes involving the same object/context. RAII
 * does not provide independent thread-safe execution.
 */
class Calculation {
 public:
  /** @native-contract generativeqc::Calculation::Calculation
   * @behavior Construct and own a prepared native calculation.
   * @inputs Supply a live Context, System and initialized method descriptor. C descriptor
   * initialization/admission rules apply.
   * @lifetime Copies system/method configuration and retains the context; the System may be
   * destroyed after construction.
   * @errors Throws Error on native validation/preparation failure; no live wrapper is returned.
   * Allocation failures may throw standard exceptions.
   * @execution Synchronous construction; serialize with operations on the same Context.
   * @units Uses C descriptor atomic units: Bohr geometry, Hartree energies and Hartree/Bohr
   * forces; memory limits are bytes.
   */
  Calculation(Context& context, const System& system, const generativeqc_method_descriptor& method)
      : context_(context.get()),
        atom_count_(system.atom_count()),
        capabilities_(method_capabilities(method.method)) {
    check(generativeqc_calculation_prepare(context_, system.get(), &method, &handle_));
  }
  /** @native-contract generativeqc::Calculation::~Calculation
   * @behavior Destroy the owned native calculation and release resources.
   * @lifetime Owns one native calculation; associated Context must outlive execution and
   * destruction. A moved-from object holds NULL and destruction is harmless.
   * @errors Nonthrowing destructor; calls the matching C destroy function.
   * @execution Synchronous; never race destruction with an operation or borrowed-handle use.
   */
  ~Calculation() { generativeqc_calculation_destroy(handle_); }
  /** @native-contract generativeqc::Calculation::Calculation-copy
   * @behavior Deleted copy operation: native ownership cannot be duplicated.
   * @lifetime Use move construction for exclusive ownership transfer; this declaration cannot
   * be called.
   * @errors Attempting this operation is a compile-time error, not a runtime Error.
   */
  Calculation(const Calculation&) = delete;
  /** @native-contract generativeqc::Calculation::operator=-copy
   * @behavior Deleted copy operation: native ownership cannot be duplicated.
   * @lifetime Use move construction for exclusive ownership transfer; this declaration cannot
   * be called.
   * @errors Attempting this operation is a compile-time error, not a runtime Error.
   */
  Calculation& operator=(const Calculation&) = delete;
  /** @native-contract generativeqc::Calculation::Calculation-move
   * @behavior Transfer exclusive ownership from other.
   * @inputs An rvalue Calculation; caller prevents concurrent use or destruction during
   * transfer.
   * @lifetime Destination acquires the native handle and associated metadata; other retains no
   * owning handle. Existing dependent native objects still require their context to remain
   * alive.
   * @errors noexcept; no native preparation is repeated.
   * @execution Synchronous ownership transfer; moved-from operations other than destruction are
   * outside the wrapper contract.
   */
  Calculation(Calculation&& other) noexcept
      : context_(other.context_),
        handle_(std::exchange(other.handle_, nullptr)),
        atom_count_(other.atom_count_),
        capabilities_(other.capabilities_) {}
  /** @native-contract generativeqc::Calculation::execute
   * @behavior Run a prepared method and return a value result.
   * @inputs properties defaults to ENERGY; nonzero supported flag combinations are required.
   * FORCES requests 3*N internally owned doubles; omitting it performs the C energy-only path.
   * @outputs Returns CalculationResult only on a successful solve and supported diagnostic
   * queries. Optional physical residual/correlation remain absent when no record is supplied.
   * @lifetime Returned values own their buffers. This wrapper retains the Context, and
   * execution updates its native result state.
   * @errors Throws Error on unsupported/zero properties, failed execution (including
   * NOT_CONVERGED), or failed SCF diagnostic queries except NOT_IMPLEMENTED. Correlation copy
   * failure leaves correlation absent. No partially filled result is returned; allocation
   * exceptions propagate.
   * @execution Synchronous; serialize calls involving the same owner/context, including queries
   * and destruction. No concurrent execute/query or destroy/use is supported.
   * @units Energy Hartree, forces -dE/dR Hartree/Bohr, atom-major xyz order.
   */
  CalculationResult execute(generativeqc_property_flags properties = GENERATIVEQC_PROPERTY_ENERGY) {
    CalculationResult result;
    if (!properties || (properties & ~capabilities_.supported_properties))
      throw Error(GENERATIVEQC_STATUS_NOT_IMPLEMENTED, "requested method property is unsupported");
    if (properties & GENERATIVEQC_PROPERTY_FORCES) result.forces.emplace(atom_count_ * 3);
    generativeqc_result_descriptor output{
        sizeof(generativeqc_result_descriptor),
        GENERATIVEQC_ABI_VERSION,
        0,
        result.forces ? result.forces->data() : nullptr,
        result.forces ? static_cast<std::uint32_t>(result.forces->size()) : 0,
        0,
        0,
        0,
        0,
        GENERATIVEQC_BACKEND_CPU_REFERENCE};
    const auto status = generativeqc_calculation_execute(handle_, &output);
    if (status != GENERATIVEQC_STATUS_SUCCESS)
      throw Error(status, generativeqc_context_get_last_detail(context_));
    result.energy = output.energy;
    result.iterations = output.iterations;
    result.reference_residual = output.density_rms;
    result.density_rms = output.density_rms;
    result.executed_backend = output.executed_backend;
    generativeqc_scf_diagnostic scf_diagnostic{sizeof(generativeqc_scf_diagnostic),
                                               GENERATIVEQC_ABI_VERSION, 0, 0};
    const auto scf_status = generativeqc_calculation_get_scf_diagnostic(handle_, &scf_diagnostic);
    if (scf_status != GENERATIVEQC_STATUS_NOT_IMPLEMENTED) {
      check(scf_status);
      result.physical_residual_rms = scf_diagnostic.physical_residual_rms;
    }
    generativeqc_correlation_diagnostic diagnostic{};
    diagnostic.struct_size = sizeof(diagnostic);
    diagnostic.abi_version = GENERATIVEQC_ABI_VERSION;
    if (generativeqc_calculation_get_correlation_diagnostic(handle_, &diagnostic) ==
        GENERATIVEQC_STATUS_SUCCESS)
      result.correlation = diagnostic;
    return result;
  }

 private:
  generativeqc_context* context_{};
  generativeqc_calculation* handle_{};
  std::uint32_t atom_count_{};
  MethodCapabilities capabilities_;
};

/** One input-indexed fleet result with independent success and diagnostics. */
/** @native-contract generativeqc::BatchItemResult
 * @behavior Input-indexed outcome from Batch::execute.
 * @outputs status and converged determine scientific success. energy, iterations,
 * energy_change, density_rms and executed_backend copy native scalars. bucket_id is scheduling
 * metadata; warm_start_used/warm_start_fallback describe warm attempts. physical_residual_rms
 * and correlation are optional diagnostics. forces holds flat atom/xyz values for success and
 * is cleared for failed items.
 * @lifetime Owns its vector/optional records independently of the Batch. Default fields do not
 * represent a completed execution.
 * @errors Per-item failure is represented by status; Batch::execute may separately throw a
 * whole-call/query Error.
 * @units energy and energy_change Hartree, forces -dE/dR Hartree/Bohr. density_rms is the
 * legacy method-dependent convergence aggregate; physical_residual_rms is separate.
 */
struct BatchItemResult {
  generativeqc_status status{GENERATIVEQC_STATUS_INTERNAL_ERROR};
  double energy{};
  std::vector<double> forces;
  std::uint32_t iterations{};
  double energy_change{};
  double density_rms{};
  bool converged{};
  generativeqc_backend executed_backend{GENERATIVEQC_BACKEND_CPU_REFERENCE};
  std::uint32_t bucket_id{};
  bool warm_start_used{};
  bool warm_start_fallback{};
  std::optional<double> physical_residual_rms;
  std::optional<generativeqc_correlation_diagnostic> correlation;
};

/** CUDA DF value/J/K plan evidence; peaks exclude generated-force staging. */
/** @native-contract generativeqc::DensityFittingMetricDiagnostic
 * @behavior Copied CUDA DF conditioning and allocation evidence.
 * @outputs bucket_id/system_index identify the diagnostic owner. effective_rank and
 * absolute_threshold describe retained metric modes; condition_number describes conditioning.
 * solver_device_workspace_bytes/solver_host_workspace_bytes report solver workspace;
 * device_resident_bytes/host_resident_bytes retained numeric payload;
 * peak_device_bytes/peak_host_bytes planned peaks. auxiliary_tile is the auxiliary-index tile
 * extent; streamed marks streaming.
 * @lifetime Self-contained scalar record; no device allocation or batch handle is retained.
 * @units *_bytes are bytes; indices/rank/tile are dimensionless counts. Threshold follows the
 * Coulomb-metric eigenvalue scale; condition_number is dimensionless. Peak evidence excludes
 * generated-force staging.
 */
struct DensityFittingMetricDiagnostic {
  std::uint32_t bucket_id{};
  std::uint32_t system_index{};
  std::uint64_t effective_rank{};
  double absolute_threshold{};
  double condition_number{};
  std::uint64_t solver_device_workspace_bytes{};
  std::uint64_t solver_host_workspace_bytes{};
  std::uint64_t device_resident_bytes{};
  std::uint64_t peak_device_bytes{};
  std::uint64_t host_resident_bytes{};
  std::uint64_t peak_host_bytes{};
  std::uint64_t auxiliary_tile{};
  bool streamed{};
};

/** Move-only ragged-batch owner whose Context outlives it. */
/** @native-contract generativeqc::Batch
 * @behavior Move-constructible RAII batch owner; copying is deleted.
 * @lifetime Owns one native batch; associated Context must outlive execution and destruction.
 * Moving transfers the handle; the source has a null handle and is only supported for
 * destruction. No move assignment is provided.
 * @errors Constructors/operations can throw Error for native failure and standard allocation
 * exceptions. Destruction and the move constructor are nonthrowing.
 * @execution Serialize operations and lifetime changes involving the same object/context. RAII
 * does not provide independent thread-safe execution.
 */
class Batch {
 public:
  /** @native-contract generativeqc::Batch::Batch
   * @behavior Construct and own a prepared native batch.
   * @inputs Supply a live Context, a nonempty span of nonnull System pointers, initialized
   * method descriptor and batch flags. C descriptor initialization/admission rules apply.
   * @lifetime Copies every system and method configuration; the span and source Systems may be
   * destroyed after construction. Retains the context.
   * @errors Throws Error on native failure; no live wrapper is returned. Batch also rejects a
   * null System element before preparation; allocation failures may throw standard exceptions.
   * @execution Synchronous construction; serialize with operations on the same Context.
   * @units Uses C descriptor atomic units: Bohr geometry, Hartree energies and Hartree/Bohr
   * forces; memory limits are bytes.
   */
  Batch(Context& context, std::span<const System* const> systems,
        const generativeqc_method_descriptor& method,
        generativeqc_batch_flags flags = GENERATIVEQC_BATCH_ENABLE_WARM_STARTS)
      : atom_counts_(systems.size()) {
    std::vector<const generativeqc_system*> handles(systems.size());
    for (std::size_t i = 0; i < systems.size(); ++i) {
      if (systems[i] == nullptr) {
        throw Error(GENERATIVEQC_STATUS_INVALID_ARGUMENT, "null system in batch");
      }
      handles[i] = systems[i]->get();
      atom_counts_[i] = systems[i]->atom_count();
    }
    check(generativeqc_batch_prepare(context.get(), handles.data(),
                                     static_cast<std::uint32_t>(handles.size()), &method, flags,
                                     &handle_));
  }
  /** @native-contract generativeqc::Batch::~Batch
   * @behavior Destroy the owned native batch and release resources.
   * @lifetime Owns one native batch; associated Context must outlive execution and destruction.
   * A moved-from object holds NULL and destruction is harmless.
   * @errors Nonthrowing destructor; calls the matching C destroy function.
   * @execution Synchronous; never race destruction with an operation or borrowed-handle use.
   */
  ~Batch() { generativeqc_batch_destroy(handle_); }
  /** @native-contract generativeqc::Batch::Batch-copy
   * @behavior Deleted copy operation: native ownership cannot be duplicated.
   * @lifetime Use move construction for exclusive ownership transfer; this declaration cannot
   * be called.
   * @errors Attempting this operation is a compile-time error, not a runtime Error.
   */
  Batch(const Batch&) = delete;
  /** @native-contract generativeqc::Batch::operator=-copy
   * @behavior Deleted copy operation: native ownership cannot be duplicated.
   * @lifetime Use move construction for exclusive ownership transfer; this declaration cannot
   * be called.
   * @errors Attempting this operation is a compile-time error, not a runtime Error.
   */
  Batch& operator=(const Batch&) = delete;
  /** @native-contract generativeqc::Batch::Batch-move
   * @behavior Transfer exclusive ownership from other.
   * @inputs An rvalue Batch; caller prevents concurrent use or destruction during transfer.
   * @lifetime Destination acquires the native handle and associated metadata; other retains no
   * owning handle. Existing dependent native objects still require their context to remain
   * alive.
   * @errors noexcept; no native preparation is repeated.
   * @execution Synchronous ownership transfer; moved-from operations other than destruction are
   * outside the wrapper contract.
   */
  Batch(Batch&& other) noexcept
      : handle_(std::exchange(other.handle_, nullptr)),
        atom_counts_(std::move(other.atom_counts_)) {}

  /** @native-contract generativeqc::Batch::size
   * @behavior Return the number of original input systems.
   * @outputs Returns copied atom-count-vector size as std::size_t.
   * @errors noexcept; no native query/allocation.
   * @execution Synchronous read; serialize against lifetime changes.
   */
  [[nodiscard]] std::size_t size() const noexcept { return atom_counts_.size(); }

  /** @native-contract generativeqc::Batch::execute
   * @behavior Run the fleet and return input-ordered per-item results.
   * @inputs coordinates is empty to reuse all prepared geometries, or has size() optional
   * vectors. nullopt selects that item's prepared geometry; a supplied vector has exactly 3*N
   * Bohr values in atom/xyz order.
   * @outputs Returns a vector of BatchItemResult with one status per input. The wrapper
   * requests forces for every item; failures have cleared force vectors. It also queries
   * per-item SCF and correlation records.
   * @lifetime Temporary coordinate buffers are borrowed during the synchronous call; returned
   * vectors own all copied output.
   * @errors Throws Error for wrong outer coordinate count, whole-call native failure, or
   * diagnostic-query failure other than NOT_IMPLEMENTED. Individual scientific statuses do not
   * by themselves throw. Standard allocation exceptions may propagate; no partial result vector
   * is returned on exception.
   * @execution Synchronous; serialize calls involving the same owner/context, including queries
   * and destruction. No concurrent execute/query or destroy/use is supported.
   * @units Coordinates Bohr, energies/energy_change Hartree, forces -dE/dR Hartree/Bohr.
   */
  std::vector<BatchItemResult> execute(
      const std::vector<std::optional<std::vector<double>>>& coordinates = {}) {
    if (!coordinates.empty() && coordinates.size() != size()) {
      throw Error(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                  "coordinate list does not match batch size");
    }
    std::vector<generativeqc_batch_input_descriptor> inputs;
    if (!coordinates.empty()) {
      inputs.resize(size());
      for (std::size_t i = 0; i < size(); ++i) {
        inputs[i] = {sizeof(generativeqc_batch_input_descriptor), GENERATIVEQC_ABI_VERSION,
                     coordinates[i] ? coordinates[i]->data() : nullptr,
                     coordinates[i] ? static_cast<std::uint32_t>(coordinates[i]->size()) : 0};
      }
    }

    std::vector<BatchItemResult> results(size());
    std::vector<generativeqc_batch_item_result_descriptor> native(size());
    for (std::size_t i = 0; i < size(); ++i) {
      results[i].forces.resize(static_cast<std::size_t>(atom_counts_[i]) * 3);
      native[i] = {sizeof(generativeqc_batch_item_result_descriptor),
                   GENERATIVEQC_ABI_VERSION,
                   GENERATIVEQC_STATUS_INTERNAL_ERROR,
                   0.0,
                   results[i].forces.data(),
                   static_cast<std::uint32_t>(results[i].forces.size()),
                   0,
                   0.0,
                   0.0,
                   0,
                   GENERATIVEQC_BACKEND_CPU_REFERENCE,
                   0,
                   0,
                   0};
    }
    check(generativeqc_batch_execute(handle_, inputs.empty() ? nullptr : inputs.data(),
                                     static_cast<std::uint32_t>(inputs.size()), native.data(),
                                     static_cast<std::uint32_t>(native.size())));
    for (std::size_t i = 0; i < size(); ++i) {
      results[i].status = native[i].status;
      results[i].energy = native[i].energy;
      results[i].iterations = native[i].iterations;
      results[i].energy_change = native[i].energy_change;
      results[i].density_rms = native[i].density_rms;
      results[i].converged = native[i].converged != 0;
      results[i].executed_backend = native[i].executed_backend;
      results[i].bucket_id = native[i].bucket_id;
      results[i].warm_start_used = native[i].warm_start_used != 0;
      results[i].warm_start_fallback = native[i].warm_start_fallback != 0;
      generativeqc_scf_diagnostic diagnostic{sizeof(generativeqc_scf_diagnostic),
                                             GENERATIVEQC_ABI_VERSION, 0, 0};
      const auto status = generativeqc_batch_get_scf_diagnostic(handle_, i, &diagnostic);
      if (status != GENERATIVEQC_STATUS_NOT_IMPLEMENTED) {
        check(status);
        results[i].physical_residual_rms = diagnostic.physical_residual_rms;
      }
      generativeqc_correlation_diagnostic correlation{};
      correlation.struct_size = sizeof(correlation);
      correlation.abi_version = GENERATIVEQC_ABI_VERSION;
      const auto correlation_status =
          generativeqc_batch_get_correlation_diagnostic(handle_, i, &correlation);
      if (correlation_status == GENERATIVEQC_STATUS_SUCCESS)
        results[i].correlation = correlation;
      else if (correlation_status != GENERATIVEQC_STATUS_NOT_IMPLEMENTED)
        check(correlation_status);
      if (native[i].status != GENERATIVEQC_STATUS_SUCCESS) results[i].forces.clear();
    }
    return results;
  }

  /** @native-contract generativeqc::Batch::clear_warm_starts
   * @behavior Discard all retained batch warm seeds.
   * @outputs Later execution cannot reuse cleared snapshots; exported result values are
   * unaffected.
   * @lifetime Mutates only the owned batch warm-state cache.
   * @errors Throws Error if the native clear fails.
   * @execution Synchronous; serialize calls involving the same owner/context, including queries
   * and destruction. No concurrent execute/query or destroy/use is supported.
   */
  void clear_warm_starts() { check(generativeqc_batch_clear_warm_starts(handle_)); }

  /** @native-contract generativeqc::Batch::set_warm_start_updates
   * @behavior Freeze or resume replacement of retained warm seeds.
   * @inputs enabled=false freezes current seeds; true advances them after successful execution.
   * @outputs Does not create or discard existing snapshots.
   * @lifetime Setting persists on this batch until changed or destroyed.
   * @errors Throws Error if the native setting call fails.
   * @execution Synchronous; serialize calls involving the same owner/context, including queries
   * and destruction. No concurrent execute/query or destroy/use is supported.
   */
  void set_warm_start_updates(bool enabled) {
    check(generativeqc_batch_set_warm_start_updates(handle_, enabled ? 1 : 0));
  }

  /** Return CUDA DF metric/allocation records from the most recent execution. */
  /** @native-contract generativeqc::Batch::last_density_fitting_metric_diagnostics
   * @behavior Return copied latest CUDA DF metric/allocation records.
   * @outputs Queries count, allocates that many rows, copies them and returns
   * DensityFittingMetricDiagnostic values. When no record exists the native query returns
   * NOT_IMPLEMENTED, which this wrapper throws as Error.
   * @lifetime Returned values own their scalars; caller must serialize both count and copy
   * against execution/destruction.
   * @errors Throws Error on either native query failure or count changing between queries;
   * allocation exceptions propagate.
   * @execution Synchronous; serialize calls involving the same owner/context, including queries
   * and destruction. No concurrent execute/query or destroy/use is supported.
   * @units Byte/rank/threshold/conditioning semantics are those of
   * DensityFittingMetricDiagnostic; peaks exclude generated-force staging.
   */
  std::vector<DensityFittingMetricDiagnostic> last_density_fitting_metric_diagnostics() const {
    std::uint32_t count = 0;
    check(generativeqc_batch_get_last_density_fitting_metric_diagnostics(handle_, nullptr, 0,
                                                                         &count));
    std::vector<generativeqc_density_fitting_metric_diagnostic> native(count);
    std::uint32_t written = 0;
    check(generativeqc_batch_get_last_density_fitting_metric_diagnostics(handle_, native.data(),
                                                                         count, &written));
    if (written != count) {
      throw Error(GENERATIVEQC_STATUS_INTERNAL_ERROR,
                  "CUDA DF metric diagnostic count changed during copy");
    }
    std::vector<DensityFittingMetricDiagnostic> result;
    result.reserve(count);
    for (const auto& input : native) {
      result.push_back({input.bucket_id, input.system_index, input.effective_rank,
                        input.absolute_threshold, input.condition_number,
                        input.solver_device_workspace_bytes, input.solver_host_workspace_bytes,
                        input.device_resident_bytes, input.peak_device_bytes,
                        input.host_resident_bytes, input.peak_host_bytes, input.auxiliary_tile,
                        input.streamed != 0});
    }
    return result;
  }

 private:
  generativeqc_batch* handle_ = nullptr;
  std::vector<std::uint32_t> atom_counts_;
};

}  // namespace generativeqc

#endif
