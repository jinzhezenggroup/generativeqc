// A shared observation/operation bridge only: contains no Johnson algebra.
// Included after either the frozen legacy fixture or the live CPU fixture.
namespace {
int normalized_result(int result) {
  if (result == OWNER_NUMERICAL_FAILURE) return 6;
  if (result == OWNER_ALLOCATION_FAILED) return 4;
  return result;
}
}  // namespace
extern "C" std::size_t consumer_state_size(void* handle) {
  return static_cast<Fixture*>(handle)->plan.state_size_bytes();
}
extern "C" void consumer_copy_state(void* handle, unsigned char* output) {
  const auto& f = *static_cast<Fixture*>(handle);
  std::memcpy(output, f.state.workspace_base, f.plan.state_size_bytes());
}
extern "C" void consumer_copy_layout(void* handle, std::uint64_t* output) {
  const auto& f = *static_cast<Fixture*>(handle);
  *output++ = f.plan.state_size_bytes();
  *output++ = f.plan.workspace_size_bytes();
  auto state_offset = [&](const auto* pointer) {
    *output++ =
        reinterpret_cast<const char*>(pointer) - static_cast<const char*>(f.state.workspace_base);
  };
  state_offset(f.state.current_inputs);
  state_offset(f.state.previous_inputs);
  state_offset(f.state.previous_residuals);
  state_offset(f.state.df_history);
  state_offset(f.state.u_history);
  state_offset(f.state.omega);
  state_offset(f.state.residual_rms);
  state_offset(f.state.residual_maximum);
  state_offset(f.state.iterations);
  state_offset(f.state.restart_counts);
  state_offset(f.state.system_statuses);
  state_offset(f.state.initialized);
  state_offset(f.state.converged);
  auto scratch_offset = [&](const auto* pointer) {
    *output++ =
        reinterpret_cast<const char*>(pointer) - static_cast<const char*>(f.scratch.workspace_base);
  };
  scratch_offset(f.scratch.residual);
  scratch_offset(f.scratch.mixed);
  scratch_offset(f.scratch.delta_f);
  scratch_offset(f.scratch.new_u);
  scratch_offset(f.scratch.beta);
  scratch_offset(f.scratch.coefficients);
  scratch_offset(f.scratch.history_slots);
}
extern "C" int consumer_batch(void* handle, const double* raw) {
  auto& f = *static_cast<Fixture*>(handle);
  f.put_raw(raw);
  return normalized_result(
      static_cast<int>(OWNER_BATCH(f.plan, f.vector, f.state, f.scratch, f.error)));
}
extern "C" int consumer_restart(void* handle, int system, const double* raw) {
  auto& f = *static_cast<Fixture*>(handle);
  f.put_raw(raw);
  return normalized_result(
      static_cast<int>(OWNER_RESTART(f.plan, system, f.vector, f.state, f.error)));
}
extern "C" int consumer_initialize(void* handle, const double* raw) {
  auto& f = *static_cast<Fixture*>(handle);
  f.put_raw(raw);
  return normalized_result(static_cast<int>(OWNER_INITIALIZE(f.plan, f.vector, f.state, f.error)));
}
extern "C" void consumer_set_counter(void* handle, int system, int restart, std::uint64_t value) {
  auto& f = *static_cast<Fixture*>(handle);
  (restart ? f.state.restart_counts : f.state.iterations)[system] = value;
}
extern "C" int consumer_reject_binding(void* handle, int kind) {
  auto& f = *static_cast<Fixture*>(handle);
  auto state = f.state;
  auto scratch = f.scratch;
  auto vector = f.vector;
  if (kind == 0) ++state.current_inputs;
  if (kind == 1) --state.workspace_size_bytes;
  if (kind == 2) ++scratch.coefficients;
  if (kind == 3) --scratch.workspace_size_bytes;
  if (kind == 4) ++vector.fields[1];
  if (kind == 5) --vector.workspace_size_bytes;
  return normalized_result(static_cast<int>(OWNER_MIX(f.plan, 1, vector, state, scratch, f.error)));
}
