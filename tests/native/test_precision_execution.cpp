#include <array>
#include <cstdint>
#include <cstdio>
#include <new>
#include <stdexcept>

#include "generativeqc/precision_admission.h"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct State {
  std::array<std::uint32_t, 6> events{};
  std::size_t count{};
  bool retry{}, qualified{}, audit{}, verification{};
  int throws{-1};
};

int operation(void* opaque, std::uint32_t event, std::uint32_t* decision) {
  auto& state = *static_cast<State*>(opaque);
  require(state.count < state.events.size(), "unbounded precision operations");
  state.events[state.count++] = event;
  if (event == static_cast<std::uint32_t>(state.throws)) throw std::bad_alloc();
  *decision = 0;
  switch (event) {
    case GENERATIVEQC_PRECISION_EXECUTE_CANDIDATE:
      if (state.retry) return GENERATIVEQC_PRECISION_CALLBACK_RETRY;
      break;
    case GENERATIVEQC_PRECISION_CLASSIFY_CANDIDATE:
      *decision = state.qualified;
      break;
    case GENERATIVEQC_PRECISION_EXECUTE_AUDIT:
      *decision = state.audit;
      break;
    case GENERATIVEQC_PRECISION_VERIFY_FALLBACK:
      *decision = state.verification;
      break;
  }
  return GENERATIVEQC_PRECISION_CALLBACK_SUCCESS;
}

void execute_contract() {
  generativeqc_precision_execution_context_v1 context{sizeof(context), GENERATIVEQC_ABI_VERSION, 1,
                                                      0};
  for (const bool retry : {false, true}) {
    for (const bool qualified : {false, true}) {
      for (const bool audit : {false, true}) {
        for (const bool verification : {false, true}) {
          State state{};
          state.retry = retry;
          state.qualified = qualified;
          state.audit = audit;
          state.verification = verification;
          generativeqc_precision_execution_receipt_v1 output{sizeof(output),
                                                             GENERATIVEQC_ABI_VERSION};
          require(generativeqc_precision_execution_run_v1(&context, operation, &state, &output) ==
                      GENERATIVEQC_STATUS_SUCCESS,
                  "valid precision schedule failed");
          require(state.events[0] == GENERATIVEQC_PRECISION_EXECUTE_CANDIDATE,
                  "candidate was not first");
          if (retry)
            require(state.events[1] == GENERATIVEQC_PRECISION_EXECUTE_FP64_FALLBACK,
                    "retry did not preserve its explicit FP64 role");
          const auto expected = qualified      ? GENERATIVEQC_PRECISION_EXECUTION_QUALIFIED
                                : audit        ? GENERATIVEQC_PRECISION_EXECUTION_AUDITED
                                : verification ? GENERATIVEQC_PRECISION_EXECUTION_VERIFIED
                                               : GENERATIVEQC_PRECISION_EXECUTION_REJECTED;
          require(output.outcome == static_cast<std::uint32_t>(expected),
                  "scientific audit disposition changed");
          require(state.count == (qualified ? 2u
                                  : audit   ? 4u
                                            : 5u) +
                                     retry,
                  "audit/verification work was repeated or skipped");
          if (!qualified)
            require(state.events[2 + retry] == GENERATIVEQC_PRECISION_RELEASE_CANDIDATE &&
                        state.events[3 + retry] == GENERATIVEQC_PRECISION_EXECUTE_AUDIT,
                    "audit preceded candidate release");
        }
      }
    }
  }
  for (const auto event :
       {GENERATIVEQC_PRECISION_EXECUTE_CANDIDATE, GENERATIVEQC_PRECISION_CLASSIFY_CANDIDATE,
        GENERATIVEQC_PRECISION_RELEASE_CANDIDATE, GENERATIVEQC_PRECISION_EXECUTE_AUDIT,
        GENERATIVEQC_PRECISION_VERIFY_FALLBACK}) {
    State state{};
    state.throws = event;
    generativeqc_precision_execution_receipt_v1 output{sizeof(output), GENERATIVEQC_ABI_VERSION};
    require(generativeqc_precision_execution_run_v1(&context, operation, &state, &output) ==
                GENERATIVEQC_STATUS_INTERNAL_ERROR,
            "C++ callback exception escaped or became retry eligibility");
    require(output.outcome == GENERATIVEQC_PRECISION_EXECUTION_FAILED &&
                state.events[state.count - 1] == static_cast<std::uint32_t>(event),
            "scientific work continued after callback exception");
  }
}

}  // namespace

int main() {
  try {
    execute_contract();
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
