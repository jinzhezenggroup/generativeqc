"""Executable host contracts for the *production* DF retry/group policy.

No CUDA compilation, source slicing, method-body copying or source-spelling
assertions: C++ calls the same private header used by df_ccsdt_force.cu.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

_HOST_CASES = r"""
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <vector>

#include "methods/df_ccsdt_retry_policy.hpp"

namespace policy = generativeqc::methods::detail::df_retry;

struct MethodError : std::runtime_error {
  int status_code;
  explicit MethodError(int status) : std::runtime_error("method error"), status_code(status) {}
  int status() const { return status_code; }
};
constexpr int out_of_memory = 7;

struct Counters {
  int attempted{}, late{}, owners_destroyed{}, live{}, clears{}, prepared{}, finished{};
};
struct Owner {
  Counters& c;
  explicit Owner(Counters& counters) : c(counters) { ++c.live; }
  ~Owner() { --c.live; ++c.owners_destroyed; }
};
struct Cache {
  Counters& c;
  std::size_t retained{};
  std::size_t storage_bytes() const { return retained; }
  void clear() {
    assert(c.live == 0 && c.owners_destroyed == c.attempted);
    retained = 0;
    ++c.clears;
  }
};
struct Diagnostic {
  const std::vector<double>* initial_density{};
  int reference{}, work{};
  bool seed_fallback{};
};
struct Guess {
  std::vector<double> density;
  double seconds{};
};
struct Result {
  bool recycling_discarded_primal_attempt{};
  int semantic_work{13};
  int work_this_attempt{};
  double total_seconds{}, reference_seconds{};
  Guess reference_guess;
};

enum class Failure { none, length, allocation, method_oom, method_physical, physical, republished };
void fail(Failure kind, Cache& cache) {
  switch (kind) {
    case Failure::none: return;
    case Failure::length: throw std::length_error("numeric budget");
    case Failure::allocation: throw std::bad_alloc();
    case Failure::method_oom: throw MethodError(out_of_memory);
    case Failure::method_physical: throw MethodError(2);
    case Failure::physical: throw std::runtime_error("physical failure");
    case Failure::republished:
      cache.retained = 10;
      throw std::bad_alloc();
  }
}

std::size_t checked_add(std::size_t first, std::size_t second) {
  constexpr auto limit = static_cast<std::size_t>(INT64_MAX);
  if (first > limit || second > limit - first)
    throw std::overflow_error("post-HF byte count overflow");
  return first + second;
}

struct Scenario {
  Counters c;
  Cache cache{c, 40};
  Diagnostic diagnostic;
  std::vector<double> borrowed{4.0};
  bool prepare_density{true};
  std::size_t budget{170};
  std::size_t extra_output{};
  Failure first{Failure::none}, second{Failure::none};
  bool no_cache{};

  Result run() {
    c.prepared++;
    Guess guess{prepare_density ? std::vector<double>{1.0} : std::vector<double>{},
                prepare_density ? 0.125 : 0.0};
    diagnostic.initial_density = &borrowed;
    policy::RestoreSeed restored(diagnostic);
    if (!guess.density.empty()) diagnostic.initial_density = &guess.density;
    Cache* cache_ptr = no_cache ? nullptr : &cache;
    const bool had_cache = cache_ptr && cache_ptr->storage_bytes();
    const auto attempt = [&]() -> Result {
      diagnostic.reference = 0;
      diagnostic.work = 0;
      diagnostic.seed_fallback = false;
      ++c.attempted;
      Owner owner(c);
      const auto reserved = policy::attempt_reservation(
          guess.density.capacity() * sizeof(double), cache_ptr ? cache_ptr->storage_bytes() : 0,
          extra_output, checked_add);
      if (reserved > budget || budget - reserved < 50) throw std::length_error("primal budget");
      ++c.late;
      fail(c.attempted == 1 ? first : second, cache);
      if (budget - reserved < 80) throw std::length_error("late phase");
      diagnostic.reference = 5;
      diagnostic.work = 17;
      Result result;
      result.work_this_attempt = 17;
      return result;
    };
    const auto finish = [&](Result result) {
      ++c.finished;
      result.total_seconds = 1.0;
      result.reference_seconds += guess.seconds;
      guess.density = std::vector<double>{};
      result.reference_guess = std::move(guess);
      return result;
    };
    return policy::retry_retained_cache_once<MethodError>(
        cache_ptr, had_cache, out_of_memory, attempt, finish);
  }
};

void successful_and_late_failure_cases() {
  for (bool fitted : {false, true})
    for (std::size_t retained : {0U, 10U, 40U, 60U})
      for (int failure = 0; failure <= 6; ++failure) {
        Scenario s;
        s.prepare_density = !fitted;
        s.cache.retained = retained;
        s.first = static_cast<Failure>(failure);
        const bool resource = s.first == Failure::length || s.first == Failure::allocation ||
                              s.first == Failure::method_oom || s.first == Failure::republished;
        const bool physical = s.first == Failure::physical || s.first == Failure::method_physical;
        bool threw = false;
        try {
          const auto result = s.run();
          assert(!physical);
          assert(s.c.finished == 1 && s.c.prepared == 1);
          assert(result.reference_guess.density.capacity() == 0);
          assert(result.reference_seconds == (fitted ? 0.0 : 0.125));
          assert(result.semantic_work == 13 && result.work_this_attempt == 17);
          assert(result.total_seconds >= 0.0);
          assert(result.recycling_discarded_primal_attempt == (resource && retained != 0));
        } catch (const std::exception&) {
          threw = true;
        }
        if (physical || (resource && retained == 0)) assert(threw);
        else assert(!threw);
        const int expected = resource && retained != 0 ? 2 : 1;
        assert(s.c.attempted == expected && s.c.owners_destroyed == expected);
        assert(s.c.live == 0 && s.c.prepared == 1);
        assert(s.c.clears == (expected == 2 ? 1 : 0));
        assert(s.cache.retained == (expected == 2 ? 0U : retained) ||
               (s.first == Failure::republished && expected == 1 && s.cache.retained == 10));
        assert(s.diagnostic.initial_density == &s.borrowed);
      }
}

void retry_exhaustion_and_budget_cases() {
  for (auto failure : {Failure::length, Failure::allocation, Failure::method_oom}) {
    Scenario s;
    s.first = failure;
    s.second = failure;
    bool threw = false;
    try { (void)s.run(); } catch (const std::exception&) { threw = true; }
    assert(threw && s.c.attempted == 2 && s.c.owners_destroyed == 2 && s.c.clears == 1);
    assert(s.c.finished == 0 && s.c.live == 0 && s.diagnostic.initial_density == &s.borrowed);
  }
  {
    Scenario s;
    s.cache.retained = 60;
    s.first = Failure::republished;
    bool threw = false;
    try { (void)s.run(); } catch (const std::exception&) { threw = true; }
    // After the original OOM, ten bytes remain, so one retry is legal.
    assert(!threw && s.c.attempted == 2 && s.c.clears == 1);
  }
  {
    Scenario s;
    s.cache.retained = 60;
    s.first = Failure::length;
    s.second = Failure::republished;
    bool threw = false;
    try { (void)s.run(); } catch (const std::bad_alloc&) { threw = true; }
    assert(threw && s.c.attempted == 2 && s.c.clears == 1 && s.cache.retained == 10);
  }
  {
    Scenario s;
    s.cache.retained = 10;
    s.budget = 60;
    bool threw = false;
    try { (void)s.run(); } catch (const std::length_error&) { threw = true; }
    assert(threw && s.c.attempted == 2 && s.c.clears == 1 && s.c.late == 1);
  }
  {
    Scenario s;
    s.no_cache = true;
    s.first = Failure::allocation;
    bool threw = false;
    try { (void)s.run(); } catch (const std::bad_alloc&) { threw = true; }
    assert(threw && s.c.attempted == 1 && s.c.clears == 0);
  }
  {
    // The direct physical-replay reservation is not part of the public
    // retained-cache retry wrapper; it must reject its own insufficient cap.
    const auto within = policy::attempt_reservation(0, 10, 20, checked_add);
    const auto refused = policy::attempt_reservation(0, 10, 21, checked_add);
    assert(110 - within == 80 && 110 - refused == 79);
  }
  {
    Scenario s;
    s.extra_output = static_cast<std::size_t>(INT64_MAX);
    bool overflow = false;
    try { (void)s.run(); } catch (const std::overflow_error&) { overflow = true; }
    assert(overflow && s.c.attempted == 1 && s.c.late == 0 && s.c.clears == 0);
  }
}

enum class Preconvergence { Legacy, Mixed, Automatic, Strict };
struct Convergence {
  bool relaxed_value{};
  double residual_value{1e-9};
  bool relaxed() const { return relaxed_value; }
  double cc_residual() const { return residual_value; }
};

void group_preconvergence_cases() {
  for (bool forces : {false, true})
    for (bool mp2 : {false, true})
      for (int electrons : {0, 16, 18, 20})
        for (std::size_t nao : {0U, 40U, 65U})
          for (auto pre : {Preconvergence::Legacy, Preconvergence::Mixed,
                           Preconvergence::Automatic, Preconvergence::Strict})
            for (bool relaxed : {false, true})
              for (auto companion : {0U, 1U, 2U, 3U}) {
                if (mp2 && (companion & 2U)) continue;
                Convergence convergence{relaxed, 3e-11};
                auto automatic = [](std::size_t occupied, std::size_t virtuals) {
                  return occupied >= 8 && virtuals >= 35;
                };
                const auto policy_result = policy::select_group_schedule(
                    forces, mp2, nao, electrons, pre, convergence, 1e-8, automatic);
                const auto o = electrons > 0 ? static_cast<std::size_t>(electrons / 2) : 0;
                const auto v = nao > o ? nao - o : 0;
                const bool accepted = !mp2 &&
                    (pre == Preconvergence::Mixed ||
                     (pre == Preconvergence::Legacy && relaxed) ||
                     (pre == Preconvergence::Automatic &&
                      (relaxed || (forces ? o >= 9 && v >= 48 : automatic(o, v)))));
                assert(policy_result.mixed_ccsd == accepted);
                assert(policy_result.mixed_lambda == (forces && accepted));
                assert(policy_result.lambda_batch_limit == (forces && accepted ? 64U : 32U));
                assert(policy_result.mixed_switch_residual == 1e-8);
              }
}

int main() {
  successful_and_late_failure_cases();
  retry_exhaustion_and_budget_cases();
  group_preconvergence_cases();
}
"""


def test_native_df_cache_retry_policy(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler required")
    source = tmp_path / "df_cache_retry_policy.cpp"
    source.write_text(_HOST_CASES)
    binary = tmp_path / "df_cache_retry_policy"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O0",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I" + str(ROOT / "src"),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True, timeout=15
    )
