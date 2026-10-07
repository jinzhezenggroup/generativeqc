// Bounded synthetic dense Problem probe; excludes SCF, MO construction and JIT.
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cc/solver.hpp"
#include "generated_rccsd_cpu.hpp"

using namespace generativeqc::cc;

static double pair_factor(std::size_t p, std::size_t q) {
  return std::cos(double(p + q + 1)) / (1 + p + q);
}

static void fill_block(std::vector<double>& out, std::array<std::size_t, 4> shape,
                       std::array<std::size_t, 4> start) {
  for (std::size_t i = 0; i < shape[0]; ++i)
    for (std::size_t j = 0; j < shape[1]; ++j)
      for (std::size_t k = 0; k < shape[2]; ++k)
        for (std::size_t l = 0; l < shape[3]; ++l)
          out.push_back(.035 * pair_factor(i + start[0], j + start[1]) *
                        pair_factor(k + start[2], l + start[3]));
}

static Problem problem(std::size_t o, std::size_t v) {
  Problem p;
  p.nocc = o;
  p.nvir = v;
  std::vector<double> eps;
  for (std::size_t i = 0; i < o; ++i) eps.push_back(-1.2 + .03 * i);
  for (std::size_t a = 0; a < v; ++a) eps.push_back(.6 + .02 * a);
  const auto fock = [&](std::size_t i, std::size_t j) {
    return i == j ? eps[i] : .0008 * pair_factor(i, j);
  };
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = 0; j < o; ++j) p.foo.push_back(fock(i, j));
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t a = 0; a < v; ++a) {
      p.fov.push_back(fock(i, o + a));
      p.d1.push_back(eps[i] - eps[o + a]);
    }
  for (std::size_t a = 0; a < v; ++a)
    for (std::size_t c = 0; c < v; ++c) p.fvv.push_back(fock(o + a, o + c));
  fill_block(p.ovov, {o, v, o, v}, {0, o, 0, o});
  fill_block(p.ovvo, {o, v, v, o}, {0, o, o, 0});
  fill_block(p.oovv, {o, o, v, v}, {0, 0, o, o});
  fill_block(p.ovvv, {o, v, v, v}, {0, o, o, o});
  fill_block(p.ovoo, {o, v, o, o}, {0, o, 0, 0});
  fill_block(p.oooo, {o, o, o, o}, {0, 0, 0, 0});
  fill_block(p.vvvv, {v, v, v, v}, {o, o, o, o});
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t c = 0; c < v; ++c)
          p.d2.push_back((eps[i] - eps[o + a]) + (eps[j] - eps[o + c]));
  p.initial_t1.resize(o * v);
  p.initial_t2.resize(o * o * v * v);
  return p;
}

static std::size_t budget(const Problem& p, bool reuse, std::size_t history) {
  const auto n = p.initial_t1.size() + p.initial_t2.size();
  const auto h = history + 1;
  const auto scratch = history ? history * history + 2 * h * h + 2 * h : 0;
  const auto iteration = reuse ? generated::iteration_reuse_arena_elements(p.nocc, p.nvir)
                               : generated::iteration_arena_elements(p.nocc, p.nvir);
  return problem_host_bytes(p) +
         sizeof(double) * ((4 + 2 * history) * n + scratch + iteration +
                           generated::replay_arena_elements(p.nocc, p.nvir));
}

static void verify(const SolverResult& a, const SolverResult& b) {
  if (!a.converged() || !b.converged() || a.correlation_energy != b.correlation_energy ||
      a.t1 != b.t1 || a.t2 != b.t2 || a.diagnostic.iterations != b.diagnostic.iterations ||
      a.diagnostic.iteration_graph_calls != b.diagnostic.iteration_graph_calls ||
      a.diagnostic.replay_graph_calls != b.diagnostic.replay_graph_calls ||
      a.diagnostic.iteration_dynamic_operations != b.diagnostic.iteration_dynamic_operations)
    throw std::runtime_error("complete endpoint parity failed");
}

static generated::Inputs bind(const Problem& p, const std::vector<double>& t1,
                              const std::vector<double>& t2) {
  return {p.foo.data(),  p.fov.data(),  p.fvv.data(),  p.ovov.data(), p.ovvo.data(),
          p.oovv.data(), p.ovvv.data(), p.ovoo.data(), p.oooo.data(), p.vvvv.data(),
          p.d1.data(),   p.d2.data(),   t1.data(),     t2.data()};
}

struct Values {
  double energy;
  std::vector<double> r1, r2, next1, next2;
};

static void fixed_work_control(std::size_t repeats, std::size_t o, std::size_t v) {
  if (!repeats || repeats > 30 || !o || !v || o > 12 || v > 32)
    throw std::invalid_argument("fixed-work control exceeds bounded probe");
  constexpr std::size_t evaluations = 9;
  const auto p = problem(o, v);
  const auto n1 = o * v, n2 = o * o * v * v;
  std::vector<double> full(generated::iteration_arena_elements(o, v));
  // Reprepare and reuse share the identical pinned buffer addresses.
  std::vector<double> pinned(generated::iteration_reuse_arena_elements(o, v));
  std::array<std::vector<double>, evaluations> t1, t2;
  std::vector<Values> expected;
  for (std::size_t step = 0; step < evaluations; ++step) {
    t1[step].resize(n1);
    t2[step].resize(n2);
    for (std::size_t i = 0; i < n1; ++i) t1[step][i] = .0001 * std::sin(double(i + step));
    for (std::size_t i = 0; i < n2; ++i) t2[step][i] = .0001 * std::cos(double(i + step));
    const auto in = bind(p, t1[step], t2[step]);
    const auto out = generated::run_iteration_cpu(o, v, in, full.data(), full.size());
    expected.push_back({out.energy,
                        {out.r1, out.r1 + n1},
                        {out.r2, out.r2 + n2},
                        {out.next_t1, out.next_t1 + n1},
                        {out.next_t2, out.next_t2 + n2}});
  }
  std::cout << std::setprecision(17) << "{\"kind\":\"fixed-work-control\",\"o\":" << o
            << ",\"v\":" << v << ",\"evaluations\":" << evaluations << ",\"triples\":[";
  for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
    double elapsed[3]{};
    for (unsigned ordinal = 0; ordinal < 3; ++ordinal) {
      const unsigned mode = (ordinal + repeat) % 3;
      for (std::size_t step = 0; step < evaluations; ++step) {
        const auto in = bind(p, t1[step], t2[step]);
        const auto started = std::chrono::steady_clock::now();
        if (mode == 1 || (mode == 2 && step == 0))
          generated::run_iteration_reuse_prepare_cpu(o, v, in, pinned.data(), pinned.size());
        const auto out =
            mode == 0 ? generated::run_iteration_cpu(o, v, in, full.data(), full.size())
                      : generated::run_iteration_reused_cpu(o, v, in, pinned.data(), pinned.size());
        elapsed[mode] +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        const auto& ref = expected[step];
        if (out.energy != ref.energy || std::memcmp(out.r1, ref.r1.data(), n1 * sizeof(double)) ||
            std::memcmp(out.r2, ref.r2.data(), n2 * sizeof(double)) ||
            std::memcmp(out.next_t1, ref.next1.data(), n1 * sizeof(double)) ||
            std::memcmp(out.next_t2, ref.next2.data(), n2 * sizeof(double)))
          throw std::runtime_error("fixed-work output parity failed");
      }
    }
    std::cout << (repeat ? "," : "") << "{\"original_full\":" << elapsed[0]
              << ",\"pinned_reprepare\":" << elapsed[1] << ",\"pinned_reuse\":" << elapsed[2]
              << "}";
  }
  std::cout << "]}\n";
}

int main(int argc, char** argv) {
  if (argc == 5 && std::string(argv[1]) == "--fixed-work-control") {
    fixed_work_control(std::stoul(argv[2]), std::stoul(argv[3]), std::stoul(argv[4]));
    return 0;
  }
  if (argc < 5 || (argc - 3) % 2)
    throw std::invalid_argument("expected repeats, DIIS size, and o/v pairs");
  const auto repeats = std::stoul(argv[1]), history = std::stoul(argv[2]);
  if (!repeats || repeats > 30 || history == 1 || history > 20)
    throw std::invalid_argument("invalid bounded repeat/history count");
  std::vector<std::pair<std::size_t, std::size_t>> shapes;
  for (int i = 3; i < argc; i += 2)
    shapes.emplace_back(std::stoul(argv[i]), std::stoul(argv[i + 1]));
  std::cout << std::setprecision(17);
  for (auto [o, v] : shapes) {
    if (!o || !v || o > 12 || v > 32) throw std::invalid_argument("shape exceeds bounded probe");
    const auto p = problem(o, v);
    SolverOptions off, on;
    off.diis_size = on.diis_size = history;
    off.iteration_invariant_reuse = false;
    on.iteration_invariant_reuse = true;
    off.max_iterations = on.max_iterations = 100;
    off.max_bytes = on.max_bytes = budget(p, true, history);
    const auto baseline = solve_cpu(p, off), cached = solve_cpu(p, on);
    verify(baseline, cached);
    const auto& d = cached.diagnostic;
    if (baseline.diagnostic.iteration_reuse || !d.iteration_reuse ||
        baseline.diagnostic.iteration_invariant_operations - d.iteration_invariant_operations !=
            d.iteration_invariant_operations_saved)
      throw std::runtime_error("admission/work did not select requested schedule");
    std::cout << "{\"o\":" << o << ",\"v\":" << v << ",\"diis_size\":" << history
              << ",\"baseline_bytes\":" << baseline.diagnostic.numeric_capacity_bytes
              << ",\"reuse_bytes\":" << d.numeric_capacity_bytes
              << ",\"iterations\":" << d.iterations
              << ",\"iteration_calls\":" << d.iteration_graph_calls
              << ",\"replay_calls\":" << d.replay_graph_calls
              << ",\"saved_operations\":" << d.iteration_invariant_operations_saved
              << ",\"baseline_invariant_operations\":"
              << baseline.diagnostic.iteration_invariant_operations
              << ",\"reuse_invariant_operations\":" << d.iteration_invariant_operations
              << ",\"dynamic_operations\":" << d.iteration_dynamic_operations << ",\"pairs\":[";
    for (unsigned repeat = 0; repeat < repeats; ++repeat) {
      double elapsed[2], tensor[2];
      for (unsigned j = 0; j < 2; ++j) {
        const unsigned k = (j + repeat) % 2;
        const auto started = std::chrono::steady_clock::now();
        const auto result = solve_cpu(p, k ? on : off);
        elapsed[k] =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        tensor[k] = result.diagnostic.tensor_seconds;
        verify(baseline, result);
      }
      std::cout << (repeat ? "," : "") << "{\"baseline\":" << elapsed[0]
                << ",\"reuse\":" << elapsed[1] << ",\"baseline_tensor\":" << tensor[0]
                << ",\"reuse_tensor\":" << tensor[1] << "}";
    }
    std::cout << "]}\n";
  }
}
