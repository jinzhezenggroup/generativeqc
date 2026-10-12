#include "cli/geometry_command.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "cli/native_basis.hpp"
#include "cli_method_catalog.hpp"
#include "generativeqc/generativeqc.hpp"
#include "generativeqc/geometry_optimizer.hpp"
#include "generativeqc/ks.hpp"
#include "methods/generated_method_manifest.hpp"

namespace generativeqc::cli {
namespace {

struct Stage {
  RunOptions inputs;
  geometry::Options optimizer;
  bool auto_preopt{};
};

struct StageRecord {
  std::string method, basis, status, detail;
  generativeqc_backend backend{};
  geometry::Result outcome;
};

std::string json_escape(std::string_view s) {
  std::string out;
  for (unsigned char c : s) {
    if (c == '\\' || c == '"') {
      out.push_back('\\');
      out.push_back(static_cast<char>(c));
    } else if (c < 0x20) {
      constexpr char digits[] = "0123456789abcdef";
      out += "\\u00";
      out += digits[c >> 4];
      out += digits[c & 15];
    } else {
      out.push_back(static_cast<char>(c));
    }
  }
  return out;
}

std::string_view backend_name(generativeqc_backend backend) {
  return backend == GENERATIVEQC_BACKEND_CUDA ? "cuda" : "cpu";
}

bool is_xtb(const RunOptions& o) { return o.method == GENERATIVEQC_METHOD_GFN2_XTB; }

bool is_correlated(const RunOptions& o) {
  const auto* entry = methods::generated::find_method(o.method);
  return entry && (entry->family == GENERATIVEQC_METHOD_FAMILY_COUPLED_CLUSTER ||
                   entry->family == GENERATIVEQC_METHOD_FAMILY_PERTURBATION);
}

void select_stage_method(RunOptions& input, std::string name) {
  std::transform(name.begin(), name.end(), name.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (name == "ccsd(t)") name = "rccsd(t)";
  if (name == "df-ccsd(t)") name = "df-rccsd(t)";
  input.composition = nullptr;
  if (name == "gfn2-xtb" || name == "gfn2") {
    input.method_name = "gfn2-xtb";
    input.method = GENERATIVEQC_METHOD_GFN2_XTB;
  } else if (name == "rhf") {
    input.method_name = "rhf";
    input.method = GENERATIVEQC_METHOD_RHF;
  } else if (name == "uhf") {
    input.method_name = "uhf";
    input.method = GENERATIVEQC_METHOD_UHF;
  } else if (const auto* row = methods::generated::find_method(name); row != nullptr) {
    input.method_name = std::string(row->name);
    input.method = row->method;
  } else if (const auto* row = method_generated::find_method(name); row != nullptr) {
    input.method_name = std::string(row->name);
    input.method = row->carrier;
    input.composition = row;
  } else {
    throw UsageError("unknown optimization stage method: " + name);
  }
}

std::vector<std::string> stage_names(const RunOptions& target) {
  if (!target.optimization_stages.empty()) {
    std::vector<std::string> names;
    std::istringstream input(target.optimization_stages);
    std::string value;
    while (std::getline(input, value, ',')) {
      if (value.empty() || value.find_first_of(" \t\r\n") != std::string::npos)
        throw UsageError("--stages requires comma-separated native method names without spaces");
      names.push_back(std::move(value));
    }
    if (names.empty() || target.optimization_stages.back() == ',')
      throw UsageError("--stages must end with the requested final method");
    return names;
  }
  if (is_xtb(target)) return {target.method_name};
  std::vector<std::string> names{"gfn2-xtb"};
  // Deterministic spin-aware low-cost preoptimization; never change the
  // requested final functional, Hamiltonian, backend or basis.
  const bool simple = target.method_name == "rhf" || target.method_name == "uhf" ||
                      target.method_name == "lda-rks" || target.method_name == "lda-uks" ||
                      target.method_name == "pbe-rks" || target.method_name == "pbe-uks";
  if (!simple) names.push_back(target.multiplicity == 1 ? "pbe-rks" : "pbe-uks");
  names.push_back(target.method_name);
  return names;
}

std::vector<Stage> build_stages(const RunOptions& target) {
  const auto names = stage_names(target);
  std::vector<Stage> stages;
  stages.reserve(names.size());
  for (std::size_t i = 0; i < names.size(); ++i) {
    Stage stage;
    const bool last = i + 1 == names.size();
    stage.inputs = target;
    select_stage_method(stage.inputs, names[i]);
    if (last &&
        (stage.inputs.method != target.method || stage.inputs.composition != target.composition))
      throw UsageError("the final --stages method must match --method");
    if (last) stage.inputs = target;
    if (!last) {
      stage.inputs.basis_name =
          target.preopt_basis_name.empty() ? target.basis_name : target.preopt_basis_name;
      stage.inputs.basis_explicit = !target.preopt_basis_name.empty();
      stage.inputs.density_fitting = GENERATIVEQC_DENSITY_FITTING_NONE;
      stage.inputs.density_fitting_explicit = false;
      stage.inputs.auxiliary_basis_explicit = false;
      stage.inputs.auxiliary_basis_name.clear();
      stage.inputs.density_fitting_memory_bytes = 0;
      stage.inputs.screening_tolerance.reset();
      stage.inputs.grid_explicit = false;
      stage.inputs.grid = KsGrid{};
    }
    if (is_xtb(stage.inputs)) {
      // Preliminary GFN2 uses CPU/FP64; the final target keeps the user's
      // backend and precision so unsupported requests fail in native admission.
      if (!last) {
        stage.inputs.backend = GENERATIVEQC_BACKEND_CPU_REFERENCE;
        stage.inputs.precision = GENERATIVEQC_PRECISION_FP64;
      }
      stage.inputs.basis_explicit = false;
      stage.inputs.representation_explicit = false;
      stage.inputs.auxiliary_basis_explicit = false;
      stage.inputs.density_fitting = GENERATIVEQC_DENSITY_FITTING_NONE;
      stage.inputs.screening_tolerance.reset();
    }
    stage.auto_preopt = !last && target.optimization_stages.empty();
    stage.optimizer.max_steps = target.optimization_max_steps;
    stage.optimizer.force_tolerance = last
                                          ? target.optimization_fmax
                                          : (i == 0 ? std::max(8e-3, target.optimization_fmax * 12)
                                                    : std::max(2e-3, target.optimization_fmax * 4));
    stage.optimizer.max_atom_step =
        last ? target.optimization_max_step
             : std::max(target.optimization_max_step, i == 0 ? 0.7 : 0.45);
    stages.push_back(std::move(stage));
  }
  return stages;
}

generativeqc_method_descriptor descriptor(const RunOptions& o, const System* auxiliary) {
  generativeqc_method_descriptor d{};
  d.struct_size = sizeof(d);
  d.abi_version = GENERATIVEQC_ABI_VERSION;
  d.method = o.method;
  d.max_iterations = o.max_iterations.value_or(100);
  d.diis_history = o.diis_history;
  d.energy_tolerance = o.energy_tolerance.value_or(is_xtb(o) ? 1e-10 : 1e-12);
  d.density_tolerance = o.density_tolerance.value_or(is_xtb(o) ? 1e-8 : 1e-10);
  d.screening_tolerance = o.screening_tolerance.value_or(is_xtb(o) || is_correlated(o) ? 0 : 1e-14);
  d.density_fitting_mode = o.density_fitting;
  d.density_fitting_auxiliary_basis = auxiliary ? auxiliary->get() : nullptr;
  d.density_fitting_relative_threshold = o.density_fitting_threshold;
  d.density_fitting_memory_budget_bytes = o.density_fitting_memory_bytes;
  d.precision_mode = o.precision;
  d.correlation_memory_budget_bytes = o.correlation_memory_bytes;
  d.mp2_denominator_threshold = o.mp2_denominator_threshold;
  d.ccsd_max_iterations = o.ccsd_max_iterations;
  d.ccsd_diis_history = o.ccsd_diis_history;
  d.ccsd_energy_tolerance = o.ccsd_energy_tolerance;
  d.ccsd_residual_tolerance = o.ccsd_residual_tolerance;
  d.ccsd_denominator_threshold = o.ccsd_denominator_threshold;
  d.ccsd_damping = o.ccsd_damping;
  d.ccsd_level_shift = o.ccsd_level_shift;
  return d;
}

Calculation prepare(Context& context, System& system, const RunOptions& o, const System* aux) {
  const auto d = descriptor(o, aux);
  const auto* row = o.composition;
  if (!row) return {context, system, d};
  KsComposition composed(row->carrier, std::string(row->domain), row->spin);
  composed.set_grid(o.grid);
  for (std::uint32_t j = 0; j < row->component_count; ++j) {
    const auto& term = method_generated::kComponents[row->component_offset + j];
    composed.add_semilocal(std::string(term.name), term.coefficient);
  }
  for (std::uint32_t j = 0; j < row->exchange_count; ++j) {
    const auto& term = method_generated::kExchanges[row->exchange_offset + j];
    composed.add_exact_exchange(static_cast<generativeqc_ks_exchange_operator>(term.kind),
                                term.coefficient, term.omega);
  }
  return composed.prepare(context, system, d);
}

// Rebuild a complete native System/Calculation at each geometry, preserving
// the qualified scientific method while avoiding stale cross-geometry owners.
class Objective {
 public:
  Objective(RunOptions inputs, const std::vector<generativeqc_atom>& atoms)
      : inputs_(std::move(inputs)),
        atoms_(atoms),
        context_(generativeqc_context_descriptor{sizeof(generativeqc_context_descriptor),
                                                 GENERATIVEQC_ABI_VERSION, inputs_.device_id,
                                                 inputs_.backend}) {
    if (!is_xtb(inputs_)) {
      orbital_ = expand_basis_input(
          inputs_.basis_name, atoms_,
          inputs_.representation_explicit ? std::optional(inputs_.representation) : std::nullopt);
      inputs_.representation = orbital_.representation;
    }
    if (!is_xtb(inputs_) && inputs_.auxiliary_basis_explicit) {
      auxiliary_ = expand_basis_input(inputs_.auxiliary_basis_name, atoms_,
                                      !is_local_basis(inputs_.auxiliary_basis_name)
                                          ? std::optional(inputs_.representation)
                                          : std::nullopt);
      if (!auxiliary_.ecp_terms.empty())
        throw UsageError("auxiliary optimization basis cannot contain ECP potentials");
    }
  }

  geometry::EnergyForces operator()(std::span<const double> coords) {
    if (coords.size() != atoms_.size() * 3)
      throw std::invalid_argument("geometry callback coordinate dimension mismatch");
    for (std::size_t i = 0; i < atoms_.size(); ++i) {
      atoms_[i].x = coords[3 * i];
      atoms_[i].y = coords[3 * i + 1];
      atoms_[i].z = coords[3 * i + 2];
    }
    const auto system_descriptor =
        make_system_descriptor(orbital_, is_xtb(inputs_), inputs_.representation);
    System system =
        orbital_.ecp_terms.empty()
            ? System(context_, system_descriptor)
            : System(context_, system_descriptor, orbital_.core_electrons, orbital_.ecp_terms);
    std::optional<System> aux;
    if (inputs_.auxiliary_basis_explicit) {
      aux.emplace(context_, make_system_descriptor(auxiliary_, false, auxiliary_.representation));
    }
    auto calculation = prepare(context_, system, inputs_, aux ? &*aux : nullptr);
    if (!(calculation.supported_properties() & GENERATIVEQC_PROPERTY_FORCES))
      throw Error(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                  "analytic forces are unavailable in this prepared geometry stage");
    auto result = calculation.execute(GENERATIVEQC_PROPERTY_ENERGY | GENERATIVEQC_PROPERTY_FORCES);
    if (!result.forces) throw std::runtime_error("native force owner returned no analytic forces");
    executed_backend_ = result.executed_backend;
    return {result.energy, std::move(*result.forces)};
  }

  generativeqc_backend executed_backend() const { return executed_backend_; }

 private:
  generativeqc_system_descriptor make_system_descriptor(
      const NativeBasisData& basis, bool xtb,
      generativeqc_basis_representation representation) const {
    return {sizeof(generativeqc_system_descriptor),
            GENERATIVEQC_ABI_VERSION,
            atoms_.data(),
            static_cast<std::uint32_t>(atoms_.size()),
            xtb ? nullptr : basis.shells.data(),
            xtb ? 0u : static_cast<std::uint32_t>(basis.shells.size()),
            xtb ? nullptr : basis.primitives.data(),
            xtb ? 0u : static_cast<std::uint32_t>(basis.primitives.size()),
            inputs_.charge,
            inputs_.multiplicity,
            xtb ? GENERATIVEQC_BASIS_CARTESIAN : representation};
  }
  RunOptions inputs_;
  std::vector<generativeqc_atom> atoms_;
  Context context_;
  NativeBasisData orbital_, auxiliary_;
  generativeqc_backend executed_backend_{GENERATIVEQC_BACKEND_CPU_REFERENCE};
};

void print_plan(const std::vector<Stage>& stages) {
  std::cout << "{\"status\":\"plan_only\",\"stages\":[";
  for (std::size_t i = 0; i < stages.size(); ++i) {
    if (i) std::cout << ',';
    const auto& s = stages[i];
    std::cout << "{\"method\":\"" << json_escape(s.inputs.method_name) << "\",\"backend\":\""
              << backend_name(s.inputs.backend) << "\",\"precision\":\""
              << (s.inputs.precision == GENERATIVEQC_PRECISION_AUTO ? "auto" : "fp64")
              << "\",\"basis\":\""
              << (is_xtb(s.inputs) ? "intrinsic" : json_escape(s.inputs.basis_name))
              << "\",\"force_tolerance\":" << s.optimizer.force_tolerance
              << ",\"max_steps\":" << s.optimizer.max_steps
              << ",\"optional_if_unsupported\":" << (s.auto_preopt ? "true" : "false") << '}';
  }
  std::cout << "]}\n";
}

void write_xyz(const std::filesystem::path& path, const std::vector<generativeqc_atom>& atoms,
               std::span<const double> coords, double energy) {
  // Format before creating the file: allocation errors cannot leave an output.
  std::ostringstream buffer;
  buffer << atoms.size() << "\noptimized with GenerativeQC; E=" << std::setprecision(17) << energy
         << " Hartree; units=angstrom\n";
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    buffer << atoms[i].atomic_number << ' ' << coords[i * 3] / 1.8897261254578281 << ' '
           << coords[i * 3 + 1] / 1.8897261254578281 << ' '
           << coords[i * 3 + 2] / 1.8897261254578281 << '\n';
  }
  if (!buffer) throw std::runtime_error("cannot format optimized XYZ: " + path.string());
  const auto contents = buffer.str();
  // Exclusive creation also rejects dangling symlinks and concurrent writers.
#if defined(_WIN32)
  std::FILE* stream = _wfopen(path.c_str(), L"wx");
#else
  std::FILE* stream = std::fopen(path.c_str(), "wx");
#endif
  if (!stream) {
    if (errno == EEXIST)
      throw UsageError("optimized XYZ output already exists (refusing to overwrite): " +
                       path.string());
    throw std::runtime_error("cannot create optimized XYZ: " + path.string());
  }
  const auto written = std::fwrite(contents.data(), 1, contents.size(), stream);
  const int close_status = std::fclose(stream);
  if (written != contents.size() || close_status != 0)
    throw std::runtime_error("cannot finish optimized XYZ: " + path.string());
}

}  // namespace

int geometry_command(const RunOptions& options, std::vector<generativeqc_atom> atoms) {
  const auto stages = build_stages(options);
  std::cout << std::setprecision(17);
  if (options.optimization_plan_only) {
    print_plan(stages);
    return 0;
  }
  std::vector<double> coords;
  coords.reserve(3 * atoms.size());
  for (const auto& atom : atoms) {
    coords.push_back(atom.x);
    coords.push_back(atom.y);
    coords.push_back(atom.z);
  }
  std::vector<StageRecord> reports;
  geometry::Result final;
  for (const auto& stage : stages) {
    StageRecord record;
    record.method = stage.inputs.method_name;
    record.backend = stage.inputs.backend;
    record.basis = is_xtb(stage.inputs) ? "intrinsic" : stage.inputs.basis_name;
    try {
      Objective objective(stage.inputs, atoms);
      std::optional<geometry::EnergyForces> initial;
      try {
        initial = objective(coords);
      } catch (const Error& error) {
        // Only auto-selected preparatory NOT_IMPLEMENTED may be skipped.
        // Never skip failed convergence, numerical output or target failure.
        if (!stage.auto_preopt || error.status() != GENERATIVEQC_STATUS_NOT_IMPLEMENTED) throw;
        record.status = "skipped_unsupported";
        record.detail = error.what();
        reports.push_back(std::move(record));
        continue;
      }
      bool use_first = true;
      final = geometry::minimize(coords, stage.optimizer, [&](std::span<const double> position) {
        if (use_first) {
          use_first = false;
          return *initial;
        }
        return objective(position);
      });
      record.backend = objective.executed_backend();
    } catch (const std::exception& error) {
      throw std::runtime_error("optimization stage '" + stage.inputs.method_name +
                               "': " + error.what());
    }
    coords = final.coordinates;
    record.status = "converged";
    record.outcome = final;
    reports.push_back(std::move(record));
  }
  if (reports.empty() || reports.back().status != "converged" ||
      reports.back().method != options.method_name)
    throw std::logic_error("final target method did not complete with a fresh gradient");
  if (!options.optimization_output.empty())
    write_xyz(options.optimization_output, atoms, coords, final.energy);

  if (options.json) {
    std::cout << "{\"status\":\"converged\",\"method\":\"" << json_escape(options.method_name)
              << "\",\"energy_hartree\":" << final.energy
              << ",\"max_force_hartree_per_bohr\":" << final.max_force << ",\"coordinates_bohr\":[";
    for (std::size_t i = 0; i < atoms.size(); ++i) {
      if (i) std::cout << ',';
      std::cout << '[' << coords[3 * i] << ',' << coords[3 * i + 1] << ',' << coords[3 * i + 2]
                << ']';
    }
    std::cout << "],\"stages\":[";
    for (std::size_t i = 0; i < reports.size(); ++i) {
      const auto& r = reports[i];
      if (i) std::cout << ',';
      std::cout << "{\"method\":\"" << json_escape(r.method) << "\",\"backend\":\""
                << backend_name(r.backend) << "\",\"basis\":\"" << json_escape(r.basis)
                << "\",\"status\":\"" << r.status << '"';
      if (r.status == "converged")
        std::cout << ",\"energy_hartree\":" << r.outcome.energy
                  << ",\"max_force\":" << r.outcome.max_force << ",\"steps\":" << r.outcome.steps
                  << ",\"evaluations\":" << r.outcome.evaluations;
      else
        std::cout << ",\"reason\":\"" << json_escape(r.detail) << '"';
      std::cout << '}';
    }
    std::cout << "]}\n";
  } else {
    std::cout << "optimization: converged\nmethod: " << options.method_name
              << "\nenergy_hartree: " << final.energy
              << "\nmax_force_hartree_per_bohr: " << final.max_force << '\n';
    for (const auto& r : reports) {
      std::cout << "stage: " << r.method << " (" << r.status << ")";
      if (r.status == "converged")
        std::cout << " steps=" << r.outcome.steps << " evaluations=" << r.outcome.evaluations;
      else
        std::cout << " reason=" << r.detail;
      std::cout << '\n';
    }
    std::cout << "coordinates_bohr:\n";
    for (std::size_t i = 0; i < atoms.size(); ++i)
      std::cout << "  " << atoms[i].atomic_number << ' ' << coords[3 * i] << ' '
                << coords[3 * i + 1] << ' ' << coords[3 * i + 2] << '\n';
  }
  return 0;
}

}  // namespace generativeqc::cli
