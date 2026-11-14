#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/dispersion/host_d3_column.hpp"

namespace {
using namespace generativeqc::dft::dispersion;
thread_local std::string error;

struct Handle {
  std::vector<std::int32_t> elements;
  std::vector<double> coordinates;
  std::unique_ptr<HostD3Column> column;
  const double* direction{};
  std::size_t validations{};
  int failure{};

  void validate() {
    ++validations;
    if ((failure == 1 && validations == 1) || (failure == 2 && validations == 2))
      throw std::runtime_error("injected D3 owner refusal");
    if (failure == 3 && validations == 2) const_cast<double*>(direction)[0] += .25;
    if (failure == 4) {
      std::vector<double> right(coordinates.size()), gradient(right.size()), action(right.size());
      column->apply(right, gradient, action);
    }
  }
};

template <class Action>
int checked(Action action) {
  try {
    action();
    error.clear();
    return 0;
  } catch (const std::exception& exception) {
    error = exception.what();
    return 1;
  }
}
}  // namespace

extern "C" const char* host_d3_error() { return error.c_str(); }

extern "C" int host_d3_plan(std::size_t atoms, std::size_t* output) {
  return checked([&] {
    const auto plan = HostD3Column::plan(atoms);
    const std::array values{plan.numeric_bytes, plan.table_bytes, plan.pair_visits,
                            plan.reference_terms};
    std::copy(values.begin(), values.end(), output);
  });
}

extern "C" void* host_d3_create(const std::int32_t* elements, std::size_t atoms,
                                const double* coordinates, std::size_t count,
                                const double* parameters, const std::size_t* caps) {
  std::unique_ptr<Handle> owner;
  if (checked([&] {
        owner = std::make_unique<Handle>();
        if (!elements || !coordinates) throw std::invalid_argument("null probe input");
        owner->elements.assign(elements, elements + atoms);
        owner->coordinates.assign(coordinates, coordinates + count);
        const D3Parameters model{parameters[0], parameters[1], parameters[2], parameters[3],
                                 parameters[4], parameters[5], parameters[6], parameters[7]};
        owner->column =
            std::make_unique<HostD3Column>(owner->elements, owner->coordinates, model,
                                           HostD3ColumnLimits{caps[0], caps[1], caps[2]},
                                           [handle = owner.get()] { handle->validate(); });
      }))
    return nullptr;
  return owner.release();
}

extern "C" void host_d3_destroy(void* pointer) { delete static_cast<Handle*>(pointer); }

extern "C" int host_d3_apply(void* pointer, const double* direction, std::size_t direction_count,
                             double* gradient, std::size_t gradient_count, double* action,
                             std::size_t action_count, double* value, int failure) {
  return checked([&] {
    auto& owner = *static_cast<Handle*>(pointer);
    owner.validations = 0;
    owner.failure = failure;
    owner.direction = direction;
    const auto result = owner.column->apply({direction, direction_count},
                                            {gradient, gradient_count}, {action, action_count});
    value[0] = result.energy;
    value[1] = result.right;
  });
}

extern "C" void host_d3_work(void* pointer, std::size_t* output) {
  const auto work = static_cast<Handle*>(pointer)->column->work();
  const std::array values{work.coordination_pairs, work.reference_atoms, work.energy_pairs,
                          work.reverse_pairs, work.reference_terms};
  std::copy(values.begin(), values.end(), output);
}

extern "C" void host_d3_mutate_snapshot_inputs(void* pointer) {
  auto& owner = *static_cast<Handle*>(pointer);
  owner.coordinates[0] += 1;
  owner.elements[0] = 1;
}

#ifdef GENERATIVEQC_D3_COLUMN_STANDALONE
int main() {
  const std::array<std::int32_t, 4> elements{28, 53, 55, 1};
  const std::array<double, 12> coordinates{0, 0, 0, 3.2, .3, .1, -.4, 4.5, .2, .1, -.2, 2.1};
  const auto plan = HostD3Column::plan(elements.size());
  HostD3Column column(elements, coordinates, {1, 1.9889, .3981, 4.4211, 0},
                      {plan.numeric_bytes, plan.pair_visits, plan.reference_terms});
  std::array<double, 12> right{}, gradient{}, action{};
  right[3] = .12;
  const auto value = column.apply(right, gradient, action);
  if (!(value.energy < 0) || std::abs(value.right) < 1e-10 ||
      std::all_of(action.begin(), action.end(),
                  [](double entry) { return std::abs(entry) < 1e-10; }))
    return 1;
  right.fill(1);
  column.apply(right, gradient, action);
  if (std::any_of(action.begin(), action.end(), [](double entry) { return entry != 0; })) return 2;
  std::cout << "native AOT D3(BJ) two-body columns: Ni/I/Cs/H, ATM off\n";
}
#endif
