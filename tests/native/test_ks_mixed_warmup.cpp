#include <iostream>
#include <limits>
#include <stdexcept>

#include "dft/mixed_precision_warmup.hpp"

int main() {
  try {
    const auto require = [](bool condition, const char* message) {
      if (!condition) throw std::runtime_error(message);
    };
    using generativeqc::dft::resolve_mixed_precision_warmup;
    const auto legacy = resolve_mixed_precision_warmup(nullptr, nullptr);
    require(!legacy.enter_refinement(true, false, 5, 100, 1e-8),
            "absent controls must retain the incumbent schedule");
    require(legacy.enter_refinement(true, true, 5, 100, 1e-8),
            "incumbent approximate convergence must request strict work");
    require(legacy.enter_refinement(true, false, 100, 100, 1e-2),
            "incumbent budget must request strict work");
    const auto early = resolve_mixed_precision_warmup("1e-4", "24");
    require(!early.enter_refinement(true, false, 1, 100, 1e-5),
            "residual switching must retain at least two approximate observations");
    require(early.enter_refinement(true, false, 2, 100, 1e-4),
            "the measured residual boundary must request strict work");
    require(!early.enter_refinement(true, false, 2, 100, 1.1e-4),
            "the physical maximum residual must control the handoff");
    require(early.enter_refinement(true, false, 24, 100, 1e-2),
            "the optional cap must bound approximate work");
    require(early.enter_refinement(true, false, 3, 3, 1e-2),
            "warmup must not exceed the user's target budget");
    require(!early.enter_refinement(true, true, 2, 100, 1e-8, true),
            "a provisional nonlocal residual cannot certify target switching");
    require(early.enter_refinement(true, false, 24, 100, 1e-2, true),
            "provisional nonlocal work must consume the mixed iteration cap");
    require(early.enter_refinement(true, false, 3, 3, 1e-2, true),
            "a provisional stage must not exceed the original solve budget");
    require(!early.target_refresh_due(true, 22, 100) && early.target_refresh_due(true, 23, 100),
            "initializers must leave one mixed target Fock within the cap");
    require(early.target_refresh_due(true, 2, 3) && !early.target_refresh_due(false, 23, 100) &&
                !early.target_refresh_due(true, 1, 1),
            "target reservation must respect strict and short solve domains");
    require(!early.enter_refinement(false, true, 100, 100, 1e-8),
            "strict execution must never restart approximate refinement");
    for (const double invalid :
         {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), -1.0})
      require(!early.enter_refinement(true, false, 100, 100, invalid),
              "invalid physical diagnostics must not authorize a handoff");
    for (const auto* invalid : {"", "nan", "inf", "-1", "1e-4tail"}) {
      bool rejected = false;
      try {
        resolve_mixed_precision_warmup(invalid, "24");
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      require(rejected, "invalid residual control was accepted");
    }
    for (const auto* invalid : {"", "-1", "2.5", "24tail", "4294967296"}) {
      bool rejected = false;
      try {
        resolve_mixed_precision_warmup("1e-4", invalid);
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      require(rejected, "invalid iteration control was accepted");
    }
    std::cout << "validated bounded KS mixed warmup selection\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
