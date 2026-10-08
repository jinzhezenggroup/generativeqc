    for (std::size_t component = 0u; component < dimension; ++component) {
      double value = current[component] + data.damping * workspace.residual[component];
      for (std::size_t history = 0u; history < history_count; ++history) {
        const std::size_t slot = static_cast<std::size_t>(workspace.history_slots[history]);
        value -= slot_omega(slot) * workspace.coefficients[history] * u_vector(slot)[component];
      }
      if (!std::isfinite(value)) {
        return record_numeric_failure(state, system, "SCC mixer Broyden result is not finite",
                                      error);
      }
      workspace.mixed[component] = value;
    }