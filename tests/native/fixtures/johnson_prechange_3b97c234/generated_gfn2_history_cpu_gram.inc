    for (std::size_t row = 0u; row < history_count; ++row) {
      const std::size_t row_slot = static_cast<std::size_t>(workspace.history_slots[row]);
      const double row_omega = slot_omega(row_slot);
      double coefficient_dot = 0.0;
      if (!std::isfinite(row_omega) ||
          !dot_product(df_vector(row_slot), workspace.residual, dimension, coefficient_dot)) {
        return record_numeric_failure(state, system, "SCC mixer Broyden coefficient is not finite",
                                      error);
      }
      workspace.coefficients[row] = row_omega * coefficient_dot;
      if (!std::isfinite(workspace.coefficients[row])) {
        return record_numeric_failure(state, system, "SCC mixer Broyden coefficient overflowed",
                                      error);
      }
      for (std::size_t column = 0u; column < history_count; ++column) {
        const std::size_t column_slot = static_cast<std::size_t>(workspace.history_slots[column]);
        const double column_omega = slot_omega(column_slot);
        double overlap = 0.0;
        if (!std::isfinite(column_omega) ||
            !dot_product(df_vector(row_slot), df_vector(column_slot), dimension, overlap)) {
          return record_numeric_failure(state, system,
                                        "SCC mixer Broyden history overlap is not finite", error);
        }
        double value = row_omega * column_omega * overlap;
        if (row == column) {
          value += kOmegaZero * kOmegaZero;
        }
        if (!std::isfinite(value)) {
          return record_numeric_failure(state, system, "SCC mixer Broyden matrix overflowed",
                                        error);
        }
        workspace.beta[row * history_count + column] = value;
      }
    }