    const std::size_t history_count = static_cast<std::size_t>(
        std::min<std::uint64_t>(static_cast<std::uint64_t>(memory), old_iteration));
    const std::uint64_t first_iteration =
        old_iteration - static_cast<std::uint64_t>(history_count) + 1u;
    const std::size_t new_slot =
        static_cast<std::size_t>((old_iteration - 1u) % static_cast<std::uint64_t>(memory));
    for (std::size_t history = 0u; history < history_count; ++history) {
      const std::uint64_t represented_iteration =
          first_iteration + static_cast<std::uint64_t>(history);
      workspace.history_slots[history] = static_cast<std::int64_t>(
          (represented_iteration - 1u) % static_cast<std::uint64_t>(memory));
    }

    const auto df_vector = [&](std::size_t slot) {
      return slot == new_slot ? workspace.delta_f
                              : state.df_history + history_offset + slot * dimension;
    };
    const auto u_vector = [&](std::size_t slot) {
      return slot == new_slot ? workspace.new_u
                              : state.u_history + history_offset + slot * dimension;
    };
    const auto slot_omega = [&](std::size_t slot) {
      return slot == new_slot ? omega : state.omega[system * memory + slot];
    };