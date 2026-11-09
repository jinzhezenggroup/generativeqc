"""Bound admission overhead without changing a component-lane K recurrence."""

from __future__ import annotations


def component_exchange_admission_loop(
    *,
    prefix: str,
    high_pair_class: int,
    low_pair_class: int,
    system_density_bound: str,
    fock_task: str,
    record_precision: str,
) -> str:
    """Emit a full-warp admission queue followed by the unchanged CTA consumer.

    The caller supplies a strict shared-Rys component schedule with at least one
    complete warp. Only restricted, density-bounded pure K enters this loop;
    all other consumers continue into the caller's original worker. Different
    pair classes retain two-dimensional concurrency with 32-product claims.
    Same-pair classes keep sorted rows and their conservative coarse tail.
    """
    different_pairs = high_pair_class != low_pair_class
    bra = "compact_bra[index]" if different_pairs else "bra_pair"
    ket = "compact_ket[index]" if different_pairs else "compact_pairs[index]"
    consume = f"""
        for (unsigned index = 0U; index < compact_count; ++index) {{
          if (threadIdx.x == 0U) {{
            {record_precision}
            {prefix}_stream_populate_task(
                topology, {bra}, {ket}, stream_task[0]);
          }}
          __syncthreads();
          {fock_task}(
              stream_task, primitive_pairs, primitive_pair_offsets,
              ao_coefficients, atom_positions, screening_tolerance,
              schwarz_bounds, density, fock, 0U);
          __syncthreads();
        }}
        // Retire every count/queue reader before the next batch publishes.
        __syncthreads();
"""
    admission = """
    if (topology.fock_consumer ==
            generativeqc::scf::detail::GeneratedFockConsumer::Exchange &&
        topology.shell_pair_density_bounds != nullptr &&
        topology.system_pair_density_bounds != nullptr)
"""
    if different_pairs:
        return f"""
  __shared__ std::uint32_t compact_bra[32];
  __shared__ std::uint32_t compact_ket[32];
  __shared__ std::uint32_t compact_count;
  __shared__ std::uint32_t compact_in_domain;
  if constexpr (!Unrestricted) {{
    {admission} {{
      while (true) {{
        if (threadIdx.x == 0U) candidate_ordinal = atomicAdd(task_head, 1U);
        __syncthreads();
        if (threadIdx.x < 32U) {{
          const unsigned lane = threadIdx.x;
          std::uint64_t remaining = std::uint64_t(candidate_ordinal) * 32U + lane;
          bool in_domain = false;
          bool keep = false;
          std::uint32_t selected_bra = 0U;
          std::uint32_t selected_ket = 0U;
          for (std::int32_t system = 0; system < topology.batch_size; ++system) {{
            const std::uint32_t high_begin = topology.pair_class_offsets[
                {high_pair_class}U * stride + static_cast<std::size_t>(system)];
            const std::uint32_t high_end = topology.pair_class_offsets[
                {high_pair_class}U * stride + static_cast<std::size_t>(system) + 1U];
            const std::uint32_t low_begin = topology.pair_class_offsets[
                {low_pair_class}U * stride + static_cast<std::size_t>(system)];
            const std::uint32_t low_end = topology.pair_class_offsets[
                {low_pair_class}U * stride + static_cast<std::size_t>(system) + 1U];
            const std::uint64_t high_count = high_end - high_begin;
            const std::uint64_t low_count = low_end - low_begin;
            const std::uint64_t system_candidates = high_count * low_count;
            if (remaining >= system_candidates) {{
              remaining -= system_candidates;
              continue;
            }}
            const std::uint64_t high_local = remaining / low_count;
            const std::uint64_t low_local = remaining - high_local * low_count;
            const std::uint32_t bra_pair = topology.pair_order[
                high_begin + static_cast<std::uint32_t>(high_local)];
            const std::uint32_t ket_pair = topology.pair_order[
                low_begin + static_cast<std::uint32_t>(low_local)];
            in_domain = true;
            const bool past_schwarz_tail = topology.shell_pair_bounds[bra_pair] *
                topology.shell_pair_bounds[ket_pair] *
                {system_density_bound} < screening_tolerance;
            double contribution_bound = 0.0;
            keep = !past_schwarz_tail && {prefix}_stream_survives<false>(
                topology, bra_pair, ket_pair, screening_tolerance, &contribution_bound);
            selected_bra = bra_pair;
            selected_ket = ket_pair;
            break;
          }}
          // Empty systems and partial batches still participate in both ballots.
          const unsigned domain_mask = __ballot_sync(0xffffffffU, in_domain);
          const unsigned mask = __ballot_sync(0xffffffffU, keep);
          if (keep) {{
            const unsigned rank = __popc(mask & ((1U << lane) - 1U));
            compact_bra[rank] = selected_bra;
            compact_ket[rank] = selected_ket;
          }}
          if (lane == 0U) {{
            compact_count = __popc(mask);
            compact_in_domain = __popc(domain_mask);
          }}
        }}
        __syncthreads();
        // A screened local tail cannot retire later bras or systems.
        if (compact_in_domain == 0U) return;
        {consume}
      }}
    }}
  }}
"""
    return f"""
  __shared__ std::uint32_t compact_pairs[32];
  __shared__ std::uint32_t compact_count;
  __shared__ std::uint32_t compact_end;
  if constexpr (!Unrestricted) {{
    {admission} {{
      while (true) {{
        if (threadIdx.x == 0U) bra_ordinal = atomicAdd(bra_head, 1U);
        __syncthreads();
        if (bra_ordinal >= bra_end - bra_begin) return;
        const std::uint32_t bra_pair = topology.pair_order[bra_begin + bra_ordinal];
        const std::int32_t system = topology.shell_pair_systems[bra_pair];
        const std::uint32_t ket_begin = topology.pair_class_offsets[
            {low_pair_class}U * stride + system];
        const std::uint32_t ket_end = topology.pair_class_offsets[
            {low_pair_class}U * stride + system + 1U];
        if (threadIdx.x == 0U) {{
          compact_end = {prefix}_stream_coarse_ket_end(
              topology, bra_pair, ket_begin, ket_end, {system_density_bound}, screening_tolerance);
        }}
        __syncthreads();
        // Widen before advancing a partial warp near the 32-bit boundary.
        for (std::uint64_t base = ket_begin; base < compact_end; base += 32U) {{
          if (threadIdx.x < 32U) {{
            const unsigned lane = threadIdx.x;
            const std::uint64_t ordinal = base + lane;
            const std::uint32_t ket_pair = ordinal < compact_end
                ? topology.pair_order[ordinal] : 0U;
            bool keep = ordinal < compact_end;
            if (keep) keep = bra_pair >= ket_pair;
            double contribution_bound = 0.0;
            if (keep) {{
              keep = {prefix}_stream_survives<false>(
                  topology, bra_pair, ket_pair, screening_tolerance, &contribution_bound);
            }}
            const unsigned mask = __ballot_sync(0xffffffffU, keep);
            if (keep) compact_pairs[__popc(mask & ((1U << lane) - 1U))] = ket_pair;
            if (lane == 0U) compact_count = __popc(mask);
          }}
          __syncthreads();
          {consume}
        }}
      }}
    }}
  }}
"""
