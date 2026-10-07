"""Bounded post-admission K queues, independent of the scientific ERI producer."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


def exchange_queue_declarations(width: int) -> str:
    """One pending batch plus one screening chunk bounds storage by 2 * width."""
    return f"""
  __shared__ std::uint32_t exchange_queue_pairs[{2 * width}];
  __shared__ double exchange_queue_bounds[{2 * width}];
  __shared__ std::uint32_t exchange_queue_count;
"""


def exchange_queue_sort_source(prefix: str) -> str:
    """Sort a small admitted window stably; the bra's primitive count is fixed.

    Never sort the Schwarz-ordered input. The optional schedule must account for
    this serial sorting overhead rather than assuming workload grouping is free.
    """
    return f"""
__device__ __forceinline__ void {prefix}_sort_exchange_queue(
    std::uint32_t* pairs, double* bounds, std::uint32_t count,
    const std::int64_t* primitive_pair_offsets) {{
  for (std::uint32_t position = 1U; position < count; ++position) {{
    const std::uint32_t pair = pairs[position];
    const double bound = bounds[position];
    const std::int64_t work =
        primitive_pair_offsets[pair + 1U] - primitive_pair_offsets[pair];
    std::uint32_t target = position;
    while (target > 0U) {{
      const std::uint32_t previous = pairs[target - 1U];
      const std::int64_t previous_work =
          primitive_pair_offsets[previous + 1U] - primitive_pair_offsets[previous];
      if (previous_work >= work) break;
      pairs[target] = previous;
      bounds[target] = bounds[target - 1U];
      --target;
    }}
    pairs[target] = pair;
    bounds[target] = bound;
  }}
}}
"""


def exchange_streaming_worker(
    *,
    prefix: str,
    class_name: str,
    internal_parameters: str,
    shell_class: int,
    high_pair_class: int,
    low_pair_class: int,
    system_density_bound: str,
    block_threads: int,
    width: int,
    packed: bool,
    local_lane_state: bool,
    supports_mixed_fock: bool,
    retained_state: str,
    record_precision: Callable[[str], str],
) -> str:
    """Emit one recurrence consumer for incumbent and optional K schedules.

    Screen original candidates once in their original order. Each scan starts
    with fewer than width pending survivors and appends at most width, so the
    2 * width arena cannot overflow. Consumers drain survivor batches, not the
    original candidate chunks. Collectives retain the full launch membership.
    """
    execution_slot = "threadIdx.x" if packed else "subgroup"
    leader = "true" if packed else "lane == 0U"
    barrier = "__syncwarp(full_warp_mask);" if packed else "__syncthreads();"
    warp_mask_declaration = (
        "constexpr unsigned full_warp_mask = 0xffffffffU;" if packed else ""
    )
    if packed and local_lane_state:
        task_reference, task_pointer, task_index = "stream_task", "&stream_task", "0U"
        storage_reference = "lane_storage"
        task_storage = f"""
  Generated{class_name}ShellTask stream_task;
  Generated{class_name}PackedFockLaneStorage lane_storage;
"""
    elif packed:
        task_reference = "stream_tasks[threadIdx.x]"
        task_pointer = "stream_tasks"
        task_index = "static_cast<std::size_t>(threadIdx.x)"
        storage_reference = "lane_storage[threadIdx.x]"
        task_storage = f"""
  __shared__ Generated{class_name}ShellTask stream_tasks[32];
  __shared__ Generated{class_name}PackedFockLaneStorage lane_storage[32];
"""
    else:
        task_reference = "stream_tasks[subgroup]"
        task_pointer = "stream_tasks"
        task_index = "static_cast<std::size_t>(subgroup)"
        storage_reference = "subgroup_storage[subgroup]"
        storage_type = f"Generated{class_name}SubgroupFockStorage"
        union_declaration = ""
        if supports_mixed_fock:
            storage_type = f"Generated{class_name}StreamingSubgroupFockStorage"
            union_declaration = f"""
  union {storage_type} {{
    Generated{class_name}SubgroupFockStorage fp64;
    Generated{class_name}MixedSubgroupFockStorage mixed;
  }};
"""
        subgroup_lanes = block_threads // width
        task_storage = f"""
  __shared__ Generated{class_name}ShellTask stream_tasks[{width}];
{union_declaration}
  __shared__ {storage_type} subgroup_storage[{width}];
  __shared__ std::uint32_t stream_keep[{width}];
  const unsigned subgroup = threadIdx.x / {subgroup_lanes}U;
  const unsigned lane = threadIdx.x % {subgroup_lanes}U;
  const unsigned subgroup_mask = {((1 << subgroup_lanes) - 1)}U <<
      ((subgroup % {32 // subgroup_lanes}U) * {subgroup_lanes}U);
"""
    append = (
        """
          const unsigned survivor_mask = __ballot_sync(full_warp_mask, keep);
          const unsigned lower_lane_mask = (1U << threadIdx.x) - 1U;
          const unsigned rank = __popc(survivor_mask & lower_lane_mask);
          if (keep) {
            exchange_queue_pairs[pending + rank] = ket_pair;
            exchange_queue_bounds[pending + rank] = contribution_bound;
          }
          if (threadIdx.x == 0U)
            exchange_queue_count = pending + __popc(survivor_mask);
"""
        if packed
        else """
          if (keep) {
            const unsigned rank = atomicAdd(&exchange_queue_count, 1U);
            exchange_queue_pairs[rank] = ket_pair;
            exchange_queue_bounds[rank] = contribution_bound;
          }
"""
    )
    consumer = "packed_fock_lane" if packed else "subgroup_fock_task"
    mixed_consumer = "packed_mixed_fock_lane" if packed else "mixed_subgroup_fock_task"
    subgroup_arguments = "" if packed else ", lane, subgroup_mask"
    fp64_storage = storage_reference + (
        ".fp64" if supports_mixed_fock and not packed else ""
    )
    mixed_storage = storage_reference if packed else storage_reference + ".mixed"
    arguments = f"""{task_pointer}, primitive_pairs, primitive_pair_offsets,
              ao_coefficients, atom_positions, screening_tolerance,
              schwarz_bounds, density, fock, {task_index}"""
    mixed_execution = (
        f"""if (precision_state == 3U) {{
          {prefix}_{mixed_consumer}<Unrestricted>(
              {arguments}, {mixed_storage}{subgroup_arguments});
        }} else """
        if supports_mixed_fock
        else ""
    )
    prepare_task = f"""
        if (keep) {{
          const std::uint32_t precision_state = {retained_state};
          {record_precision("precision_state")}
          {prefix}_stream_populate_task(topology, bra_pair, ket_pair, {task_reference});
        }}
"""
    if not packed:
        prepare_task += f"""
        stream_keep[subgroup] = keep ? {retained_state} : 0U;
"""
    execute_condition = "keep" if packed else "stream_keep[subgroup] != 0U"
    execution_state = retained_state if packed else "stream_keep[subgroup]"
    return f"""
{exchange_queue_sort_source(prefix)}
template <bool Unrestricted>
__device__ __forceinline__ void {prefix}_streaming_fock(
{internal_parameters}) {{
  static_assert(kGenerated{class_name}FockBlockThreads == {block_threads}U);
  {warp_mask_declaration}
{task_storage}
{exchange_queue_declarations(width)}
  __shared__ std::uint32_t bra_ordinal;
  const auto& topology = *topology_pointer;
  if (topology.generated_overflow != nullptr &&
      topology.generated_overflow[{shell_class}U] == 0U) return;
  const std::size_t stride = static_cast<std::size_t>(topology.batch_size) + 1U;
  const std::uint32_t bra_begin = topology.pair_class_offsets[
      {high_pair_class}U * stride];
  const std::uint32_t bra_end = topology.pair_class_offsets[
      {high_pair_class}U * stride + topology.batch_size];
  while (true) {{
    if (threadIdx.x == 0U) bra_ordinal = atomicAdd(bra_head, 1U);
    __syncthreads();
    const std::uint32_t claimed_bra = bra_ordinal;
    // An empty row must not let a leader overwrite a late lane's claim.
    __syncthreads();
    if (claimed_bra >= bra_end - bra_begin) return;
    const std::uint32_t bra_pair = topology.pair_order[bra_begin + claimed_bra];
    const std::int32_t system = topology.shell_pair_systems[bra_pair];
    const std::uint32_t ket_begin = topology.pair_class_offsets[
        {low_pair_class}U * stride + system];
    const std::uint32_t ket_end = topology.pair_class_offsets[
        {low_pair_class}U * stride + system + 1U];
    const double system_density_bound = {system_density_bound};
    const std::uint32_t coarse_ket_end = {prefix}_stream_coarse_ket_end(
        topology, bra_pair, ket_begin, ket_end, system_density_bound,
        screening_tolerance);
    const bool compact_exchange =
        (topology.fock_consumer ==
             generativeqc::scf::detail::GeneratedFockConsumer::Exchange ||
         topology.fock_consumer ==
             generativeqc::scf::detail::GeneratedFockConsumer::HartreeFockExchange);
    const bool queued_exchange = compact_exchange &&
        topology.exchange_task_schedule !=
            generativeqc::scf::detail::GeneratedExchangeTaskSchedule::Incumbent;
    if (threadIdx.x == 0U) exchange_queue_count = 0U;
    {barrier}
    std::uint32_t ket_base = ket_begin;
    while (ket_base < coarse_ket_end || exchange_queue_count != 0U) {{
        const std::uint32_t pending = exchange_queue_count;
        // Snapshot before any subgroup changes the shared queue count.
        {barrier}
        bool keep = false;
        std::uint32_t ket_pair = 0U;
        double contribution_bound = 0.0;
        if (ket_base < coarse_ket_end && pending < {width}U) {{
          if ({leader}) {{
            const std::uint32_t ordinal = ket_base + {execution_slot};
            const bool in_range = ordinal < coarse_ket_end;
            ket_pair = in_range ? topology.pair_order[ordinal] : 0U;
            keep = in_range;
            if (keep && {str(high_pair_class == low_pair_class).lower()})
              keep = bra_pair >= ket_pair;
            if (keep) {{
              keep = {prefix}_stream_survives<Unrestricted>(
                  topology, bra_pair, ket_pair, screening_tolerance,
                  &contribution_bound);
            }}
            if (compact_exchange) {{
{append}
            }}
          }}
          {barrier}
          ket_base += {width}U;
        }}
        const std::uint32_t available = exchange_queue_count;
        if (queued_exchange && available < {width}U && ket_base < coarse_ket_end)
          continue;
        if (queued_exchange && threadIdx.x == 0U &&
            topology.exchange_task_schedule ==
                generativeqc::scf::detail::GeneratedExchangeTaskSchedule::Primitive) {{
          {prefix}_sort_exchange_queue(exchange_queue_pairs, exchange_queue_bounds,
                                      available, primitive_pair_offsets);
        }}
        {barrier}
        const std::uint32_t consumed = available < {width}U ? available : {width}U;
        if ({leader}) {{
          if (compact_exchange) {{
            keep = {execution_slot} < consumed;
            if (keep) {{
              ket_pair = exchange_queue_pairs[{execution_slot}];
              contribution_bound = exchange_queue_bounds[{execution_slot}];
            }}
          }}
{prepare_task}
        }}
        {barrier}
        if ({execute_condition}) {{
          const std::uint32_t precision_state = {execution_state};
          {mixed_execution}{{
            {prefix}_{consumer}<Unrestricted>(
                {arguments}, {fp64_storage}{subgroup_arguments});
          }}
        }}
        {barrier}
        const std::uint32_t remaining = available - consumed;
        std::uint32_t retained_pair = 0U;
        double retained_bound = 0.0;
        if ({leader} && {execution_slot} < remaining) {{
          retained_pair = exchange_queue_pairs[consumed + {execution_slot}];
          retained_bound = exchange_queue_bounds[consumed + {execution_slot}];
        }}
        {barrier}
        if ({leader} && {execution_slot} < remaining) {{
          exchange_queue_pairs[{execution_slot}] = retained_pair;
          exchange_queue_bounds[{execution_slot}] = retained_bound;
        }}
        if (threadIdx.x == 0U) exchange_queue_count = remaining;
        {barrier}
    }}
  }}
}}
"""
