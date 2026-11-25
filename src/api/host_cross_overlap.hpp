#pragma once

#include <cstddef>

#include "generativeqc/generativeqc.h"

/** Query (null output, count zero) or execute a CPU cross overlap with a
 * modeled byte limit for native temporary storage. The limit excludes the
 * caller's metric and input systems. Only the query takes required_bytes;
 * execution takes output, output_count and an optional three-double translation
 * in Bohr to subtract from target centres only. Capacity refusal publishes neither
 * a metric nor a query result. */
extern "C" GENERATIVEQC_API generativeqc_status generativeqc_system_cross_overlap_host_v1(
    generativeqc_context* context, const generativeqc_system* target,
    const generativeqc_system* source, std::size_t maximum_bytes, double* output,
    std::size_t output_count, std::size_t* required_bytes, const double* subtract_from_target);
