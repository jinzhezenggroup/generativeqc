# Decision: stage mixed precision at the DF triples W contractions

Status: compiler candidate only; no runtime/default promotion
Date: 2026-10-04

## Problem

Issue #1764 asks for component-wise mixed precision in RCCSD(T) without turning
the complete correlation solve or force chain into FP32. The bounded native DF
triples phase from #1785 already exposes its dominant W contractions as two
audited TensorIR reductions, but its runtime is deliberately strict FP64.

A direct CUDA-only float branch would duplicate precision semantics outside the
compiler and make later provenance/audit difficult. Requiring FP64 accumulation
inside the FP32 GEMM is also not the first performance candidate: the current
TensorIR contract correctly routes FP32-compute/FP64-accumulation reductions
away from SGEMM because SGEMM itself accumulates in FP32.

## Decision

Add one explicit TensorIR candidate, `w_fp32_candidate_program`, that lowers only
the two W contractions to FP32 storage, compute and accumulation. Every input
and published output remains FP64 through explicit cast boundaries. The W sum,
all V algebra, denominators, energy epilogue and final reductions remain FP64.
The schedule keeps FP64 as the strict audit dtype and records a qualification
identifier scoped to #1764.

This slice does not alter `cc::triples::evaluate_df_cuda`, does not add a public
precision switch, and does not claim a speedup. Static CUDA planning must keep
both lowered contractions GEMM-eligible; host reference execution checks the
candidate against the strict equation while requiring the untouched V path to
remain bitwise identical.

## Why this boundary

The W contractions scale as the dominant matrix products in the occupied-tile
triples phase. Starting there isolates the likely throughput win while keeping
precision-sensitive division, cancellation-heavy energy assembly and all final
reductions strict. It also avoids prematurely converting the cached DF integral
panels or CC amplitudes to persistent FP32 storage before transfer/cast traffic
has been measured.

## Remaining qualification

1. Bind the compiler schedule to the #1785 native CUDA owner without inventing a
   CC-specific handwritten float kernel or hidden precision path.
2. Measure cast/traffic cost and actual SGEMM use on RTX 5090 and RTX PRO 6000.
3. Compare complete triples and complete molecular endpoints with strict FP64,
   including E_(T), Lambda/orbital residuals and analytic/finite-difference force
   gates once the response chain is selected.
4. Exercise small denominators, near-degenerate spaces and cancellation-heavy
   triples before any AUTO/default promotion.
5. Record the selected precision schedule in result/method provenance and retain
   strict FP64 fallback.

Refs #1764, #1785, #528, #485, #981.

Agent: ChatGPT
Model: GPT-5.6 Sol
