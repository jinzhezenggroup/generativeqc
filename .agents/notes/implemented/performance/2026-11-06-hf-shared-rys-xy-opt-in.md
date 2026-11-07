# Decision: keep SM90 HF shared Rys x/y behind a frozen opt-in

Status: implemented, endpoint qualification pending
Date: 2026-11-06

## Problem

DPPP/PPPP component-lane Fock workers repeat the same unweighted x/y recurrence
across component lanes. The independently qualified phase-A schedule shares those
tables without changing weighted z, primitive/root accumulation, screening or
scatter. Its primary `KernelSelection` guard deliberately prevents accidental
promotion to a default production schedule.

## Decision

Declare `shared_rys_xy_fock` as an explicit optional companion of the portable
manifest's two classes, and construct it only for the exact SM90 build target.
Generic optional streaming variants remain disabled in the portable profile.
Distinct namespaces and symbols keep primary/private source bodies unchanged.
The existing conservative private fallback for at most two retained components
is unchanged; it is not a tuned optimum.

Only exact `GENERATIVEQC_CUDA_HF_SHARED_RYS_XY=1` requests the companion. Unset,
empty and `0` remain off; other values fail before graph preparation. Admission
requires restricted FP64 quartet-Direct materialized pages, compatible compiled
capability, and no incremental/separate J/K, relevant no-materialization streaming,
persistent ERI or compensated output. Unsupported valid requests retain an
explicit refusal in the plan policy. The plan freezes request, admission and class
mask, and rejects incompatible cached reuse before graph preparation. Python CUDA
prepared/restart identity additionally preserves the exact environment value.
CPU identity is unchanged.

`bounded_direct_streaming` is an arena/queue-policy name: its primary materialized
wave and exact-class retry pages remain eligible. Its separate no-materialization
streaming consumers are untouched. The generated companion dispatcher falls back
before enqueue for absent capability, other classes, UHF or compensated output;
an error from an enqueued candidate propagates without an incumbent retry.

## Evidence and boundaries

Measured arithmetic source: b56c2c935aa63f68049a116e420aa5a5dcb529e8.
The synthetic qualification covers signed multiple primitives, coincident/reversed
centers, full/partial/one/two/zero masks, CTA reuse, ordinary and graph replay.
The actual recovered first-queue replay result SHA256 is
96900da8c011830385935044763f4b2ac29656d020edfbb6e1bce7b2f022fb3f.
It passed twelve full-matrix predicates and ten balanced unfiltered pairs/class;
median native/candidate ratios were 1.97589183557 (DPPP) and 1.94726016708 (PPPP).
The original post-dump capture process remains failed; its separately sealed CPU
recovery authority is 869a088cdff5ed389737884f36b6b71d4db1692f1334b935b4193f846b79148c.
These local queue results do not establish an endpoint speedup. The 24-file
payload inventory is 297012197 bytes including metadata; raw binary payload is
296998176 bytes.

The runtime adds only the measured 576/432 shared bytes per CTA. No new numeric
arrays, queues, task capacity, stack limit or retained cache enter production.
Built resources, actual new-dispatch graph correctness and the original HF96
cold plus three-warm off/on pilot remain execution gates. Both arms retain the
original `native_selector=0`; only the new control differs. The strict native
E/atom, Fmax and Frms bounds are 1e-9, 1e-7 and 3e-8. A positive pilot retains this
opt-in stage; it does not enable the default or complete the original30-row goal.

## Revisit when

Broader/default promotion requires source-bound endpoint qualification against the
unchanged independent oracles, all original30 rows with at least three fresh
same-GPU native/reference pairs, every ratio above1, median above1.05 and at most
5% warm regression. Terminal gather is a separate design and is not part of this
integration.
