# Decision: separate one-electron angular families at the CUDA module boundary

Status: implemented; new native/device and endpoint qualification pending
Date: 2026-11-15

## Evidence and revised attribution

The primary RTX acquisition regression was an explicit
`GENERATIVEQC_AOT_PROFILE=portable_cuda` override, not three-process repetition.
The default `auto` selects the qualified SM120 Fock schedule. A same-source
native-18A comparison retained 58,325 kernels per profiled PBE0 Direct endpoint
pair while shell-class Fock time fell from 9.885 s to 5.590 s. Restore that build
default independently; do not relabel the portable observations or force SM120
on H100, whose automatic profile remains portable.

Two DF cold comparisons still missed after the correction. Scheduled Nsight job
1589 localized another difference before actual GPU execution. For the first
HF DF one-electron gradient launch, the earlier image's runtime launch API took
about 93 ms and the kernel 92 ms. The expanded g image took about 574 ms in the
launch API and 96 ms in the kernel. Its subsequent launch took about 0.015 ms
in the API. These diagnostic clocks remain separate from clean endpoint timing.

The earlier through-f specialization in
[the angular-family note](2026-11-15-one-electron-angular-families.md) restored
small kernel resources but kept f and g in one CUDA translation unit. Nsight
confirmed that the f program really executed: its kernel took about 92 ms,
while its first launch API still took about 636 ms. Correct mathematics or a
smaller per-kernel frame did not establish a smaller first-use module cost.

## Decision

Place through-f values and derivatives in separate CUDA translation units from
the complete g programs. Keep the normalized primitive/AO traversal in shared
private runtime headers, rather than copying its weighting, sign, partitioning
or accumulation behavior. Keep scalar mathematics and typed program bindings
owned by the existing compiler emitters.

The public host dispatch includes only lightweight compiler-emitted angular
bounds. It calls the f module for validated physical shell maxima through f;
unknown or higher admitted metadata retains the g module. Cartesian g remains
unrelated to expansion stride. Calling host dispatch must not require including
the unused high-angular scalar program in the f translation unit.

Ordinary core CUDA separable compilation is off by default, so these are
independent lazy-load modules. An explicitly combined device-link mode retains
correctness but may remove the first-use benefit; no performance claim is
transferred to that configuration without complete-consumer measurements.

## Invariants and pending acceptance

No initialization is moved outside the existing cold timer. No physical task,
precision, screening, convergence, geometry/basis or numerical gate changes.
Both angular domains remain available. No new tensor allocation, derivative
materialization or ownership/lifetime shortcut is introduced.

Re-run the cached independent host blocks and the real Cartesian/spherical f/g
arbitrary-weight, resource, complete energy/force and owner-identity controls.
Then require a fresh scheduled endpoint comparison against the corrected auto
baseline and a complete three-process paper cohort. Preserve the earlier failed
combined-module experiment and all diagnostic reports; this decision alone is
not a 64/64 latency or overall scientific acceptance claim.
