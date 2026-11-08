# Decision: reject cooperative Rys series as an HF cold optimization

Status: rejected after balanced A100 and RTX5090 complete endpoints
Date: 2026-11-08

## Hypothesis and implementation

The clean default HF96 A100 census LSF228690 has repeated combined-Fock
persistent DPPP/PPPP kernels. Their component worker computes independent
root/weight interpolation series on lane zero. An experiment factored the
existing limiting rules and Clenshaw emitter, emitted a per-series companion,
and assigned disjoint series to CTA lanes between two publication barriers.
The producer lanes ran independently of Cartesian screening. Arithmetic,
tables, thresholds, force support, shared-XY, weighted-Z, packed and mixed
owners stayed unchanged. A distinct schedule bit selected only the existing
portable DPPP/PPPP value companions; no GPU-product-name check was introduced.

Eight scalar default table/precision variants stayed byte-identical in a
one-off audit, not a permanent source-fingerprint test. The actual SM120
DPPP/PPPP production shaders also stayed byte-identical. The candidate source
was frozen from clean HEAD with an explicit owned-file list, excluding the
older adaptive-delta experiment's thirteen dirty paths.

## Decision

Reject and remove all seven production/compiler edits and three new fixture/
test files. Do not ship an unused optional API, new schedule opt-in or changed
default. Preserve the complete experiment, including failures. Do not retime
this image seeking a better sample or reinterpret root-only speed as endpoint
headroom. This is different from, but reinforces the scope lesson of,
`2026-11-08-scalar-component-rys-axis.md`.

## Complete endpoints

Each card and size has four fresh native processes in control/candidate/
candidate/control order. Every process measures complete cold energy, analytic
forces and host export, then three warm calls from its own frozen cold density.
All first samples are retained. No oracle imports enter native production.

|Card / HF Direct|Control cold median/s|Candidate/s|Cold reduction|
|---|---:|---:|---:|
|A100 /96 atoms|16.198758|16.159716|0.2410%|
|A100 /48 atoms|6.127234|6.073499|0.8770%|
|RTX5090 /96 atoms|16.862727|16.903351|-0.2409%|
|RTX5090 /48 atoms|6.331561|6.349973|-0.2908%|

Every paired warm passes the unchanged3% protection. All120 native sample
pairs per card/case pass energy1e-8Eh and force1e-7Eh/Bohr. Every cold96
uses15 target iterations/Focks, cold48 uses10, and every warm uses1. Seven
preliminary native atomic Hückel Focks are separate. Quartet counters remain
invalid, not zero work. Both96 cohorts pass the authenticated independent
tighter GPU4 oracle228320: maximum E about1.402e-9Eh and F1.548e-9Eh/Bohr.
There is no new stronger48, displaced or final-default qualification.

These points do not support a material cold improvement. They do not close
the immutable README HF96 gap of roughly16s versus6s, establish all-A100
leadership, or authorize combining different images into a new64-case count.

## Why the microbenchmark was insufficient

The isolated64-primitive pipeline, including the extra barrier, improves by
1.44–3.99x on A100 and2.76–7.02x on5090 across2–5 roots and128/256 threads.
It is not an ERI, SCF or complete-endpoint workload. A100's actual production
HF96 baseline kernel names are
`generated_sm80_dppp_shell_class_fock_rhf_persistent_kernel` and
`generated_sm80_pppp_shell_class_fock_rhf_persistent_kernel`; they match the
changed source owner, not an unmodified SM120 or force companion. This join
uses the already acquired228690 census, not another profile run.

Offline Slurm16791 resource audit of the actual SM80 persistent functions:
DPPP registers168→104, stack96B unchanged; PPPP168→105, stack72B unchanged;
shared488→496B for each. These are **persistent** descriptors. The previous
scalar-axis note's96/80-register observations describe different streaming
functions and must not be substituted. Lower register use and root-only
speedup did not establish complete cold profitability. Static LOCAL0 is not
a dynamic-traffic or occupancy measurement. No exact removable root seconds
or atomics bottleneck is inferred from this negative experiment.

## Scientific and runner evidence

- 183 root/value/shared-owner host gates and135 profile/tuning gates pass;
  compiler structure checks587 modules with zero dependency errors.
- Root fixtures cover2–5 roots, all table/limiting boundaries and neighboring
  FP64 arguments, stride1/3/7, full/partial/empty masks, invalid lanes,
  disjoint writes, persistent CTA reuse and graph replay. Independent70-digit
  incomplete-gamma Boys moments0..2*nroots-1 use relative3e-12, with observed
  errors about3.2e-14 or less. Scalar parity is a separate gate.
- Slurm16783 root and16787 generated DPPP/PPPP RHF/UHF Fock fixtures pass
  independent Libcint, signed multiple primitives, coincident/reversed centers,
  screening masks, CTA reuse/replays, and all three CUDA sanitizers without
  errors/hazards. The same generated Fock fixture passes on A100228708.
- First build16781 fails on host-linker flag forwarding after compilation;
  cached16782 succeeds. First Fock16785 fails in control UHF because the new
  fixture omitted spin_offset; corrected16787 passes without changing production
  scatter. Both failures remain retained.
- A100 root228695 finishes its numerical/pipeline work, then exits13 because
  the existing sanitizer lacks its injection library. Native228708 finishes
  all eight complete endpoint populations and the Fock fixture, then exits1
  at the staged sanitizer's version check: host GLIBC_2.25 is missing. Those
  jobs are **not** sanitizer/whole-qualification passes. No stronger/displaced
  phase ran. The timed native populations are complete and preserved, and no
  benchmark is repeated to hide the later tool failure. Further sanitizer
  repair is not justified for an already unprofitable candidate.
- Native build16788 initially waits because mem0 requests the entire node;
  its same pending job is changed to192GiB, not resubmitted. Build16788,
  exact delta reconstruction16789, protected5090 cohort16790, resource audit
  16791 and CPU setup228707 succeed. Borrowed native source/runtime is restored.

## Provenance and retention

Control source50ca688e7f9dc3d51c05f9aa336fab14226372fcbe8f970536062753a87ff6e7,
ELF2977049443dd116779ad583f67da20c902fcede2fdb0ae721d2995d8aee147ff.
Candidate source3f91064437c3693f85fcf4f7a2060411cddd8e6abb855510c6e15727e0fb1142,
ELFa0aaa0bdc159592ff68acb1d6f1191c77544aec0174898db1305f2fcf24c73eb.

Raw source snapshots/owned-file hashes, removed code, original outputs/arrays,
all failures, compiler/cache receipts, analysis and strict-oracle reductions:
`.artifacts/hf-cooperative-roots-v1-20261108/`. A100 outer archive
13668b92486f176b0b468af2b1b36e676bf3e960852ab2f37905e883d456c8cd,
original native228708 archive
6222977bbe80daa3b16853a364fdd8fe0225244a5c29a1067a69afe317d1d8df;
5090 native16790 archive
5dc9bc4dba5b2be9dd9d6a54e06fb81fcc0ef855403624947009d49649135edd.
The original A100 root evidence is2ae7df955883d422d6583d8d2e52a88f674443a035a9f9e62c21f0f3b9dd7913.
Issue2976 is the external evidence/decision ledger; its progress comment is
6715654704. No release, external backup host, retagging, or evidence deletion.

## Revisit when

A different complete workload independently shows that this serial producer
has material critical-path cost. Do not revisit merely because another
root-only benchmark or static resource descriptor looks attractive. For the
current HF96 cold gap, investigate actual algorithmic/data-movement reductions,
not more noinline, register-cap or root interpolation scheduling tweaks.
