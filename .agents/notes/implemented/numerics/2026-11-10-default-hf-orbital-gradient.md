# Decision: promote owner-bound Direct RHF orbital-gradient convergence

Status: implemented; promoted-image device, scientific and full-endpoint qualification complete
Date: 2026-11-10

## Decision

After reviewing the explicit-mode qualification, the user authorizes making
orbital-gradient convergence the default and opening a PR. This supersedes the
opt-in-only decision in
[the initial alignment note](2026-11-10-hf-reference-convergence-alignment.md),
without relabeling that note's historical source or observations.

Absent or empty `GENERATIVEQC_CUDA_RHF_CONVERGENCE` now selects the physical
restricted MO-gradient norm for standard-control, all-electron FP64 Direct CUDA
RHF. Explicit `strict` retains the AO-maximum policy. Either tighter public
energy or density control restores strict admission even when the environment
explicitly requests `orbital-gradient`.

Default admission must not turn previously supported UHF, ECP, mixed-arithmetic,
application-budget or physical-reference requests into errors or silently relax
them. These owners keep strict convergence when the control is absent. An
explicit orbital request still rejects unsupported owners. DF HF and KS retain
their independent existing policies. Python resource/checkpoint and captured
native-plan identities continue to bind the selected policy.

## Accuracy and publication boundary

The complete endpoint and independent observable gates are unchanged. This is
not a new 1e-7 maximum-force guarantee: the prior HF48/96 experiment shows both
native orbital mode and default GPU4PySCF exceeding that stronger diagnostic.
Strict overrides and independent tighter audit solves remain available.

GPU4PySCF settings must remain completely unchanged. README figures for the
new default require a complete source-matched cohort, not selective replacement
of the two previously slower Direct HF points. Retain accuracy failures and
 semantic work counts alongside endpoint timings. The qualification below tests
the promoted default itself, not the earlier opt-in image.

## Promoted-image qualification

The measured implementation is commit
`8c9acd542840f5c059ce866dc2e9f11c0aaf245d`, based on master
`d35ca77a4c7142f8d6d3f0cbc1e8e96718d9af30`. Its native source identity is
`d2ea46c7523cf8e5861be719d11522c2712711274e962af5f4ec4fa74916a5e0`;
the Release/SM120 ELF SHA-256 is
`f65eb17b4b48f83ea8d51b245fbdedf973315afa403086756e6cf35068f71381`.
Build job 17115 and qualification job 17120 ran through finite Slurm
allocations on node1/main with assigned RTX 5090 visibility. CUDA 12.9.1,
GCC 12, fast compile off, device link on and verified sccache 0.16.0 were
used; compiler commands and before/after cache statistics are retained.

The actual-device physical convergence/finalization tests, including
32/256-thread graph and non-graph controls, pass. The new image passes 267
Python tests; four optional CUDA-restart tests are skipped, not claimed as
executed. Five focused local host tests also pass. Independent native-only
HF48 guards against the retained tighter PySCF oracle give:

| Request | Physical Focks | Absolute energy error (Eh) | Maximum force error (Eh/Bohr) |
| --- | ---: | ---: | ---: |
| Absent default | 10 | 3.00e-11 | 1.82e-7 |
| Explicit orbital-gradient | 10 | 3.00e-11 | 1.82e-7 |
| Explicit strict | 14 | 1.00e-11 | 1.65e-9 |
| Energy tolerance 1e-12 only | 14 | 1.18e-11 | 1.65e-9 |
| Density tolerance 1e-10 only | 21 | 1.05e-11 | 4.24e-11 |

Default and explicit orbital mode agree in energy, force vectors and work
within FP64 reduction noise. All three modes pass the unchanged common
observable gates; the default does **not** pass a stronger 1e-7 force diagnostic.
Default UHF remains supported and agrees with strict fallback. Malformed
requests, explicit unsupported orbital mode and changing policy after
resource preparation are rejected. These untimed guards are separate from
the complete README timing population.

### Acquisition harness correction

The outer SSH build launcher accidentally used bare `python` in two process
substitutions on a shell without that executable. Their empty outputs made
`cmp` vacuously succeed. The build itself independently checked the embedded
identity against its own source inventory, but that did not authenticate the
local expected identity. An absolute-path Python recheck subsequently
confirmed the expected local source identity and measured ELF above.
`identity-recheck.json`, both original shell errors and the successful
independent check are retained; the offline full-vector audit independently
binds the expected identity and clean measured-commit source archive. This is
a harness correction, not evidence of a source/image mismatch or a production
solver fix.

## Evidence retention

The new publication retains every endpoint timing, semantic work count,
unrounded independent error and original receipt hash. Full vectors, raw
receipts, build commands, cache statistics, qualification guards and the clean
source archive remain in the disclosed ignored acquisition directory. Its
scalar replay does not pretend to repeat vector subtraction; the separate
full-vector replay does that against the unchanged historical independent
oracles after exact scientific-input and explicit environment checks.

The fixed evidence budget is not raised. To make room, the related Oct26
derived timing/work tables and acquisition-conditions Markdown are compressed losslessly, with their
public paths retained as forwarding pages. Original raw measurements,
independent references and the pinned historical README remain unchanged.
Only attachment storage metadata and the storage-aware reproduction reader
change; historical numerical/decision leaves are preserved. The historical
replay with its pinned Matplotlib 3.11.2 reproduces both original table byte
streams, all gates and original figure hashes. New figures use the separately
recorded Matplotlib 3.10.9 environment.

## Complete default-cohort outcome

The [Nov11 publication](../../../../benchmarks/results/readme-rtx5090-20261111/README.md)
retains all 64 cases: Direct wins 32/32 cold and warm; DF wins 30/32 cold and
32/32 warm. Independent unchanged gates qualify 58/64 pairs for each engine;
the six existing DF-HF/default-grid failures remain visible. All 1024 selected
cross-engine force-vector comparisons pass. Qualified timing wins are 57/58
cold and 58/58 warm, not a claim that every timing or scientific case passes.
Direct HF48/96 use 10/11 cold physical Focks. DF HF96 and r2SCAN-3c96 remain
slower cold points; the native policy change does not affect their DF owners.

Original worker 17119's final GPU4PySCF DF wb97m-v48 process fails with
`cudaErrorLaunchTimeout` on its second warm call, then aborts during cleanup
(exit -6). The campaign exit remains 1 in the original receipt. Exactly one
bounded paired retry, job 17123, succeeds without changing settings, source,
inputs, engine order or gates. Both engines are reacquired on the retry's
assigned GPU; the native original is not paired with a new reference on
another card. All original receipts and six displaced completed calls remain
retained, including compact unrounded timing/work/error samples. The complete
selected cohort contains 128 processes/512 calls; acquisition totals are 130
processes, one failed process and 518 completed calls. The full-vector audit
authenticates original execution order and failed status before admitting the
single pair replacement, and prohibits replacing any successful pair.

The source-bound full-vector audit and compact replay reproduce the same
timing and scientific counts; normal and optimized Python retain the audit
gates. No GPU4PySCF default is modified, no slower/accuracy-failing selected
case is filtered and no earlier image's timing fills a new-source point.
