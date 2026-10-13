# Decision: restore HF/DF coverage and connect complete endpoint medians

Status: implemented
Date: 2026-10-13

## Problem

The initial README refresh omitted HF and DF from the inline comparison. Its
renderer also replaced a warm median with NaN whenever repeats had different
SCF work. This made successful GPU4PySCF endpoints look missing and prevented
readers from following size trends.

## Decision

Extend the shared complete-endpoint runner with actual RHF and explicit DF
selection. Direct and DF retain separate references; DF uses the identical
cc-pVDZ-JKFIT snapshot in both engines and includes auxiliary/metric response.
The reference full-density wrapper binds after `density_fit()` so it cannot
retain a bound method on the original direct owner. Native DF uses default
planning, without historical packed-B or response-budget overrides.

Connect the median of every accepted complete endpoint population. Keep the
full min/max range, individual crosses for differing SCF work, and all observed
iteration/build pairs. Only incomplete or failed endpoints break the line.
These statistics describe requested endpoint latency, not fixed-work kernel
cost or iteration-normalized speedups. This supersedes the mixed-work plotting
rule in [the original publication note](2026-10-13-readme-dft-cold-warm.md).

## Reference audit

A separate finite Slurm audit of GPU4PySCF 1.8.1 PBE/48 reproduces warm iteration
counts 4, 5, 4, 4, 4 with a byte-identical frozen input density. Repeated potential
evaluations on that density differ by up to 3.553e-15. The stock SCF implementation
creates a fresh DIIS object per call and requires both energy and orbital-gradient
thresholds. Recorded energy differences near the strict 1e-12 Eh cutoff can
therefore change the stopping iteration; every audited replay converged.
No density mutation, progressive warm-seed refinement or incorrect force result
was established. Do not change reference DIIS or tolerances to smooth the plot.
Audit timings remain outside benchmark populations.

## Evidence and boundaries

[The measurement report](../../../../benchmarks/results/readme-dft-20261013/README.md)
retains the measured sizes, source/library identities, all 12 endpoint checks
per successful point, negative capability probes, and the reference audit.
Do not promote a converged energy when analytic forces or independent numerical
gates fail. New HF/DF runs use the force-memory-fixed binary; older direct DFT
points keep their actual source identities.

The new summary, source-campaign and force-memory records use lossless XZ
compression to stay within the unchanged evidence budget. Earlier gzip bytes
remain in the existing Git history, and no measurement, vector or diagnostic
field is removed. The separately reviewed upstream density-fixture storage
repair remains intact. Builds and full trial logs remain ignored.

## Large DF endpoint stop and follow-up

The 96-atom PBE/r²SCAN calls exposed a default streamed-DF cliff: their cold
endpoints eventually returned in 1225/1222 seconds, including 628/618 seconds
in integral derivatives, versus approximately 0.4 seconds for that interval at
48 atoms. Both cold calls pass the scientific gates, but the populations do not
complete. PBE naturally exhausts its 1800-second limit in warm/0; r²SCAN is
cancelled at the user's request in warm/0. Preserve these distinct outcomes and
all partial records, without using them as successful chart points. An attempted
finite scheduled gdb attachment was denied by ptrace; it supplies no stack-based
evidence. Do not repeat a full timeout just to reconfirm this symptom.

[#2411](https://github.com/jinzhezenggroup/generativeqc/issues/2411) tracks the
resource/admission and repeated-work investigation. Separate issues #2412–#2417
track the B3LYP DF force failure, WB97M-V DF accuracy failure, two DF capability
gaps and two large-system endpoint/reference blockers. The report links each
issue. The README force-memory fix and connected medians do not resolve those
scientific/runtime findings; cancellation is not a numerical failure or proof
of deadlock. Future repairs require new source-bound endpoint qualification.

## Revisit when

New default DF force domains pass the same independent gates, or a new
GPU4PySCF version changes the observed reference convergence behavior. Neither
an updated plot nor a storage conversion requalifies different scientific code.
