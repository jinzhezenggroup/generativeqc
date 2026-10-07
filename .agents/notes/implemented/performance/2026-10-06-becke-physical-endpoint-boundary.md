# Decision: retain the Becke primitive as an unpromoted physical experiment

Status: implemented
Date: 2026-10-06
Related: #1894; draft PR #1996

## Problem

Synthetic source-owner qualification and a smaller reverse panel do not establish
a complete molecular performance win. The retained coefficient route changes
reverse/gather dataflow but not the dense pair domain. Physical endpoints must
include solver work, source-owner setup and moved-geometry rebinding, with
separate diagnostic profiling rather than subtraction of intrusive intervals.

## Evidence

Finite n1 `main/gpu:5090:1` jobs 6173, 6184 and 6193 finish with zero foreground
Slurm/SSH and run-receipt exits. Job 6173 additionally has a retained controller
`COMPLETED`, `ExitCode=0:0` receipt. Slurm accounting storage is disabled and the
controller has purged the other two terminal records; do not invent accounting
results for them. Preserve their complete outputs and separate zero-exit receipts.

The reviewed publication is
`benchmarks/results/becke-physical-endpoints-20261006/publication.json`, with
tables and interpretation in its `summary.md`. Eight native protocols retain
96 complete energy/force observations and full actual SCF histories. Two fresh
independent GPU4PySCF reference protocols retain 24 observations. These are two
molecular sizes with repeated calls, not 96 independent molecular test cases.
The shared `tools.generativeqc_validation.record.load_record` reader restores
`observations.json.gz` from checksum-bound, named protocol companions. Verify
the reconstructed ten protocols against the original unsplit observations;
all samples, histories, vectors and summaries remain unchanged. Host evidence
qualification reports 79 passed; full-PR retention and configured hooks pass.
Maximum independent energy/force errors across clean and intrusive calls are
`1.0868461686186492e-10` hartree and `3.218909860880359e-11` hartree/bohr, under
the unchanged `1e-8`/`1e-7` gates.

Clean warm medians, phased/primitive, are 11.665443/11.202456 seconds at 48 atoms
and 39.788979/39.998613 seconds at 96. Moved-warm medians are
11.236248/11.177308 and 39.894366/40.084957. Arms are grouped, with reversed order
at 96, not an interleaved paired trial. Retain all samples, including the two
slower 48-atom baseline warm calls; no resolved speedup or promotion follows.

Cold work is 27/25 iterations/Fock builds at 48 and 30/33 at 96; moved work is
12/12 and 14/16. Never normalize those histories or attribute their complete
wall differences to source speed. All warm/moved-warm calls actually perform
one iteration and one Fock build. Cold totals include outer preparation. Moved
outer preparation is zero in the existing protocol, but internal source-owner
geometry rebinding and preparation remain measured inside the endpoint.

Separate intrusive original-warm Becke totals, phased/primitive, are
762.287715/760.886594 ms at 48 and 3593.864880/3734.906351 ms at 96. Totals are
medians of per-call sums, not sums of component medians. The 96-atom reverse
interval improves from 388.357374 to 356.356984 ms while gather worsens from
415.302404 to 547.373646 ms. Keep all seven intervals and moved-warm observations;
do not add them to clean wall time or fold AO/XC work into this mechanism.

## Decision and invariants

Keep the primitive opt-in and default off, with the generic/phased fallback and
all losing publications. Dense primal/reverse visits and phase launches do not
decrease: at 96 atoms each phase visits 10,758,389,760 point/pairs, with 64,512
Becke launches per force. Extra gather direction reads offset the smaller
logical reverse panel. These are logical panel values, not measured hardware
transactions. Branch-dependent log/sqrt counts are unavailable, not inferred
from visit counts. No privileged profiler workaround is warranted.

The original core source revision plus dirty patch is retained, not relabeled
as a later clean build. Source reconstruction verifies all 1400 scientific/build
input hashes. Ccache launchers are verified on all 382 compiler commands; retain
before/after shared statistics without assigning aggregate hits to this owner.
Initial cold owner lookup/construction remains about 14 seconds. No fresh
compiler-only total or per-call generated artifact binding was instrumented.
The reproduction shell passes syntax checking, but the entire reproduction
recipe was not rerun as another campaign.

## Revisit when

Consider promotion only after a materially different, bounded dataflow reduces
the complete bottleneck and passes at least five genuine interleaved pairs,
independent numerical gates, compilation/memory accounting and a full acceptance
audit. Do not bypass the once-before-topology native configuration guard to
toggle a live owner. Unchanged log traversal, dense/xyz caching, broad-zero
skipping and losing fusion are not new experiments. Keep #1892 Direct work and
#1893 AO/XC work separate. This progress does not close #1894 or mark its goal
complete; PR #1996 remains draft.

The independent physical consumer gates are documented separately in
`.agents/notes/implemented/numerics/2026-10-06-becke-physical-force-gates.md`.
