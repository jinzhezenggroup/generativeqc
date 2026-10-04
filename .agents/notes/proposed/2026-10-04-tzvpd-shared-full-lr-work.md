# Priority: remove repeated full/LR source work in the TZVPD integral endpoint

Status: proposed; no shared-source endpoint gain qualified
Date: 2026-10-04

## Evidence and scope

The user explicitly prioritizes integral values and derivatives over further
VV10 tuning. The source-bound [12-atom profile](2026-10-04-tzvpd-warm-integral-diagnosis.md)
supports this: full J/K values, LR K values, full J/K derivatives and LR
derivatives sum to 89.49% of the profiled endpoint. Total orders 5--8 account
for 76.04% of **value-kernel** duration. The force-class fractions remain
unknown; do not assign all derivative duration to those orders. Full J and K
already share work. LR is not a second Coulomb J calculation.

Treating that kernel-duration fraction as serial endpoint time, with all other
costs unchanged, the Amdahl model gives 1/(0.1051 + 0.8949/s): 1.81x for s=2
and 3.04x for s=4. These are conditional model estimates, not measurements,
critical-path proofs, or promises that a particular transformation achieves s.
Classification, new memory traffic and resource regressions must be charged.

## Existing work is not the proposed result

`direct_jk_kernels.cu::canonical_jk_kernel` already consumes screened pair rows
and angular buckets. `direct_source_contraction_cuda.py` already dispatches
exact shell-class templates. The missing result is lower actual setup,
recurrence and resource cost, not merely naming another specialized kernel.
`direct_jk.cpp::enqueue_cuda_direct_rsh_values_device` currently admits the
bounded value route only; canonical full and LR values consequently remain
separate. Do not promote the independently rejected through-f bounded route
as a shortcut to fusion.

PR #1836 head fa1e8f49d was read, including its compiler pass IR and native
launcher. It explicitly performs five screening scans, specializes existing
4/5/6 workers, retains generic 7--12, and leaves LR unchanged. Its cost is
5S + sum(C_o), not one classification followed by class-specific draining.
It has a separate owner and must not be duplicated or treated as evidence
that one-pass classification already exists. The earlier thirteen-scan
[angular experiment](../rejected/2026-10-04-tzvpd-angular-force-schedule.md)
is measured negative evidence, not a starting point for unchanged retuning.

## Work contract for the next structural experiment

For full/LR admitted sets F/L, primitive-product count P_q, common geometry
and Hermite setup G_q/H_q, and range-specific recurrence/contraction costs
R_q^F/R_q^L, the value-path accounting is:

    current = S_F + S_L
              + sum_(q in F) P_q (G_q + H_q + R_q^F)
              + sum_(q in L) P_q (G_q + H_q + R_q^L)
    proposed = S_union + Q_class
               + sum_(q in F union L) P_q
                 (G_q + H_q + [q in F] R_q^F + [q in L] R_q^L).

The symbols denote work categories, not interchangeable operation counts or
FLOPs. Actual molecular counts remain null until instrumented. Each consumer
keeps its exact admission mask, density weights, output and radial moments;
the union must never replace its screening predicate with a looser published
scientific result. Only common preparation is shared. Avoid simultaneously
retaining two large Coulomb auxiliary arrays when sequential root consumption
can retain just geometry/Hermite data; measure the resulting lifetime and
register/local-memory cost rather than presuming it is smaller.

For generic derivatives above order six, include the existing U_q-1 unique
center seeds in both current and proposed work. Existing all-center 4--6
consumers must not be counted as newly optimized. Sharing full/LR preparation
does not by itself remove center derivatives or radial recurrence work.

One-pass classification must use bounded chunks and retain admitted task IDs,
so a histogram/prefix/scatter over those IDs does not rescreen the full domain.
Its memory is O(Q * task_bytes + class_metadata) for explicit chunk capacity Q,
not an unbounded list of all shell quartets. Exact task format, capacity,
overflow continuation and bytes need implementation-level proof; no numerical
memory estimate is claimed yet. Retain the old bounded path when admission or
allocation fails, without missing or double-counting any admitted source.

## Qualification and cold relevance

Count actual candidate/admitted tasks and per-class primitive/source work;
compare full/LR admission masks before comparing timing. Use independent
fixed-density full/LR J/K and force oracles, both spins, disabled sources,
repeated/distinct centers, diffuse/f functions, screening boundaries, budget
edges and sanitizers. Then compare same-source/same-GPU complete original and
displaced cold plus five warm calls, preserving 1e-8 Eh / 1e-7 Eh/Bohr gates.
Linked resource data and dynamic traffic must support any occupancy explanation.

Value work repeats across cold SCF iterations, whereas final force acceleration
alone cannot close that gap. Keep the full preliminary-density lifecycle
separate; TZVPD12 LDA already regresses cold by 15.12% and remains off.
The value-only reachable recurrence and independent Hermite contraction in
PR #1839 are bounded preparatory ablations, not the completed shared full/LR
architecture. Their measured gains must not be added to this model.
