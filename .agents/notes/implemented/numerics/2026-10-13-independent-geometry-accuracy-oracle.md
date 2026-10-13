# Decision: separate geometry accuracy from reference timing completeness

Status: implemented
Date: 2026-10-13

## Problem

The README WB97M-V 48-atom reference population exhausted its original 1800 s
limit before returning moved-geometry forces. Native moved results therefore
could not pass an independent accuracy gate, even when they finished. Extending
the reference timeout or filling its timing population with separate runs would
change the benchmark contract.

## Decision

Accept an explicitly separate, independently reconverged geometry receipt for
accuracy validation. `benchmarks.readme_geometry_oracle` starts GPU4PySCF from
its default atomic guess and consumes no native density, orbitals, energy or
forces. It checks the exact retained scientific protocol and basis. Each run
still uses a finite Slurm allocation; its duration is diagnostic only.

The native runner binds each supplementary receipt by filename and SHA-256. The
renderer rechecks those bytes, complete finite E/F arrays, convergence residuals,
the protocol, and the basis before qualifying native phases. Supplementary
receipts cannot replace an existing geometry baseline or contribute timing
records to the reference population.

## Invariants

- Keep the original 1800 s endpoint population and its timeout outcome.
- Keep the full grid, full-density Fock rebuild and original SCF tolerances.
- Apply the unchanged independent 1e-8 Eh / 1e-7 Eh/Bohr E/F gates.
- Preserve source and driver hashes, scheduler provenance, and complete arrays.
- Never imply a reference speedup or completed reference timing population from
  an accuracy-only solve. An absent receipt remains unchecked.

## Evidence

The 48-atom moved geometry independently converged in 45 SCF iterations with
energy change 4.55e-13 Eh, density RMS 9.45e-14, and orbital gradient 7.52e-11.
Its diagnostic solve took 1180.014 s inside Slurm job 8174; the original timing
run remains timed out. Host tests reject mismatched protocols/bases, incomplete
or nonfinite arrays, failed residuals, native-seeded declarations, duplicate
geometries and modified receipt bytes. A renderer test qualifies native phases
while retaining the reference timeout and incomplete moved timing population.

The original one-off driver and receipt must remain with publication evidence;
the maintained driver added here supports future reproducible independent runs.
This change does not establish a 96-atom reference result or improve runtime.

## Source-review corrections

Bind the full-density Fock wrapper only after optional `density_fit()`. The
GPU4PySCF DF constructor copies instance attributes, so a wrapper attached to
the original direct engine otherwise retains that direct owner. Rewrapping the
copied closure does not fix the owner and can mix direct SCF with fitted forces.
A host regression exercises both direct and fitted driver execution, including
the new-owner copy behavior; it fails on the original fitted driver.

The runner and renderer parse and hash one receipt byte snapshot. Both enforce
same-directory resolution, including symlinks; the original renderer accepted
an escaping symlink in the negative control. Malformed source/driver digests
and nonintegral iteration counts are rejected. These checks preserve declared
provenance and byte integrity, not independent authentication of an acquisition.

The separate historical one-off driver and 48-atom receipt are not included in
this source PR. This host-only repair does not establish how that earlier solve
was configured, requalify its result, or qualify a corrected-driver GPU run.
Preserve the original acquisition bytes for any later evidence review.

## Revisit when

A complete reference population becomes available under the original limit.
Retain these historical receipts even if a later complete campaign supersedes
them in the current figure.

## References

Issues #2417 and #2426; original benchmark PR #2401.
