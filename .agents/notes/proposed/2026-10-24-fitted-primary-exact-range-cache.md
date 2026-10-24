# Candidate: reuse exact LR values beside a fitted primary J/K owner

Status: proposed; independent composition and endpoint qualification pending
Date: 2026-10-24

## Diagnosis

Source `1ccc53419acd4c22d9496178a65fffd83f26d6b1` completes stock water96
omegaB97M-V DF cold in 676.154 seconds and Direct in 303.948 seconds with the
same 28 SCF iterations (Direct also reports one rebase Fock). Both pass the
independent complete energy/full-force gates. These matrix observations are
not repeated GPU4 populations.

A separate finite-Slurm DF cold/warm Nsight trace records about 426 seconds
in canonical/materialized exact range J/K kernels over 29 applications.
The existing phase-source admission unconditionally excludes a fitted primary,
even though its separate long-range operator remains exact. CPU sampling at
the subsequent XC entry point includes waiting for this earlier stream work;
it must not be interpreted as exclusive XC arithmetic. Raw traces and complete
receipts remain in Stage24's ignored `stage24-cold-nsys` artifacts.

## Candidate and invariants

Allow the existing explicit `GENERATIVEQC_KS_RANGE_RESIDENT_VALUES=1` selection
for strict restricted SPD fitted full-range J/K plus its separate exact LR owner.
Automatic selection, higher-angular sources, fully fitted LR, mixed precision, chunks and public
resource ledgers retain their previous policies. All existing VV10/operator,
geometry, screening and source audits remain in force.

The optional source is built and consumed on the primary KS stream. Ordinary
fallback retains the separate range-provider event handoff. The source owns
only immutable ERIs, not D, a fitted factor, or a final response lease. Reuse,
policy changes, force-memory retirement and geometry replacement retain the
existing audited range-source lifetime. Warm calls never prepare a new source.

Count both providers' device storage and host observation/preparation before
optional admission. Keep the existing ceiling and actual free-device reserve.
When no bucket mask is requested, admit only complete angular buckets fitting
that allowance; every unretained bucket uses the original streamed exact
consumer. No allocation cap or scientific approximation is changed.

## Qualification

Extend executable source-selection and replay tests to fitted-primary owners,
including explicit full/partial storage, capacity refusal, automatic fallback,
changed geometry and post-preparation policy invalidation. Independent exact
range matrices, fitted molecular E/F references, constrained resources and
sanitizers remain required. Repeated complete endpoints, including preparation,
rebase work and forces, decide whether this composition is useful.
