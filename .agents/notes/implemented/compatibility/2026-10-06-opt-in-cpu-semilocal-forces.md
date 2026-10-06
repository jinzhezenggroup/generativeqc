# Decision: preserve CPU semilocal energy defaults

Status: implemented
Date: 2026-10-06

## Problem

Promoting a force capability also changes the facade's default property set.
The bounded stationary owner admits at most 16 AOs and eight atoms, so ordinary
water/def2-SVP energy calls would newly fail after SCF. Capability-based planning
also charged energy-only callers the 256 MiB force allowance.

## Decision

Keep newly qualified direct all-electron CPU LDA/PBE forces explicit. Share the
composition admission predicate between capabilities and execution; VV10 and
external dispersion combinations remain outside this promotion. Preserve ECP,
DF and other methods' existing defaults and reservations.

Prepared semilocal CPU batches prospectively revalidate requested output
capacity against the same budget and provider selections. Only the serialized
force workspace changes. Publish the new plan after successful replay; budget,
native or generated-force failure leaves the accepted plan unchanged.

## Evidence and invariants

Native-free control-flow tests cover one-shot and prepared defaults, explicit
force routing, output-sensitive estimates, budget transitions, failed requests
and replays, additional resource owners, ECP/HF defaults and constructor cleanup.
They use dummy ABI outputs and do not claim numerical qualification. Existing
independent analytic/finite-difference tests remain unchanged. No force domain,
scientific kernel, numerical threshold or work budget is expanded.
