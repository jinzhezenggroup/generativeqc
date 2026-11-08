# Experiment: HF48 cold proposal instead of globally tighter controls

Status: proposed, explicit trial only; no default promotion
Date: 2026-11-08

## Diagnosis

The one-public-control experiment in
`benchmarks/results/hf48-control-isolation-20261108/` reuses authenticated
source33db42ef / ELF60f40b15, not the later integrated PR head. A100229107
and Slurm16820 each complete all four fresh-process conditions and their
three own-density frozen warms. Untouched default and screening-only`1e-14`
both miss the unchanged independent`1e-7` force gate at about`1.8184e-7`.
Energy-only`1e-12` and density-only`1e-10` pass at ordinary`1e-12` screening,
with14 and21 target iterations instead of10. Reused full-Fock CPU16815 is
converged independently, with orbital-gradient norm`4.45e-11`.

This supports a convergence-family explanation for this HF48 geometry rather
than an integral-screening floor. It does **not** distinguish the public energy
criterion from the stricter internal physical gate: either tighter public
energy/density request also changes the physical ceiling from`1e-5` to
`min(1e-8,density_tolerance)`. Do not assert that energy alone is the sole cause.
All32 full force arrays, work, controls and raw logs are retained; the default
failures do not disappear when tighter requests pass.

## Bounded new hypothesis

HF48 has384 spherical AOs and is outside the existing at-least512-AO cold
proposal. That existing larger-owner algorithm already requires strict
physical stationarity in its exact target tail. An explicit experiment admits
the same proposal down to384 AOs and keeps automatic admission at512. All
existing closed-shell, single-item, through-d, cold, ordinary-Direct, sufficient
allowance and resource exclusions remain. There is no card/product-name gate,
CPU seed, oracle work, new approximation in publication, or new public switch.

The intended gain is replacing expensive early strict Focks with the existing
bounded screened proposal while recovering stricter final physical acceptance.
It is not blindly tightening all defaults and paying the14/21-iteration cost.
This is distinct from the rejected384-AO prepared/delta expansion: it borrows
the existing screened-Direct stage and does not instantiate that prepared owner.
The old rejection remains valid in
`../rejected/2026-11-08-hf-fixed-coarse-delta.md`.

The frozen candidate is based on actual upstream master`b29247c60` integrated
in PR head`d82e4a1aa`, excluding the13 older adaptive dirty paths. Its source
identity is`9e292fde3d8b42c773e569f63d870dd479698ae0270591609118b0b7a17021eb`.
It has not earned automatic admission. A stale fork's unrelated master is not
an upstream update; the authoritative origin master was fetched and has no
new commits beyond the already integrated`b29247c60`.

## Acceptance and fallback

Require complete cold E/analytic-force/host-export and every paired warm on
both real cards, unchanged strict independent force gates, scientific/control
provenance, resource/allowance/lifetime refusal, and no protected5090 regression
before any default expansion. Preserve original automatic512-AO and explicit0
fallbacks while screening. Reject/remove the new experimental domain if its
complete endpoints are not profitable; do not leave another unqualified opt-in.
The overall non-CBS A100 README objective remains open, especially HF96.

## References

Issue2976; draft PR3048; one-factor capsule above; local immutable candidate
and build/endpoint receipts under`.artifacts/hf48-coarse-proposal-20261108/`.
