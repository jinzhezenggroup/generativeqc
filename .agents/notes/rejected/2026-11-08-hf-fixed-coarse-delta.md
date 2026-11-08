# Decision: reject fixed coarse incremental HF screening and unqualified expansion

Status: rejected
Date: 2026-11-08

## Problem

The complete A100 README audit has HF Direct96 cold16.277175s versus
GPU4PySCF6.018824s. The seven strict-tail Fock applications, especially K,
dominate; diagonalization is not the ten-second explanation. The named CUPTI
228261 census reports DPPP2.100s, exact-class compaction1.736s/1520 launches
and PPPP1.677s over cold plus one warm, with no collector loss.

## Decision

Do not ship the experimental HF prepared owner, fixed coarse delta or its
384-AO expansion as defaults or leave an unprofitable opt-in in production.
Retain the measured negative results and original independent tests. This
does not reject an adaptively error-controlled incremental operator, which
would be a new numerical experiment requiring new independent qualification.

The experimental owner borrowed existing FP64 MD-J and generated/raw-K with
512MiB optional storage and explicit refusal, graph-before-source/output
destruction and finite-result propagation. It introduced no new ERI formulas,
DF approximation or reference-engine dependency. These ownership properties
were necessary, but insufficient for complete-endpoint profitability.

## Evidence

- Same baseline ELF, LSF228266: ordinary16.0550s cold/2.2649s warm;
  all-primary streaming21.8166/2.6525; J/K both Rys16.1887/2.2779. Reject both.
- Prepared v4, source`0070d43a537e834f29eca4cd455ad2570faa3f790c6341404f7f83a927c6003e`,
  ELF`316f2fc741f75f6f0a2c8c51359a32e09cfa3e8f868e6597062f9723b30fb1cc`:
  A100228299 off16.051394/on13.205568/ref5.751552s; target15/14/10 steps.
  Cold improves17.73%, but the first warm regresses3.37% even though the
  median regresses only2.83%. Do not hide the failed per-warm guard.
- RTX5090 Slurm16744 off16.797722/on15.015934s; warm medians2.684733/2.686993.
  Independent tighter GPU4 HF test228320 has six passes: original/displaced,
  own-density warm, checkpoint, resource refusal/release, iteration allowance
  and default force gate. It qualifies v4 numerics, not later delta changes.
- Named228328: prepared device347641252 bytes, setup about0.745s; seven
  strict raw-K applications about6.467s, DPPP1.935s/PPPP1.705s. K remains large.
- Fixed delta v1, source`508997db56c2be46c47debd3a7a6ec342eb067dc9fcb0419a54278129625a6a0`,
  ELF`c6f0f7a9ca1e98098789fc53afd931cb57b0b5a0d7d1b28408e0d6c62aa54574`,
  normal Release CPU Slurm16750; LSF setup228358/screen228359 complete.
  HF48 off/prepared/delta/reference cold6.182503/6.646382/9.415416/5.200494s,
  target10/10/25/10 iterations. The delta arm does11 full+14 delta builds.
  HF96 cold16.107916/13.143466/20.920287/5.671662s, target15/14/27/10 steps;
  delta11 full+16 delta builds. Warm medians2.271334/2.322102/2.388707/1.468192s.
  HF48 ordinary/delta maximum force difference1.818e-7Eh/Bohr also fails the
  stricter1e-7 native comparison. HF96 maximum difference is about1.6e-9.

The fixed effective cutoff near8e-8 saves screened delta work but produces
a late error floor/history pollution and excessive full-energy refinement.
Restoring J/K scaling before adding H avoids catastrophic cancellation; it
does not remedy the convergence or performance failure. Reducing only a
kernel time is not acceptance. Extending the prepared pipeline to384 AOs
already regresses HF48 by7.5% without delta, so reject that domain too.

## Retention and reproducibility

All timings include constructor, preparation, SCF, analytic forces and host
export. Warm calls freeze each engine's own cold density. Defaults, strict
physical publication gates and the oracle-isolated native process remain
unchanged. Reference work is external and untimed by the native endpoint.

Raw local campaign directories are `.artifacts/readme-a100-20261107/`,
`.artifacts/hf-prepared-fock-v4-20261108/` and
`.artifacts/hf-scaled-delta-v1-20261108/`; frozen source archives, failed
build/runner receipts, strong tests and rejected patches are retained there.
The diagnostic archive SHA256 is
`86253a51436a4306bb584ec20568ba60a13ea5e8758ca1bc55fdeaa937e37d6f`,
the v4 followup archive
`042390de2b4e17131e31dd23334256b17cacd597782f4d0e6527bae8f63ef992`,
and the scaled-delta archive
`c190d91dfdb4075c3c9cc851ef36425351866cb3b568a2c4a3c98b1ad6ba8265`.
Issue2976 preserves the decision, job/image identities and negative samples.
The complete64-pair baseline remains in
`benchmarks/results/readme-a100-20261108/`; none of its losses is rewritten.

## Revisit when

A new proposal removes substantial exact K work, or makes the delta error
budget progressively vanish at the original physical target, and passes
independent stricter HF gates plus every complete warm/5090 regression guard.
Do not rerun the fixed-cutoff experiment merely with different timing order.
