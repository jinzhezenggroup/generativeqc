# Decision: isolate borrowed LR scheduling without changing the primary owner

Status: implemented (acceptance harness; no production-policy change)
Date: 2026-10-10

## Problem

#1861 implements #1855's borrowed retained block domain. Its remaining
qualification must compare LR scheduling on one source/binary identity, not
reuse the pre-#1861 profile as indexed-domain evidence. Disabling the global
bounded Schwarz policy also changes primary preparation and full-range forces,
so it cannot isolate the LR mechanism.

## Decision

Use the Linux/ELF test-only adapter in
`benchmarks/experiments/issue1855-lr-domain/`. It interposes only the LR launcher
and the combined derivative consumer. The triangular arm clears the optional
domain at the LR launch; both arms retain the original full-range owner,
full-range scheduling, physical coefficients, density and screening semantics.
Check the primary owner's allocation counts, charged bytes and prefix before
and after every combined call. Do not add a production diagnostic ABI.

Separate two binaries and populations:

- The thin C++ adapter forwards all kernels to the unchanged main library.
  Only its uninstrumented complete returned E/F population supplies timings.
- The CUDA observer instantiates the frozen native kernel's existing Long
  specialization and post-screen class ledger. It records actually drained
  claims, launches and admitted shell/tile/AO/primitive quartets. Independent
  numerical gates must still pass. Its separate temporary scratch, synchronization
  and downloads preclude clean timing or production register/stack claims.

Optional value admission may legitimately omit a generated exchange owner when
shell AOT is disabled. The native range fixture therefore explicitly prepares
the stationary consumer's bounded force owner within the same combined 64 MiB
allowance, using already normalized systems. It must neither dereference an
absent value owner nor normalize the copied basis again. An admitted existing
owner is left unchanged.

## Invariants and rejected alternatives

- Keep indexed/triangular agreement, displaced independent RKS/UKS Cartesian and
  spherical through-f oracles, and the original tolerances.
- Preserve exact/one-byte-short/no-prefix budget edges and sparse/dense/empty
  claims. Empty admitted domains publish zero full/LR contributions.
- Keep the bounded triangular fallback; this harness changes no default.
- Do not replace actual post-screen work with nominal prefix products or infer
  whole-process memory peaks from allocation inventories.
- Require `DT_NEEDED libgenerativeqc.so.0`. An `RTLD_NEXT` wrapper without that
  dependency can fail when Python loads the main library with `RTLD_LOCAL`.
  NVCC can reorder paired `--no-as-needed`/`--as-needed` flags before libraries;
  use only the former at the observer link and inspect the resulting ELF.
- Compile objects through the verified cache launcher, then link separately.
  Direct compile-and-link `ccache` invocation is not evidence of a cache hit.

## Evidence and scope

Production is frozen at `15bc697009d191a88120607e0e50a15561d35d61`, tree
`7b32f9d8c2f547d78ed27ddae9c534c3c863c30b`, library SHA256
`7fc478aa78b15a1ee29fb8659c6121c8feb01f4f61aa917e59207d55af7c37c0`.
The library reports Release code generation for the device but the portable
`generic_cuda` policy; optional shell/stationary AOT is off. This is not an
installed-production/no-runtime-compilation qualification for the wider
semilocal or mixed-precision campaigns.

The targeted host schedule suite passes three tests. Slurm 7061 completes the
native budget/range gates and all four sanitizer tools with no errors/hazards.
Slurm 7072 also completes the nine budget/screening cases (including empty),
the sanitizers and thin-interposed indexed/triangular independent through-f
oracles. Its `srun` exit code is zero and pre-exit allocation receipt is retained;
the controller's completed record expired before it could be fetched.

The exact WB97M-V/RKS water-12 recipe uses 232 spherical def2-TZVPD AOs,
the moving original Becke-3 48×16×32 grid and the immutable matched independent
reference from the 2026-10-05 campaign. Gates remain 1e-8 Eh and 1e-7 Eh/bohr.
This water recipe is not OMol validation. Constructor/import/teardown are outside
the explicitly documented cold scope; preparation through returned host E/F
is inside it. Reference work and gate checks are outside native timers.

Artifacts and all failed/incomplete pilots remain under ignored
`.artifacts/issue1855-acceptance-20261010/`, mirrored at
`n1:/data/jzzeng/qc-issue-1856-acceptance-20261010/`. The first 45-minute paired
pilot has inadequate headroom after measured ~590 s cold and ~63 s warm calls.
Do not relabel its interruption as PASS or splice its rows into a fresh complete
campaign. Restart with an explicit finite 75-minute scheduler allocation.

Complete endpoint/work results are required before issue closeout. Neither a
compiled observer nor the partial pilot establishes profitability. Record
negative results as readily as positive ones, stratifying comparisons by the
physical GPU of each matched pair rather than treating different GPUs as one.

## Revisit when

A source-matched complete population identifies a reproducible profitable
selection domain, or production routing/ABI changes invalidate ELF isolation.
Qualify any replacement against the same independent scientific gates and retain
all unsuccessful observations. A diagnostic specialization cannot justify
promoting a changed production kernel or resource claim.

## References

- Issues #1855 and #1477; implementation PR #1861.
- `benchmarks/experiments/issue1855-lr-domain/README.md`.
- `tests/native/test_cuda_fock_provider.cpp`.
