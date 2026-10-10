# Decision: qualify automatic canonical reuse through order nine

Status: implemented
Date: 2026-10-11

## Problem

The [six/seven decision](2026-10-10-bounded-canonical-multi-packet-reuse.md)
leaves orders eight/nine as roughly 42.5% of its first-density 12-atom source
time. Extending that consumer without proving complete angular bounds would
silently omit components. Source wins alone also cannot justify promotion of
a production default.

## Decision

Extend the same immutable-cache, strict-FP64 canonical consumer to orders
eight/nine, automatically and without a method, basis or size selector.
Keep the compiler's recurrence and scatter mathematics. Under the explicit
s/p/d/f bound, dddd has 1296 components and fddd has 2160, requiring six and
nine slots per 256-lane CTA. One CTA owns all components and shares one
recurrence per primitive product. All lanes publish and retire together.

Five prefix planes cover the newly reachable second pair order four. Blocks
(6,2)/(5,3)/(4,4) and (6,3)/(5,4) use disjoint first-order segments; equal
buckets retain their sorted triangle. Exclude impossible pair orders before
indexing the eight-entry offsets table. Charge the extra plane and preparation
peak to the same optional lease, preserving immutable owners and MD-J priority.
Missing capability, index storage, budget or an allocation keeps the incumbent.

## Rejected alternatives

- Eight slots for order nine, based only on ffdp's 1800 components. fddd needs
  2160; eight slots would omit its last 112 components. Qualification uses the
  production nine-slot consumer and all high-order shell compositions, not
  the legacy forty-slot whole-shell test.
- Use the previous force implementation for one timing arm. Both new arms and
  unmodified master share fetched `ec04f1301`, including mixed-f force domains.
- Relabel those timings after upstream advances. Integrate on `b3a0eb2cf`
  separately and rerun native/Libcint/primitive/sanitizer/physical gates. Its
  full-Coulomb overwrite change affects the shared numerical owner, so source
  compatibility cannot substitute for real-device acceptance.
- Promote ten through twelve merely because the generator supports them. Their
  larger slot bounds need their own resource and endpoint evidence, and their
  remaining share is small compared with lower-order sources.
- Reenable the already rejected generic order-three/four 256-lane consumer.
  Their smaller domains need a different scheduling design, not the same
  schedule hidden behind a default or molecule-size switch.

## Invariants

- Preserve exact component-level AO Schwarz products, raw primitive coefficient
  multiplication/traversal, full/SR/LR identity and positive independent J/K.
- Preserve RHF/UHF and Cartesian/spherical projection, cache identity/lifetime,
  optional-allocation rollback and MD-J preparation precedence.
- Do not change other orders, compensated/fixed/resident/paired consumers,
  mixed precision or forces. Production must not consult a reference oracle.
- Keep complete endpoint timing separate from selected sources and private
  resource attributes. AO- and shell-quartet outer counts are different units.

## Evidence

The [source-matched receipt](../../../../benchmarks/results/wb97mv-canonical-orders89-20261010/README.md)
owns the frozen three-arm experiment, newer integration, raw archive inventory
and source/binary hashes. It uses the complete def2-TZVPD README water proxy,
not OMol25-distribution samples, with unchanged strict WB97M-V gates and a
4 GiB incremental force budget. Compilation and artifact caches are preserved.

Matched complete 12-atom E+F falls from 344.358152 to 248.930230 seconds after
adding eight/nine (27.71%); unmodified frozen master takes 529.333544 seconds.
Every arm performs 21 iterations/Focks, with matching AO/force work and traffic.
Three alternating 3-atom repeats give medians 11.980636/11.821136 seconds;
every sample passes the independent gate with 15 iterations/Focks. The
12-atom candidate errors are 5.230e-12 Eh / 2.818e-11 Eh/Bohr.

All 56 first-density 12-atom sources take 12.021961/7.511317 GPU seconds.
Selected order-eight/nine 96-atom sources take 304.085949/81.507865 seconds;
six requested matrices differ by at most 4.253e-14. This is not a complete
96-atom endpoint or independent physical acceptance. Local memory is not zero:
the frozen restricted kernels use 768/896 local bytes and 10264/12024 shared
bytes, with 146/132 registers. Newer integration resources are retained too.

Four native suites, eight independent Libcint cases, exact primitive-work
qualification and four persistent sanitizers pass on both bases. The newer
base's complete 3-atom first-use physical gate passes; its time is not pooled
with matched repeats. Native remains 1.362x slower than the independently
seeded full-Fock GPU4PySCF reference, not a stock-default reference ratio.

## Consequences and revisit

The default covers a larger complete component domain with finite register and
shared state, while keeping the optional geometry/index lease bounded. Revisit
if s/p/d/f bounds, cache provenance, device resources, primitive traversal or
supported range/precision contracts change. The next source target is dedicated
lower-order scheduling: three/four/five are about 55.5% of the qualified
12-atom source capture, versus about 6.7% for ten through twelve.

The task's prior order-five/six/seven receipts and this receipt retain their
exact JSON bytes in deterministic gzip. This satisfies the aggregate evidence
budget without weakening it, discarding numerical evidence, publishing a
Release, or changing source reconstruction.
