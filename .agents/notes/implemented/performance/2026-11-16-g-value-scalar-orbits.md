# Decision: share scalar g values across coordinate and center orbits

Status: implemented; complete whole-wheel acceptance pending
Date: 2026-11-16

## Problem

The full-g S/T/V header still occupies 14,904,718 bytes after the conservative
sharing recorded in `../../proposed/2026-11-12-bounded-g-component-sharing.md`.
Separately emitting every Cartesian component and both shell-center orders
repeats scalar mathematics. A unity source would increase the compiler working
set rather than remove this duplication. Unlike the fixed-axis derivatives in
`2026-11-16-g-gradient-center-exchange.md`, values admit all six simultaneous
permutations of Cartesian powers and coordinates.

## Decision and invariants

Only g-bearing helpers use the new dispatch. Canonicalize each component under
all six axis permutations, adding A/B exchange for g/g. Reverse g/lower calls
the lower/g owner with transposed indices and exchanged Gaussian fields.
Representative counts are s/g 15 to 4, p/g 45 to 9, d/g 90 to 18, f/g 150 to
28 and g/g 225 to 26.

Permute every Cartesian pair field and external nuclear center together. Copy
already rounded radial geometry; exchange alpha/beta, A/B and P-A/P-B ownership
without rebuilding the pair. Values acquire no derivative sign. Prefer
identity/no-exchange on ties. Use a separate noinline representative body,
not recursive dispatch. Refuse invalid components before transposition,
coordinate transforms or floating arithmetic.

Retain per-component scientific DAG construction and non-speculative CSE:
only ancestors required by every representative precede its switch. Preserve
Boys evaluation/order, normalized contraction, every shell/component, schedule,
architecture, compute80 PTX and explicit NaN refusal. This is a mathematical
identity, not bitwise FP64 arithmetic equivalence.

The default through-f and scoped COSX headers remain byte-identical. All 75
scientific value programs and the complete qualified g derivative header stay
unchanged. One-off comparisons document this; permanent CI does not pin
generated-source hashes or implementation spelling.

## Complete compiler-owner evidence

The unchanged complete native `one_electron_values.cu` owner uses production
optimization, NVCC 12.9.86 and verified sccache 0.16.0. Both objects retain
sm80/sm90/sm120 CUBINs, compute80 PTX and all 12 architecture/launch resource
entries. Each request records eight cache-stage misses and no hits.

| Metric | Previous whole-pair CSE | Scalar orbits |
| --- | ---: | ---: |
| Generated header bytes | 14,904,718 | 4,341,862 |
| Complete object bytes | 38,962,488 | 16,889,544 |
| Maximum reaped-worker RSS, KiB | 5,789,464 | 3,083,836 |
| Observed compiler-request seconds | 1,502.597 | 842.053 |

Source falls 70.87%, object size 56.65% and worker maximum RSS 46.73%.
The observed request-duration ratio is 0.5604. Requests overlap each other
and part of the separately qualified merged native build: these are not
controlled cold-wheel speedup samples. Worker maximum RSS is not a summed
or cgroup peak and does not establish concurrent container headroom.

Registers never increase: baseline entries report 255, candidate 248--255.
Shared/reported local allocations stay unchanged; stacks increase 0--256 bytes.
Retain full resource records rather than claiming spill-free execution,
improved occupancy or universally reduced device memory.

Candidate header SHA-256:
`af3b2752826b3cdf78ff2bef170e67e02d15f98e39a862bcde67a83d115b3ee9`.
Complete multiarchitecture object SHA-256:
`9b1b18446bb8f6bd9088d74f4e9120534e682ef3b3090faf04ce541b4fbfe706`.

## Independent arithmetic and complete native calls

- The permanent cached emitted-host module passes 432 tests: every shell pair,
  seven geometries, both diffuse/tight g orders, independent Libcint S/T/V,
  invalid indices and unchanged derivative controls. Preserve incumbent raw
  gates, including `atol=2e-9, rtol=2e-11` for g primitives.
- Independent extreme stress covers all nine g-bearing pairs, seven geometries
  and five exponent pairs spanning 0.0005--150: 315 cases and 28,875 primitive
  component pairs. Raw and physically normalized gates both pass; maximum
  normalized error is 4.04e-14. This does not relabel the distinct gradient
  stress diagnostics in the preceding decision.
- Slurm jobs 17521/17522 on authorized node n1 execute each complete compiled
  owner. Each passes 2,900 S/Hcore/T/V launcher cases: 700 independent fixtures
  across four schedules plus 100 optional-diagnostic cases. Cover all 25 shell
  pairs, seven geometries, two exponent families, negative two-primitive
  contractions, normalized Cartesian/three-term AO projections, three nuclei,
  exact matrix symmetry and output guards. Independently contract/project
  Libcint matrices. Maximum oracle error is 1.03e-13; old/new error 7.11e-15.
- Five post-warm samples of eight complete updates retain AO-pair, primitive
  contraction, projected-component and nuclear-visit counts. Event timing
  includes output clearing and all native updates; separate wall times retain
  allocation/upload/download. Observed candidate/baseline medians span
  0.561--1.034 across s/s, f/f, g/s and g/g, Cartesian/mixed3 and all schedules.
  Sequential unlocked-clock jobs do not establish a universal runtime bound
  or public-method speedup.

The production one-electron/AOT/CI/catalog-focused suite passes 2,234 tests
with 63 host-run skips. The pinned `ty==0.0.82` check, all 605 compiler modules,
default-promotion inventory and ordinary catalog integrity also pass. The
separate matching-library profile/ABI module subsequently passes all 47 cases
against the new clean native artifact, not an older source identity.

The clean post-master-merge `bd9d6cc30` library, ABI and 16 independent Direct-HF
endpoints are qualified separately in the preceding gradient note. They precede
this value change and must not be relabeled as its final library. A subsequent
clean complete sm120 Release library/CLI build at code commit
`f6159c523f7e7cd56de2b092dc22f2fdba045de6` passes with portable automatic XC
AOT and the source clean before and after compilation. Slurm job 17523 passes
all 16 independent Direct-HF RHF/UHF, Cartesian/spherical and
cold/warm/moved/restored endpoints, preserving all baseline iteration/Fock
counts. Maximum energy/force errors are 6.40e-14 Hartree / 8.04e-11
Hartree/Bohr. Its library SHA-256 is
`b8d8ba567b0bb74d702beefea1872c425c76dcb09052311798b4dab0b7dde7d3`.
The actual artifact, source/ABI proof and reports are frozen in
`clean-code-native/`. The 639-second incremental build records 16 CUBIN hits
and 70 stage misses; this is not a cold-wheel speedup measurement. These
Direct-HF endpoints retain the existing through-f method domain; changed g
arithmetic is qualified by the complete S/Hcore/T/V owner above.

Manylinux build/link/repair and installed-wheel acceptance remain unverified.
Current Libxc resource/lifetime acceptance is unclaimed;
official catalog refresh changes only two current source identities, not
historical measurements or receipts.

## Remaining bottleneck and revisit

The completed four-way sm120 merged build has a 2,130.439-second Direct-JK task
and 1,514.488-second bounded-fallback task. Retain their overlap/cache context
in the investigation census. Scalar-value sharing does not remove those
owners' density/range/component specializations or prove the entire wheel is
fixed. Preserve scientific methods, bounded fallbacks and numerical/safety
tests when investigating those next.

Revisit when geometry fields, shell domains, precision or contraction ownership
changes. Requalify the complete owner, not just primitive microkernels or source
bytes. Frozen sources, commands, cache statistics/resources, independent
fixtures, Slurm reports and preservation proofs are retained under
`/tmp/gqc-wheel-g-value-orbits/`.
