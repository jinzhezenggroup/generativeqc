# Bounded canonical UMP2 energy core

Status: implemented
Date: 2026-10-04
Agent: dot

## Decision and scope

The compiler owns spin-labelled TensorIR equations for real canonical UHF MP2.
For ordered occupied/virtual tuples, same-spin channels use
`(g - x)^2 / (4 D)` and the alpha-beta channel uses `g^2 / D`, with
`g = (ia|jb)`, `x = (ib|ja)` and `D = eps_i + eps_j - eps_a - eps_b`.
Spin labels are part of index-space identity, including equal-sized tiles.

`PreparedUMP2Energy` borrows a validated UHF snapshot and conventional,
unscreened source. The existing AO-to-MO provider selects the coefficient owner
for each of its four slots. Exchange uses its own rectangular block request;
swapping axes within a possibly disjoint direct tile is not equivalent.
Restricted blocks keep their existing cache identity. Spin-labelled blocks add
all four spin owners, preventing alpha/beta cache aliasing at identical indices.

This slice supports conventional CPU FP64 energy only. It does not register a
public UMP2 method or provide forces, amplitudes, density fitting, mixed
precision, or GPU execution. Tile storage is bounded; repeated AO source passes
are retained as an explicit correctness baseline, without a performance claim.

## Admission and publication invariants

The shared provider planner charges reference/source buffers even when no spin
channel has an occupied-to-virtual excitation. An empty channel must not bypass
source geometry/basis validation, axis-tile validation, or memory admission.
Active tiles additionally charge direct/exchange feeds and TensorIR storage.
Numeric capacities exclude interpreter/allocator overhead under the existing
provider contract.

Both denominator extrema must be finite before any source read; the nearest
denominator must be strictly negative and outside the requested threshold.
There is no denominator regularization. A failed execution clears retained
blocks and the last published result; only a complete finite scalar result is
published. The caller retains ownership of the source lifetime.

Review caught three defects: adding an unconditional cache-key field broke
restricted CCSD warm-cache admission (and its test accessed nonexistent
`MOBlock.spins`); empty excitation sets skipped all provider admission; and a
finite nearest denominator did not exclude overflow at the far extreme.
Regressions exercise real provider warming and repeated CCSD preparation,
distinct spin-owner cache entries, empty-channel validation and exact budgets,
and finite spectra whose far denominators overflow before integral reads.

## Evidence and acceptance boundary

The UMP2 tests compare independent dense AO contractions, alpha/beta exchange,
and the restricted identities `E_ab = E_OS` and `E_aa + E_bb = E_SS` to
`2e-12` tolerances. Review also checked explicit four-spin orbital sums for nine
occupancy pairs across three tile configurations: 27 comparisons, maximum
absolute error `1.73e-18`. Mid-source failure, retry, nonfinite inputs, and
retained-cache cleanup were checked without native integral generation.

On the reviewed current-master union, 62 bounded host tests passed across
`test_ump2_energy.py`, `test_mp2_energy.py`, `test_posthf_reference.py`,
`test_response_uhf.py`, and the CCSD collective-provider-budget regression.
The optional pinned PySCF 2.14.0 Li energy oracle and two native UHF exports were
deliberately excluded from that host review. Required pre-commit checks passed;
this evidence does not establish native/GPU chemistry or endpoint performance.

Revisit the repeated-transform baseline only with independent scientific gates,
complete endpoint work/memory accounting, and explicit capability changes.

References: #1820 and #1837; `tests/python/test_ump2_energy.py`;
`tests/python/test_cc_solver.py`.
