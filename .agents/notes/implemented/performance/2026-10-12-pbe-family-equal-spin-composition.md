# Decision: Compose PBE family selection with producer-proven equal spin

Status: implemented (device and endpoint qualification pending)
Date: 2026-10-12

## Problem and decision

The upstream RKS equal-spin path already binds the admitted functional at the
kernel boundary. PBE's opt-in family switch remains a separate prepared policy,
but both settings enter the same method-bound physical consumer when equal-spin
is enabled. Preserve that path rather than manufacture duplicate equal-spin
kernels. Its PBE algebra uses canonical `evaluate_with_channels<4, true>`.
UKS and response retain their qualified ordinary/signed-response consumers.
Ordinary physical and batched PBE keep their prepared generic/specialized
identity; an incompatible batch identity still fails closed.

Ingredient-derived launch widths from upstream apply equally to both family
arms. Tile packing, AO maps, reductions, workspace and precision are preserved.
Automatic providers keep their separate admission and fallback boundaries.

## Experiment and evidence scope

The RKS point-family campaign now writes schema v2, declaring one spin channel
and the same explicit equal-spin request for both arms through preparation,
warm and moved evaluations. The primary family experiment defaults to an
equal-spin opt-out on both arms; production's default-on equal-spin policy is
unchanged. An explicit equal-spin-on experiment records that its physical
family difference is superseded. It can retain complete endpoint evidence but
cannot earn promotion credit for a physical family change that did not occur.
An opt-out result alone also cannot establish profitability of the current
default production path; the remaining production endpoint gates still apply.

The fields record requested policy and source-established routing, not a new
measurement of actual GPU kernels. Device traces and complete-source installed
build provenance remain required. The v1 reader retains historical scope and
does not infer a new equal-spin policy for an old receipt.
Current promotion requires v2 explicit provenance, including when a new record
is relabeled v1 or its policy fields are removed. Historical positive timing
remains auditable without current promotion credit. A synthetic downgrade
negative control reproduced acceptance before this guard and rejection after it;
this is verifier logic evidence, not a device measurement.

## Executable contracts

Production integration and host qualification consume the same emitted point
program and launch ABI. The host contract uses real canonical PBE/LDA algebra,
the pinned 97-point SCF and 87-point restricted PBE0 reference corpora, signed
response finite differences, both spin layouts, ragged batches, buffer canaries,
invalid domains and incompatible policy/launcher rejection. CUDA scheduling is
stubbed; unrelated generated mathematical providers reject if reached. This
does not qualify GPU arithmetic, performance or dynamic traffic.

The 97-point independent gate exercises ordinary two-spin consumers under both
equal-spin policy settings. It does not execute the RKS equal-spin consumer;
that consumer's 87-point cross-policy parity is scoped separately below.
The wider 87-point subnormal corpus independently qualifies the separate
`evaluate_pbe0_restricted_bound` consumer at the original strict tolerance for
all nine components. Ordinary and equal-spin consumers retain full 87-point
cross-policy parity, but their independent errors are reported separately and
are not an ordinary numerical PASS. The first composed test incorrectly applied
the bound consumer's oracle gate to ordinary arithmetic. Host reproduction
retained ordinary failures at rows 8, 9, 65, 66, 67 and 68; the bound consumer
passed all rows. No production arithmetic, reference inputs or tolerance changed.
See [the retained subnormal diagnosis](../../proposed/2026-10-10-pbe0-bound-subnormal-exchange.md).

This replaces affected implementation-string and exact internal-call-count
guards. Independent reference, resource admission and runtime-selection cache
integrity coverage remains. Compilation requires a verified native compiler
cache; a missing local fixture is an explicit skip, never a numerical pass.

## Composition with weighted native stationary response

The merged native stationary controller passes MethodIR exchange/correlation
weights through the shared point traversal. Preserve both weights through
`CuratedPointEvaluator<static_family>` and its canonical response call; carrying
the family alone would silently restore unit-weight response. Generic `-1`,
specialized PBE, physical batches and shared equal-spin identity remain intact.
The canonical response avoids evaluating a zero-weight singular component.

The host ABI contract checks weighted response consistency at two finite-
difference steps across both spin layouts and family/equal-spin policies.
Those differences use the same physical algebra and are consistency evidence.
A separate Dirac exchange derivative checks the empty-spin, zero-correlation
boundary independently. The inherited Libxc weighted-response oracle remains
a separate prerequisite-sensitive gate; a missing oracle is a skip. Graph
identity/capacity and stationary contract v2 admission remain source matched.
None of these host gates qualifies joint-source NVIDIA execution, production
sm90 derivatives, dynamic local traffic or complete 48/96 E+F.

## References

- Supersedes the shared-traversal composition discussion in
  [the original family note](2026-10-08-pbe-point-family-consumer.md) when the
  equal-spin physical path is active; historical measurements remain unchanged.
- `benchmarks/pbe0_xc_tile_pairs.py` and
  `tools/generativeqc_validation/pbe0_xc_tiles.py`.
- `tests/python/test_pbe_point_policy_contract.py`.
