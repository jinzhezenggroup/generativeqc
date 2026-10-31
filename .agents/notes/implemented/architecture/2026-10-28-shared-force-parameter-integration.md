# Decision: share immutable parameter rows across energy and staged forces

Status: implemented
Date: 2026-10-28

## Problem

The shared force AOT branch and the main-branch parameter worker package both
owned automatic point generation. Choosing either side of their merge conflict
would discard the other's execution path. Main also added range/VV10 stationary
sources and optimized catalog enumeration; these must survive the integration.
An automatic text merge placed a literal stationary launcher inside common
validation, where its provider and density-layout variables were undefined.

## Decision

Keep common launch validation independent of provider emission. The parameter
worker optionally emits a staged physical point producer using the same immutable
coefficient row as energy. Each registration gates packaged spins and advertises
a stationary callback only when selected. Provider source identity version 3
binds callback availability and executable identity, while canonical mathematics
and CPU/CUDA science admission retain their original meanings.

Force spin semantics remain independent of the energy equal-spin optimization:
LDA RKS retains its total-density differential; GGA/MGGA uses the full polarized
differential. The shared native consumer retains external range/VV10 callbacks
and their ordered source sum. Optional resident storage keeps bounded fallbacks.

## Historical evidence

The retained shared-AOT verifier now offers explicit `--retained-only` replay.
This authenticates old stored artifacts, scientific gates, finite differences,
work counts, performance samples and evidence-envelope references. It does not
attach those envelopes to today's compiler. Default verification still requires
live compiler subject identity and fails after an incompatible identity change.
Do not repin historical subjects to make current admission pass.

## Rejected alternatives

Disabling parameter sharing for force packages loses existing energy reuse.
Applying the energy-only paired adapter to force tiles can erase physical spin
derivatives. Replacing source identities with mathematical identities loses
executable provenance. Dropping external callbacks breaks complete composite
forces. Increasing discovery timeouts hides the catalog-enumeration regression.

## Validation

Executable host tests compile the generated parameter registrations and staged
kernel bodies, compare physical derivatives with independent Libxc, reject
unpackaged spins, and propagate producer failures. Existing graph, native binding,
package audit, capacity and reconfiguration suites cover consumer composition.
Actual CUDA compilation and full SCF/energy/force finite differences require
separate finite Slurm jobs; host simulation makes no device performance claim.
