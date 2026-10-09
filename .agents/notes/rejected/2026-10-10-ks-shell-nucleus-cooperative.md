# Decision: enumerate cooperative one-electron response from resident shell topology

Status: rejected; the 48-atom complete-endpoint gate fails
Date: 2026-10-10

Superseded by the implicit one-AO-pair-per-warp candidate in
`../implemented/performance/2026-10-10-ks-implicit-cooperative-pairs.md`.

## Disposition

The formal V2 48-atom same-binary ablation has five interleaved repeats per
side. Warm and moved-warm median improvements are approximately 1.7% and
1.6%, both below the unchanged 2% minimum. No PR or default promotion is
authorized by these results. The 96-atom population remains retained but
cannot rescue the failed 48-atom gate. Subsequent V3 compilation overlaps
that later population, so it is not clean promotion evidence in any case.

CPU binary resource inspection reports 255 registers, 1,168 stack bytes and
1,600 shared bytes for both V1 and V2 cooperative kernels. A register-frame
regression is therefore not supported by those static attributes. Serializing
several AO pairs within each shell-owned warp is the next scheduling hypothesis;
the replacement must qualify independently rather than selectively reuse V2
timings or lower the noise gate.

## Problem

The prepared stationary KS bridge borrows Direct-owned shell metadata, D/W,
force scratch and its stream, but has no triangular AO-pair index list. It
therefore hard-coded the component-lane schedule even though the normal
generated one-electron derivative owner already defaulted to nuclear-center
lanes. The retained #2166 96-atom warm trace attributes 1.094 s to two
`shell_warp_gradient` calls, almost entirely nuclear attraction. This is an
intrusive diagnostic, not a clean endpoint timing or an additive speedup claim.

## Proposed decision

- Add a shell-topology adapter for existing generated schedule 3. Null AO-pair
  pointers and a zero pair count select it; explicit AO-list callers retain
  their existing implementation. Partially supplied AO lists fail validation.
- A warp enumerates each admitted shell's lower-triangular AO component pairs
  and calls the existing `contract_pair_nucleus_cooperative`. Its primitive
  derivative DAG, shared primitive geometry, center ownership, signs and
  reduction remain unchanged. Every lane visits the same pair before entering
  the existing warp collectives; inactive systems and tail shell tasks skip
  uniformly.
- Choose the wider capped parallel domain per shell pair:
  `min(Natom, 32) > min(component_pairs, 32)`. Same-shell component counts are
  triangular. Ties retain component lanes. The component complement is a
  separate kernel so a scalar contraction frame does not inflate every
  cooperative block. Both scans are bounded by the existing shell-pair list;
  there is no sorted queue, prefix panel or added retained allocation.
- Check the first launch status before submitting the complement. Preserve
  failure status without clearing it, and do not silently retry another owner.
- The prepared Hcore bridge honors the existing default/explicit cooperative
  selector. Pulay remains component-lane because it has no attraction channel.
  An explicit `shell_warp` retains the previous Hcore path; other
  noncooperative controls retain that prepared bridge's bounded component
  implementation rather than inventing missing AO-list/serial capabilities.
- Standalone schedule-3 bridges pack shell-pair rather than AO-pair indices.
  Their existing conservative resource admission bound remains unchanged.

## Rejected alternatives

Reintroducing an O(NAO^2) AO-index reservation into the prepared Direct owner
would undo its bounded metadata reuse. Duplicating the attraction recurrence
would violate compiler ownership. Always serializing every AO component inside
one cooperative warp underfills small-nucleus/high-angular shapes; the
disjoint component complement avoids that scheduling cliff without a
molecule-specific size threshold. Combining both contraction frames in one
kernel was not chosen because it defeats resource specialization.

## Invariants

The compiler remains the only S/T/V mathematical owner. No new public ABI,
scientific screen, tolerance, precision, density/Pulay convention or production
CPU/reference dependency is introduced. Explicit AO-list schedule 3 remains
valid. Existing resource ceilings, stream ownership, borrowed scratch lifetime
and failure propagation remain enforced. All real-GPU work uses finite Slurm
allocations on n1 with assigned visibility; compiler caching is reused.

## Evidence so far

- Actual native enumeration is compiled as a host ownership test across one
  and three systems, masked neighbors, 1/2/3/16/32/48/96/128 nuclei and mixed
  component widths through Cartesian f. Every triangular AO pair occurs once
  in the disjoint composition, and all cooperative lanes see the same order.
- Actual launch code is compiled with injected submit failures: malformed
  metadata, partial AO lists, empty tasks, matrix-product overflow and either
  launch failure reject correctly. Existing AO-list dispatch remains intact.
- Two independent PySCF/libcint arbitrary nonsymmetric S/T/V cases exercise
  Cartesian and spherical s/p/d/f, negative contractions, additive center
  response, all generated mappings and two finite-difference steps.
- V1 pilot complete endpoints at 48 and 96 atoms retain all setups and calls;
  each warm call has one SCF iteration/Fock and passes the existing independent
  energy/force gates. The pilot is not a promotion population: it has one
  paired repeat and overlaps compilation on another Slurm GPU.
- V2's first build rejected host-only `std::min` in device code. Explicit capped
  comparisons fix the source rather than changing compiler correctness flags.
  The failure log and intermediate binaries remain retained.

## Qualification boundary

The formal V2 comparison is a same-native-binary schedule ablation, not a
cross-binary timing claim. The frozen native/AOT basis is the retained #2166
v7 composition on `125a4e33f`, plus the exact three changed native sources.
Each arm uses its own independently converged, publicly frozen native seed;
byte-identical seeds are not claimed. The independently reconstructed full
GPU4PySCF E/F references and scientific thresholds remain unchanged.

Five interleaved repeats per side at 48/96 atoms and both warm/moved-warm
geometries must pass the existing >2% and twice-summed-relative-MAD gate before
promotion. Keep all complete setups, priming, errors, vectors, work ledgers,
resource bounds and outliers. Cold and first-moved calls are retained numerical
checks, not matched-work cold speed claims. Separate sanitizers/diagnostics do
not enter the clean timing population.

Raw evidence lives in ignored `.artifacts/shell-nucleus-stage/` and n1
`/data/jzzeng/qc-next-hotspot-20261009-125a4e33f/evidence/shell-nucleus-v{1,2}/`.
The review branch starts at current master `e8ce9f17e`; the emitted primitive
and changed native source comparison, rather than a whole-master timing label,
must establish the port boundary. Monitor master increments by affected paths;
unrelated commits do not justify repeating the entire qualification matrix.

## Revisit when

Investigate another ownership mapping if complete endpoints expose an angular
or nuclear-domain cliff, or the Direct owner gains a budgeted compressed AO-pair
domain that is genuinely cheaper than the retained shell enumeration. Do not
replace the fallback or promote a new threshold from isolated kernel timings.

## References

- `.agents/notes/implemented/performance/2026-09-20-generated-nucleus-cooperative-one-electron-default.md`
- `.agents/notes/implemented/performance/2026-10-01-default-screened-through-f.md`
- `.agents/notes/implemented/performance/2026-10-09-bulk-stationary-point-producer.md`
- `docs/developer/one_electron_derivatives.md`
