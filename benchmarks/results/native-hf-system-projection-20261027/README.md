# Native CPU HF system-to-system projection

This bounded #2227 R4/R6 migration moves overlap preparation and the RHF/UHF
spin-projection loop into one native owner. It reuses existing cross-overlap
and occupied-projection mathematics. Python retains per-item source eligibility,
reports and transactional seed installation. Missing protocol, custom policy
and unsupported domains retain explicit compatibility; native errors never retry it.

Measured base: `c8ba02453b326084d5175ef644c6be1c810afce7`. Apply the decompressed
`candidate.patch.xz` to a separate checkout. Its SHA-256 is
`9d4ef8cfaaa80725d1477c69c5aa9f4a4be0381681b5789f1a73ef14ca857494`.
Publication, finalized rationale and lossless historical-summary compression
follow the frozen patch; measured production/test source is unchanged.

## Replay

From the repository root with Python and NumPy:

```bash
python benchmarks/results/native-hf-system-projection-20261027/replay.py
```

The shared reader authenticates all members. Replay checks all 1536 independent
E/F/D publications including warmups, native/compatibility projection ownership,
unchanged iteration/Fock work and all eight original 1.05 timing gates. Absolute
E/F/D gates are 1e-8 Ha, 1e-6 Ha/bohr and 1e-7; maximum errors are
2.731148640577885e-14, 8.287537323070637e-12 and 1.5955570198400437e-11.

All eight candidate/baseline timing ratios pass; maximum is 1.0065456757099125.
Four RHF/UHF x Direct/DF cases use two different geometries, ABBA and BAAB,
21 observations plus three warmups per block. Every sample is retained.
The pilot predicted 84.20 seconds; the campaign took 85.32 seconds. This is
bounded nonregression evidence, without a significant-speedup promotion claim.

Each timed observation includes both geometries' source/target planning and
solves, projection, physical audit and cleanup. Candidate collection forbids
Python cross-overlap/per-spin preparation and requires `native_cpu_system`
ownership. The independent reference is reused byte-for-byte because model,
geometries and controls are identical; `reference-origin.json` records that
provenance. Oracle results never enter production. Final verification correctly
remains `unverified` without independent observed accuracy supplied to it.

## Reproduction and qualification

Restore the original runners without executing them:

```bash
python benchmarks/results/native-hf-system-projection-20261027/replay.py \
  --extract-frozen .artifacts/native-hf-system-projection-reproduction
```

The frozen capsule contains collector, campaign, original audit, pilot runner,
observations/estimate, source manifest and reference origin with individual
original byte hashes. Patch/reference XZ members preserve exact original bytes;
full samples use token-preserving JSON compaction followed by standard XZ.

Create baseline/candidate checkouts from the pinned base, applying the patch only
to the candidate. Use `build/native-ks-catalog`, CPU Release, Ninja, GNU 11.4.0,
CUDA OFF and verified sccache 0.16.0. The local dependency prefix was
`/home/jzzeng/miniconda3`. The existing clean baseline build was updated from
`02562ba42` to the measured base, whose differences are documentation/evidence,
and rebuilt. Configure/build/installation/test/cache receipts are in `summary`.
Shared-server cache counters are provenance, not an attributable speedup metric.

Run the collector's `oracle` mode separately if regenerating references, then
`campaign.py --baseline ... --candidate ... --output ...`. Preserve the protocol,
adapting only paths and CPU affinity; recorded affinity is cores 46 and 47 on
AMD EPYC 7K62, with OMP/OpenBLAS/MKL threads each set to one.

121 CTests and 279 Python tests plus 25 subtests passed, with Sphinx, pre-commit,
API and runtime-ownership audits. An external client compiled against the
installed SDK executes with `PATH=/nonexistent`. Independent PySCF tests cover
Ne+H2, moved geometry, RHF/UHF, Cartesian/spherical and STO-3G/def2-SVP source
bases; an empty beta channel and exact/below-budget refusal are covered.
Transactional multi-spin, alias/count/ABI/nonfinite and model-identity errors
preserve outputs. Native failure closes descriptors without Python retry.

PySCF receives the exact requested basis primitives/coefficients: its named
STO-3G differs in coefficient rounding. It independently evaluates integrals
and least-squares; Cartesian AO normalization is transformed explicitly.
Initial setup/oracle failures are retained, with unchanged production equations
and acceptance gates.

Capacity composes existing projection workspace, live metrics and detached spin
results. Caller buffers, prepared fleets, metadata and opaque provider overhead
are separate; no RSS bound is claimed. No new CUDA/ECP admission, full progressive
controller migration or complete #2227 closure is implied.
