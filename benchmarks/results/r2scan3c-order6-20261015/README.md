# Rejected r²SCAN-3c order-six experiment, 2026-10-15

**No speedup or README timing refresh is claimed.** The experimental consumer
is removed; production dispatch and both README plots remain unchanged. The
retained patch is for reproduction, not an available production selector.

## Basis diagnosis: both panels

Both Direct and DF use r²SCAN-3c's defining spherical **def2-mTZVPP**, including
D4 and gCP. Other README methods use spherical def2-SVP. Replacing the 3c basis
with SVP would change the method, not optimize it. The bundled H/O records have
no zero contraction coefficients to remove.

For the 96-atom water topology, the actual bundled inventories are:

| Basis | Spherical AOs | s shells | p shells | d shells | Unscreened dd × dp rectangle |
| --- | ---: | ---: | ---: | ---: | ---: |
| def2-SVP | 768 | 224 | 128 | 32 | 2,162,688 |
| def2-mTZVPP | 1088 | 288 | 160 | 64 | 21,299,200 |

The last column is `d*(d+1)/2 * d*p`: an inventory bound, **not executed work**.
It grows approximately 9.85×, substantially faster than the public AO count.
The retained October 13 direct 96-atom sample spends 62.196 seconds in stationary
integral derivatives out of 90.089 seconds complete warm E+F. Its DF counterpart
spends 2.915 seconds in integral derivatives and 8.349 seconds in semilocal
geometry work. The direct bottleneck must not be assumed to dominate DF.
The original records remain in `../readme-dft-20261013/`.

## Experiment and decision

Starting from master snapshot `cb8781fb1`, the explicit `cooperative-six`
schedule reused the existing compiler-emitted shared recurrence for angular
order six, preserving the order-zero-through-five sources and order-seven/eight
consumers. The full generated coefficient layout needs 256 lanes; candidate
admission stayed at 128 descriptors per packet. No precision, basis, screening,
SCF tolerance, correction or recurrence formula changed. Automatic dispatch
stayed on the incumbent qualified policy throughout.

The same Release/sm_120 FP64 image ran both arms on Slurm n5 RTX 5090, with
eight CPU threads and Slurm's assigned device visibility. The complete endpoints
include preparation and host force return, D4 and gCP. Each point requested cold,
five fixed-density warm calls, moved, and five moved-warm calls under the original
1800-second point limit. Independent E/F gates stayed at 1e-8 Eh and 1e-7 Eh/Bohr.

All **48 returned canonical endpoints** pass those numerical gates. Maximum
energy/force errors are 1.592e-11 Eh / 3.726e-11 Eh/Bohr. The same-image diagnostic
observations are below; warm cells include all five calls, not a selected repeat.

| Atoms | Arm | Cold (s) | Warm median [min, max] (s) | Moved (s) | Moved-warm median [min, max] (s) |
| ---: | --- | ---: | --- | ---: | --- |
| 12 | control | 9.060 | 1.625 [1.624, 1.664] | 6.671 | 1.616 [1.614, 1.616] |
| 12 | candidate | 9.319 | 1.870 [1.869, 1.906] | 6.896 | 1.866 [1.864, 1.867] |
| 48 | control | 84.227 | 22.476 [22.397, 22.684] | 62.962 | 22.529 [22.436, 22.626] |
| 48 | candidate | 96.941 | 34.911 [34.909, 34.917] | 75.239 | 34.906 [34.900, 34.914] |

Both arms have the same observed SCF iteration/Fock-build pairs: 12-atom cold
18/19, 48-atom cold 19/19, moved 13/13, and every warm/moved-warm call 1/1.
Work is not normalized. Slurm jobs 1608 and 1611 used finite 15- and 20-minute
allocations respectively. The 48-atom warm regression is approximately 55%; the
candidate therefore fails the no-regression requirement and is removed rather
than promoted. No 96-atom native candidate population or DF speedup is claimed.

**These are diagnostic-only, not source-qualified publication timings.**
The intermediate image's advertised source identity
`f692115850ecfca46108271bd1fec3ab2628a67d605a366a61f8f831da7a1218`
differs from the frozen experimental native inventory
`2afb0ba1c0ef8004a3b2e029460b4e65205da1a4ef8406b71a015f6ade011b5f`.
Consumer hashes match, but that does not repair the native receipt. The final
identity-refresh rebuild was stopped after the substantial regression. The
reducer retains this mismatch as unqualified, and no timing here replaces an
accepted README measurement.

The candidate adds shared workspace, a wider mixed consumer and CTA barriers.
These are possible costs, not measured occupancy or stall explanations. This
campaign does not provide a large-system order-six execution census or profiler
attribution; actual work/response checks are provided by the independent fixture.
See the rejected Agent Note for the boundary on any renewed implementation.

## Validation and unsuccessful attempts

- Six real-GPU Libcint derivative tests pass in Slurm n5 job 1606, including
  separate/combined J′/K′, Cartesian/spherical layouts, restricted/unrestricted
  spin, translation and executed order-six recurrence work.
- The final retained README/provenance tests pass, including the canonical
  basis and correction protocol for both direct and fitted panels.
- Early complete pilots reject old stationary AOT plugins with an unchanged
  `contract_identity` mismatch. Fresh matching r2SCAN RKS and SPD plugins permit
  the later numerical checks; no manifest or admission rule was weakened.
- A one-repeat triage attempt rejects the retained five-repeat reference
  protocol. It is retained as a setup failure, not a numerical or timing result.
- Fresh independent 3/96-atom references complete in Slurm job 1603; their
  timings do not replace prior README reference populations.

## Retention and reproduction

`observations.json.xz` keeps all diagnostic samples, numerical rechecks, work,
independent geometry oracles, source receipts, and all four unsuccessful point
attempts. `receipts.json.xz` keeps build/test/scheduler logs and the
launchers. `experiment.patch.xz` contains the rejected implementation and its
independent tests against the master snapshot. `manifest.json` binds these files,
the intermediate library and the matching AOT artifacts. No release or external
archive is created.

Raw vectors, binaries, caches, frozen source and logs remain in ignored storage:
`/data/jzzeng/gqc-r2scan3c-order6-20261015` on n5, with local documents under
`.artifacts/r2scan3c-order6/`. NVCC 12.9.1 and verified ccache 4.5.1 built the
Release image; existing caches were retained. No native source change is retained.
The local experimental build is not a source-matched production image and must
not be reused for publication.

Recheck the retained raw documents from the repository root:

```bash
PYTHONPATH=.:python python benchmarks/results/r2scan3c-order6-20261015/reduce.py \
  .artifacts/r2scan3c-order6 \
  --output .artifacts/r2scan3c-order6/rechecked-observations.json.xz
```

To rerun the rejected implementation, reconstruct the exact recorded master
revision plus the decompressed patch in a separate checkout, build fresh matching
native/AOT artifacts, and use the archived launcher only inside a finite Slurm
allocation. Do not edit artifact identities to reuse a stale build. Reusing this
intermediate image can reproduce only its diagnostic observations, never a
source-qualified publication. Any new promotion requires new matching receipts,
complete 48/96-atom populations, independent oracles and smaller controls.
