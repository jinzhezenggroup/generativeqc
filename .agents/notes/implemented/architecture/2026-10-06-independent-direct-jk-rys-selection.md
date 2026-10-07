# Independent prepared Direct J/K Rys lowering

Status: implemented, qualified; Rys alternatives remain opt-in
Date: 2026-10-06

## Decision

Compile derivative-free Rys alternatives alongside each incumbent streaming
Fock class in the same stable architecture build unit. The compiler intersects
the production streaming inventory with value IR/root/decoder/target legality.
The optional inventory is not a measured preference or a new coverage contract.
Absent classes, including one-root ssss/psss and the unsupported dddd decoder,
retain their exact incumbent/native paths. No method or molecule list selects
recurrence mathematics in the runtime.

Native prepared J and K owners freeze independent masks through
`GENERATIVEQC_DIRECT_J_FOCK_LOWERING=incumbent|rys` and
`GENERATIVEQC_DIRECT_K_FOCK_LOWERING=incumbent|rys`. The default is incumbent.
Disabled AOT classes remain disabled in the alternative inventory. Range K and
mixed-J contracts keep their existing fallbacks. Launch errors propagate rather
than retry into partially accumulated output.

HF also consumes these choices. A complete strict-FP64 bounded streaming owner
splits into J-only and HF-weighted K-only passes when either choice requests a
Rys inventory, or `GENERATIVEQC_DIRECT_HF_SEPARATE_JK=1`. Both immutable views
borrow the original topology and accumulate into existing Fock scratch. K
scatter applies the original RHF -1/2 or UHF -1 factor exactly once. The task
consumer bits together encode this weighted K contract; ordinary J-only,
raw-K-only and fused HF retain their established meanings. Native dddd uses the
same compiler-owned scatter contract. Partial/higher-l and mixed owners retain
the qualified fused route. Prepared replay freezes the choice; rebuilt geometry
creates a new owner.

## Timing and qualification

The shared optional CUDA component ledger records complete generated Direct-J/K
builds (density transforms, screening metadata, traversal, projection) and HF
streaming passes (queue reset and generated/native class work). HF's density
preparation/projection remain common work. Capture receipts are explicitly not
GPU timings. Traced endpoints are diagnostic; clean wall time is measured with
tracing disabled. Candidate selection does not establish profitability.

Independent Libcint gates must cover the complete compiler candidate inventory,
RHF/UHF raw J/K and HF-weighted K matrices, repeated and changed geometry E/F,
and empty/component tails. Complete 48/96-atom PBE0/HF warm/moved-warm comparisons
must retain numerical gates, semantic work, binary/source identities and
per-class resource/timing evidence. No default promotion is justified before
that evidence exists; in particular resource/spill costs may reject high roots.

### Completed PBE0 qualification

Parent #2020's frozen `efd2c852b26e9f5122ac76f41da31af3325c9589` build passed
432 complete-inventory Libcint matrix checks, 12 prepared J/K checks and 18
public PBE0 Libcint endpoints (finite Slurm job 6219). All 48 cold/warm/moved/
moved-warm endpoint records passed their numerical gates. Clean job 6220
retained five observations per steady phase:

| Atoms | K lowering | Warm median (s) | Moved-warm median (s) |
| --- | --- | ---: | ---: |
| 48 | Incumbent | 9.826260 | 9.801570 |
| 48 | Rys | 10.347511 | 10.354223 |
| 96 | Incumbent | 26.532668 | 26.606918 |
| 96 | Rys | 28.665267 | 28.649512 |

Trace-only complete-build job 6222 measured warm J/K incumbent at 1183.588 /
1059.999 ms for 48 atoms and 2428.025 / 1997.739 ms for 96 atoms. Warm Rys K
was 1607.898 and 4078.822 ms respectively; J stayed incumbent. Actual-work
job 6221 confirmed 18 selected Rys classes plus three exact fallbacks. Warm
admitted K shell quartets were 22,776,236 for both 48-atom controls, and
72,116,586 / 72,116,584 for 96-atom incumbent/Rys. The tiny threshold-boundary
difference is not a material work reduction. End-to-end Rys K is slower by
approximately 5-8%; no default promotion is warranted.

The parent HF measurements exposed an empty alternative streaming tail.
Child #2021 fixes primary ownership and repeats the entire HF qualification;
see [the primary-ownership note](2026-10-06-independent-hf-primary-ownership.md).
Previous tail-only HF timing is not Rys performance evidence.

GPU4PySCF's energy/force oracle remains an independent numerical reference,
but its installed 1.8.1 RKS `get_veff` uses delta density when `vhf_last.vj`
exists even with `direct_scf=False`. Finite Slurm probe 6235 directly observed
this for PBE0, while HF honors that flag. A future full-density timing comparison
must enforce and observe full rebuilds; setting the flag alone is insufficient.
This does not invalidate the native incumbent-versus-Rys comparison above.

Parent receipts remain under its ignored `.artifacts/jk-rys/`, including the
strict `qualification-summary.json` and `incremental-probe-6235/result.json`.

Refs #2015, #2017, #2020, #2021, #1892. This builds on #2007 and the shared force scheduler;
force output layout does not force value J/K to choose the same lowering.
