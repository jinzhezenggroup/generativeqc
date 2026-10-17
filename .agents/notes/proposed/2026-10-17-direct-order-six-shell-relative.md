# Trial: share generated order-six relative responses across AO components

Status: implemented in an isolated qualification export; default remains off
Date: 2026-10-17

## Boundary and hypothesis

Current master `a6ac79757e38be772f3a22c5e0b5b4bea314fad3` is composed with
the six exact-force compiler files from #2558
(`719214668aeb69146174a7039490fd95d5607533`). Neither PR #2558 nor occupied-XC
PR #2577 is merged. The composition does not enable occupied XC or guarded FP32.
Do not relabel its measurements as master or as the older 43-second source cohort.

The shared compiler prerequisite also changes the already-admitted generated
order-seven/eight consumers; the order-six title describes the additional drain,
not the entire source delta from master. Same-image selector measurements isolate
the added reuse schedule and do not qualify every prerequisite change against
unmodified master. Preserve this distinction when composing or promoting it.

The older cooperative-six trial was rejected with approximately 55% regression
at 48 atoms. Its per-atom AD contraction and mixed low/order-six private frame
are not restored. This trial borrows the subsequently generated shared
relative-center response basis and gives order six its own bounded CTA drain.
The low-order scalar consumers retain their 128-lane specialization. Spatial
reuse, explicit disables, missing-cache/topology and unsupported-domain
fallbacks remain available. `GENERATIVEQC_DIRECT_SPD_FORCE_DOMAINS=relative-reuse`
is explicit only; automatic selection is unchanged.

This changes actual work, not just enumeration: one Coulomb/Boys recurrence and
relative Hermite preparation is shared by every admitted Cartesian component
of a primitive shell quartet. Constant nuclear Jacobians are applied after the
shared nine-response contraction. Exact AO-specific screening, signed source
weights, symmetry, independent atom mapping, strict FP64 and force accumulation
remain owned by their existing generated/scientific contracts.

## Preliminary evidence and limitations

Finite Slurm node2 job 3328 uses an assigned RTX PRO 6000 Blackwell, not the
target RTX 5090. Six interleaved 512-quartet event samples show about 2.6--3.3x
lower time for the three tested angular arrangements. This is an isolated
primitive consumer diagnostic, not an E+F speedup or a hardware FLOP count.
Their maximum force differences from the retained spatial consumer are below
`8e-19`; independent all-center displaced-integral and Hermite quadrature
oracles are separate checks. All 454 focused host tests pass after reconciling
the force dependency's updated oracle fixture; the initial stale instrumentation
wrapper compile failure is retained in raw logs, not suppressed.

A separate counted/sanitized Slurm node2 launch reports per fixture four bra,
16 ket and 16 Coulomb preparations, with 3456/5184/5184 actual component
contractions and 216/324/324 published components. The corresponding component-
local primitive preparation domain is 3456/5184/5184. Memcheck, synccheck and
racecheck report zero errors/hazards. These are not molecular endpoint counts.

Node2's installed userspace driver does not match its loaded kernel module.
The diagnostic uses only process-local official NVIDIA 595.91.07 libraries,
verified against NVIDIA's redistribution SHA256 manifests. It does not install
or alter a system driver, override Slurm device visibility, or establish 5090
qualification. The 5090 probe is queued as node4 job 1103; full independent
GPU source tests and matched complete populations are still required.

## Remaining gates

Qualify the actual linked native dispatcher, fallback and all source layouts
against independent Libcint. Authenticate the final rebuilt source inventory
and core/AOT identities before deployment. Retain all cold, five warm, moved
and five moved-warm calls with the unchanged 1800-second population cap, actual
SCF/Fock counts and independent energy/force gates. Measure order-six/seven/eight
device times and entered work separately from mutually exclusive complete wall
owners. Do not promote a default, update the README or claim GPU4PySCF parity
based on the preliminary probes.

Raw commands, source exports, cache statistics, failures, counters and receipts:
`.artifacts/direct-parity-20261017/`; isolated source export:
`/data/jzzeng/gqc-direct-parity-20261017/relative-six/source`.
