# Decision: resolve an unspecified KS tile from capability and workload

Status: proposed (production candidate; default-path qualification pending)
Date: 2026-11-02

## Problem

The public 256-point SCF bound left launch and mapped FP64 contraction overhead
on large fitted RKS workloads on A100. An explicit larger bound shortened complete
energy-plus-force endpoints. Applying one larger value to every GPU was not
supported: r2SCAN-3c warm endpoints regressed on both tested Blackwell devices,
and WB97M-V regressed on PRO 6000.

## Decision

Python `None`, and C/C++ zero, request preparation-time selection. Explicit
positive integers remain distinguishable, including the historical value 256.
The capability test uses architecture and measured hardware FP32/FP64 throughput
properties, never a product-name allowlist. The current candidate admits large
fitted restricted, gradient-dependent, device-fused strict-FP64 workloads with
at least 512 AOs and 524288 grid points. Full-rate matrix-capable devices use
1024, or 512 when nonlocal correlation is present.

Resource-promised owners retain the planner's 256-point shape. Unpromised
preparation also requires at least 1 GiB observed free device memory. That
observation is advisory, so a failed enlarged construction releases its owned
storage and retries once at 256 on `bad_alloc`. The conservative retry's failure
propagates. Explicit requests and non-allocation errors never take this retry.
Both host and device storage exhaustion can motivate a smaller automatic owner;
this is a bounded preparation restart, not allocation-error suppression.

The final candidate resolves force batching separately. Large all-electron,
through-d, fitted strict-FP64 RKS owners with 48–128 atoms use 1024 force points
on compute capability 8 or newer; they do not need FP64 matrix acceleration.
Application mixed-gradient budgets, unavailable native stationary profiles,
resource promises and other workload domains retain 256. An explicit SCF tile
continues to supply the inherited force bound; an explicit force tile overrides
it independently. A private prepared-owner flag permits exactly one return to
256 after automatic allocation/admission failure, retiring the controller and
its force diagnostic token first. The complete retry re-queries resident sources
at the same SCF final token, including possibly consumed projection leases.
A smaller post-preparation allowance may retire an enlarged cached controller;
explicit bounds never silently shrink. Resource shape queries resolve automatic to 256 because a
numeric promise does not permit subsequent enlargement.

## Rejected alternatives

A universal 1024 bound would include measured slower Blackwell populations.
For WB97M-V on A100, 512 improved cold/warm by 2.1195%/0.5917%, whereas 1024
improved cold by 1.3324% and regressed warm by 0.1460% in the same-binary pilot.
Consequently nonlocal work has its own selected bound. A100 pilot gains do not
establish a new default until the no-argument public path has been measured.

## Invariants and evidence

Retain every population, including conditioning, slower variants and interrupted
owners. Timing needs successful finite scheduler allocations and owner receipt
hashes/timestamps, in addition to independent energy/force gates. Real LSF
`bhist` completion is valid controller evidence when `bjobs` has purged a job;
it must not be relabeled as a fabricated Slurm or `bjobs` record.

A100 pilots 224406–224408 and 224442–224445 used fixed force tile 1024 and
explicit SCF tiles 256/512/1024. They qualify that experiment, not the production
force default. New default-path runs use the actual omitted force setting and
compare omitted SCF, explicit 256 and GPU4PySCF. Larger/moved and numeric-capacity
checks are separate obligations. The SCF-only candidate source was
`74dd0e9ada2db1c3e07818ed25516807f2b04c68ea6922d8f8be406ab527ff3c`.
It improved six early completed A100 cold populations but left force256 as a
warm bottleneck. Its 120-atom r2SCAN-3c level4/5 oracle difference passed the
full observable gate but failed the stricter one-quarter grid-stability gate;
this historical validation is retained without a promotion claim. The combined
SCF/force candidate uses denser level5/6 independent references and source
`d243aaf78c82f870d8110b40eeddc2950489cfbe866f586d070487f1e695562f`.
Two pre-format CPU builds are not endpoint qualification; final GPU runs bind
both that source identity and the actual main/companion library hashes.

## Revisit when

Extend the eligible work/device domain only after complete endpoint evidence
shows a benefit with unchanged independent observable gates and explicit work
counts. Preserve overrides, resource admission and conservative retry semantics.

## PR snapshot

The retained [final-source evidence](../../../benchmarks/results/automatic-ks-tiles-20261102/README.md)
has 20 qualified timing groups. A100 cold still trails GPU4PySCF for PBE0,
r2SCAN-3c and WB97M-V; WB97M-V uses 28 native iterations versus GPU4's 10.
The automatic defaults improve all completed paired 256-point controls.
Larger/moved PBE passes, but r2SCAN-3c reference stability and corrected resource
qualification remain incomplete. The DF provider workspace hint is not a whole
XC capacity promise; do not repeat the failed probe's assumption. PRO 6000
remaining work was cancelled at the user's request. This PR does not promote
the candidate to universal device/workload qualification.
