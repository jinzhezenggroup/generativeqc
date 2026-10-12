---
orphan: true
---

# Canonical UCCSD(T) energy owner

`tools.generativeqc_cc.PreparedUCCSDT` adds complete real-FP64 conventional
canonical triples energy to an accepted [UCCSD state](uccsd_solver.md). It
reuses the parent's native CPU/CUDA executor and the compiler-owned
[four-channel triples definition](uccsd_triples.md); it does not introduce
another integral producer, restricted normalization or spin-orbital embedding.
This is an internal supplied-reference energy API, not a Calculator registry
entry, DF method, resident GPU solver, Lambda implementation or analytic force.

## Binding and execution

```python
from tools.generativeqc_cc import PreparedUCCSD, PreparedUCCSDT, UCCSDTOptions

with PreparedUCCSD(snapshot, provider) as parent:
    state = parent.solve()
    if not state.converged:
        raise RuntimeError(state.reason)
    with PreparedUCCSDT(parent, state, options=UCCSDTOptions()) as prepared:
        result = prepared.energy()
        if not result.converged:
            raise RuntimeError(result.reason)
        total_energy = result.total_energy
```

Preparation requires a live parent, a converged `UCCSDResult`, matching
reference/Hamiltonian/integral/equation identities, finite consistent CCSD
energies and exact agreement between full and packed amplitudes. It owns
irreversibly immutable copies of the amplitude state and borrows immutable
parent feeds/reference and the parent's executor. Both the parent and its
execution configuration must remain unchanged and open. Closing a provider or
AO source after parent preparation is allowed. Closing the triples owner does
not close the parent or executor and releases its amplitude/template bindings.

`energy()` freshly evaluates the accepted CCSD energy and all five full
physical residual blocks on the selected backend. The parent's energy and
residual tolerances must pass again before any triples tile executes. A forged
converged flag, finite energy, or packed/full agreement is not sufficient.

Every possible `aaa`, `aab`, `abb`, `bbb` channel covers its complete ordered
six-axis domain, including repeated same-spin indices. Channels without enough
distinct occupied/virtual orbitals contribute exact zero without tile execution.
Host gathers obey the compiler's canonical operand bindings; all scientific
contractions and divisions execute through the parent's selected native backend.
Channel accumulation uses compensated host FP64 sums and a final `math.fsum`.

CUDA is explicitly staged, not an all-device endpoint: bounded feed gathering,
amplitude checkpoints and scalar accumulation are host-owned. CC residual
replay, MO transforms and triples contractions use CUDA, with no CPU equation
or PySCF fallback. Cached CUDA plans/artifacts do not retain overlapping device
arenas between executions. Source, integral and tensor backends are reported
separately, as in the parent contract.

## Physical denominator gate

No shift or regularization changes the physical triples denominator. Admission
uses occupied/virtual extrema in the graph's actual FP64 sum order: three
occupied additions followed by three virtual subtractions. Every prefix must
be finite and the largest executed denominator must be negative with magnitude
strictly greater than `denominator_threshold` (default `1e-10`). Extrema use the
whole occupied/virtual segments, not an assumption of sorting more tightly than
the UHF reference validation tolerance.

The gate includes repeated coordinates even though their exact antisymmetric
moments vanish; tiny or zero denominators there can amplify cancellation
roundoff. Empty triples domains have no denominator minimum. Near-zero,
non-Aufbau and nonfinite denominators fail before native execution.

## Numeric and work admission

`UCCSDTOptions` defaults are:

- `tile_shape=(4, 4, 4, 2, 2, 2)` for independent `i,j,k,a,b,c` limits;
- `max_bytes=512 << 20` for logical numeric storage;
- `max_tiles=100_000` for complete-grid executions;
- `max_total_work=100_000_000_000` for complete semantic scalar-loop work.

Preparation plans exact full/edge shapes and their multiplicities before
execution, without materializing the tile grid. Shape-identical tiles reuse
offset-independent programs. Template identities include both the graph and
canonical operand bindings; a scalar graph hash alone is not a binding identity.

The numeric bound includes the retained reference/parent MO feeds/denominators,
amplitude copies, physical replay and bounded tile gather/native execution
buffers. It is a logical numeric-buffer bound, not process RSS or arbitrary
caller-owned source/provider/cache storage. Fresh replay and triples execution
occur sequentially, so their peaks are combined by maximum, not addition.
There is no retained full molecular T3 or denominator grid; temporary moments
are limited to the explicitly admitted tile.

An oversized requested tile shrinks deterministically along its largest axis
when complete numeric or per-call executor capacity requires it. A minimum
tile that cannot fit is rejected. Fallback never omits coordinates or bypasses
complete tile/work caps; smaller tiles can increase semantic work. The CPU
emitter supplies the same pure metadata scalar-loop work model on both backends
but does not execute CPU equations behind CUDA. Work covers one fresh physical
parent replay plus every triples tile, not hardware FLOPs or the earlier CCSD
iterations. Memory-bounded does not imply scalable algorithmic work.

## Results and failures

`UCCSDTResult` publishes the parent CCSD correlation, four channel energies,
total triples correction, corrected correlation and corrected total energy.
Successful publication requires complete admitted tile coverage and finite
channel/correlation/total sums. Empty excitation or triples domains publish
exact zero corrections.

Admission, parent-qualification and lifetime/configuration violations raise.
A late nonfinite tile or native triples failure returns `nonfinite` or
`execution_error`, respectively, with no channel, triples, corrected correlation
or total energy. Completed-tile counts remain diagnostic, not partial scientific
results. Result channel energies and nested provenance mappings are immutable.

Provenance reports physical/state/template identities, backend roles, actual
denominator minimum, fresh parent residual maximum/energy drift, numeric bound,
requested/accepted tile controls, fallback steps, planned/completed coverage
and replay/triples semantic work. `triples_endpoint_seconds` covers this owner's
preparation, fresh replay, compilation, gathering, execution and accumulation.
It excludes earlier SCF/provider setup/CCSD solving; the parent separately
reports preparation/solve timings and physical evaluation counts. These
diagnostics are not complete Calculator performance claims.

## Qualification

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=python:. \
  GENERATIVEQC_LIBRARY="$PWD/build/libgenerativeqc.so" \
  python -m pytest tests/python/test_uccsd_t_owner.py \
  tests/python/test_uccsd_native.py -q
```

Independent PySCF molecular gates cover Li, OH, NH2, broken-symmetry H4 and
asymmetric H6; H6 has nonzero contributions from all four spin channels. The
collapsed-spin complete endpoint agrees with the existing RCCSD(T) owner.
Nonzero molecular and exact one-body tests exercise deterministic memory
fallback, complete coverage, work/tile caps, empty domains, denominator sum
order/overflow, fresh-state rejection, immutable lifetimes and late failures.
The native producer test uses exact shared shell definitions for native UHF,
streamed shell integrals and the complete CPU UCCSD(T) endpoint.

Real-CUDA molecular qualification selects `-k complete_cuda` with
`GENERATIVEQC_TENSOR_CUDA_TEST=1`, an explicit compiler/architecture, and an
available permitted GPU through a finite Slurm allocation. It covers NH2,
broken H4 and all four asymmetric H6 channels with actual CUDA transforms,
converged UCCSD iterations and triples execution. A host skip is not CUDA
evidence. Fixed-amplitude derivative foundations do not qualify unrestricted
Lambda, relaxed orbital response or nuclear forces.
