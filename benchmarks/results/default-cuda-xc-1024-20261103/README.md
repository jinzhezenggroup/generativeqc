# 1024-point CUDA default application verification

## Applied behavior

Ordinary CUDA semilocal XC, including direct/global-hybrid and unrestricted
calculations, now defaults to **1024-point SCF and force tiles**. No local
profile installation, geometry identity or explicit tile argument is needed.
The Python option remains `None`, preserving automatic-versus-explicit
provenance and the bounded 256-point allocation-failure retries.

Explicit SCF/force tiles still win. CPU, numeric resource promises, active
ledgers and reference-memory budgets keep their existing bounded schedules;
nonlocal CUDA retains its separately qualified policy. In particular, an
automatic `ResourceBudget()` owner still quotes and uses 256 rather than
growing after admission. See `docs/user/ks_options.md` for the current contract.

## Fresh-source complete endpoint verification

Slurm **1476**, n4 / RTX 5090 / SM120, finite 25-minute allocation, unchanged
Slurm device visibility. Three isolated new processes run the same 96-atom
water32 PBE0-RKS / 768 spherical def2-SVP AO / Treutler level-3 / FP64 workload:

| Configuration | Actual SCF tile | Cold complete E/F | Warm complete E/F |
| --- | --- | --- | --- |
| Ordinary automatic, no resource promise | 1024 | Pass | Pass |
| Explicit 1024, no resource promise | 1024 | Pass | Pass |
| Automatic with `ResourceBudget()` | 256 | Pass | Pass |

Each owner converges its own native density and returns full host energy and
analytic force vectors. Warm execution freezes that owner's updates and follows
one untimed replay. No density, orbitals or prepared owner is shared between
processes, and no CPU-oracle data enters the production solve.

The automatic and explicit-1024 endpoints have identical actual work: cold
**17 Fock builds / 17 J / 17 K**, warm **1 Fock build / 1 J / 1 K**. Their
energy difference is at most **9.094947017729282e-13 Hartree**; maximum force
differences are **2.227107387398064e-13** cold and **3.561595462997502e-13** warm.
All six endpoints pass the unchanged independent numerical gate against the
retained PySCF/libxc reference. Its independently recomputed full-Fock orbital
gradient norm is **2.0347787350495472e-11**, below the `1e-10` reference gate.

These are **application/parity checks, not a new paired performance campaign**.
The historical five-pair resident and complete-process cold speedups remain in
`../dft96-local-autotune-20261103/` and `../dft96-cold-autotune-20261103/` with
their original source identities and acceptance boundaries. Do not compare
the single unpromised and resource-promised timings here as a tile speedup:
they also have different resource/lifecycle policies. No universal workload
speedup or memory reduction is claimed.

## Regression scope and unrelated failures

- Final-source CPU selection: **357 passed, 31 opt-in GPU skipped**.
- Scheduled diagnostic regressions: **28 passed, 2 deselected**; independently
  referenced public LDA/PBE/r2SCAN RKS/UKS force regressions: **6 passed,
  54 deselected**. These scheduled selections include CPU fixture variants;
  they are not 34 exclusively GPU test cases.
- Small complete-E/F tests cover ordinary versus resource-promised defaults,
  explicit force overrides, and automatic versus explicit tile parity.
- Native policy tests exercise small/direct, consumer/unknown-FP64-ratio,
  explicit override, promised-capacity and unchanged nonlocal cases.
- The broader run in Slurm **1474** exposed an unchanged transport-ledger
  assertion: cold automatic seeding adds 64 setup-H2D bytes instead of keeping
  the setup counter unchanged. The same assertion fails on the original
  source/library in Slurm **1475**. The qualified selection excludes that
  pre-existing test (and its CPU-only skip), without changing its assertions.
  Both failures and the old-source identity are retained.
- Four additional post-SCF admission checks cannot load the locally absent
  pinned `xtbloom-gfn1-d3` reference cache. They fail before scientific execution;
  their original log is retained separately from the passing scoped selection.

## Source, binary and retention

Canonical and observed native source identity:
`771c2dd604e3b980bd15b0c17bcdeb9ab422cedfcb2b3c74e205bdab6ab4ef3a`.
Release / SM120 native library SHA-256:
`f22c39ed6d9533f566c009d95463cdad7da560a869025353019b4a40bf89c70b`.
The build uses its configured verified `ccache` launcher; before/after statistics
and the final build log are retained. Formatting changed source bytes after an
earlier precheck, so the final binary was rebuilt and the endpoints rerun; the
earlier pre-format observations are not the evidence above.

`receipts.json.xz`: **81,296 bytes**, SHA-256
`aa8c1ffd41eb30ba50db9e6065ffd6c0012af52ccab97c8e19397597ea32d5f4`.
It retains all six full E/F records, work/iteration/resource observations,
independent reference, executed verification/launch source, production source
and patch against the named base commit, cache/build logs, regression logs and
classified unrelated failures. The legacy plan supplies scientific inputs only;
its old library/source fields do not identify this execution. The worker checks
the current canonical and observed native identities independently.

No native binary, profile bundle, release or global J/K/DF policy is published.
Durable rationale:
`.agents/notes/implemented/performance/2026-11-03-default-cuda-xc-1024.md`.
