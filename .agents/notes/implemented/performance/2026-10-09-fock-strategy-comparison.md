# Decision record: Fock provider strategy comparison

Status: implemented (historical acceptance evidence)
Date: 2026-10-09
Source documented: GenerativeQC master `fb9586569769fccac67bc23411d902e08119024d`

## Purpose and limits

The shared-Fock migration's source-matched CPU/CUDA non-regression
comparison is archived here rather than in the maintained API/science
contract. The authoritative measurement bundle remains
[benchmarks/results/fock-strategies](../../../../benchmarks/results/fock-strategies/README.md).
Its source/build identity, raw samples, accuracy checks and iteration counts
must accompany any comparison. Original numbers below are **historical**
and do not claim that the present master is faster or slower.

## Original protocol and observations

The [retained production comparison](../../../../benchmarks/results/fock-strategies/README.md)
contains matched, synchronized CPU/CUDA endpoints, raw samples, quantitative
errors and hardware/build provenance. Complete endpoint medians changed by
+0.77% on CPU and +0.28% on CUDA; all energies and raw J/K matrices are
unchanged, with complete force differences below 1e-14 Hartree/bohr.
This is an architecture non-regression study, not a speedup promotion.
No complete DFT SCF method is advertised.

`tools/benchmark_fock_strategies.py` runs one worker per revision/backend
against production Release builds with `GENERATIVEQC_CUDA_FAST_COMPILE=OFF`. It
records fixed-density direct CPU and DF CPU/CUDA matrices, complete RHF/UHF
SCF/forces, warm replay, changed geometry and four-item batch timings, with
all raw samples and matched-approximation numerical gates. The standalone
`benchmarks/fock_dispatch_probe.cpp` compiles against either revision and
compares the old contraction entry with the new provider boundary. Baseline
CUDA has no independent direct raw API, so its standard direct route is
compared through complete fused HF endpoints instead.

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:35:00 python tools/benchmark_fock_strategies.py \
  --baseline /path/to/3da5841-worktree --head "$PWD" \
  --build-relative .artifacts/overhead-cuda-build \
  --output .artifacts/fock-strategy-overhead
```

The runner never publishes its artifact directory or changes a production
selector. Accepted evidence is selected through the repository evidence policy
after accuracy and overhead review.
Optional `--case`, `--spin`, `--approximation` and `--endpoint` filters repeat
a selected endpoint with its normal setup/warmup. The retained CPU bundle
includes an alternating-order focused reproducer and the code-layout diagnosis
that led to the local DF function alignment hint. Python plan identities use
native-normalized controls and device indices, including accepted NumPy scalars.

## Decision and revisit conditions

Preserve independent J/K term preflight, correct scalar coefficients,
transactional SCF failure, and complete endpoint numerical acceptance.
Use the active tool and a fresh matched source whenever a provider
dispatch change is proposed; do not extrapolate old percentage differences
to other backends, workloads, or public methods.

Agent: ChatGPT
Model: GPT-6
