# Native CLI initial-guess initializer follow-up

Measured source: `ffc6537b15b7df10240a1b9bb94f6f6190715408`. Baseline: `40c31f7bd6130903ade0cd87ef60fb06c56d9d8b`.
This is a fresh qualification of the descriptor-initialization warning fix in
PR #2797. It preserves the original source `824b94bde` evidence in
`../native-cli-initial-guess-20261026/`; those older measurements are not relabelled.

All eight CPU and eight CUDA ABBA/BAAB complete CLI gates pass the unchanged
1.05 ratio limit. Maximum ratios are 1.000952711751
(CPU) and 1.019518771843 (CUDA). Four water/STO-3G
cases retain 21 samples after three warmups per block: 1,344 timed fresh-process
endpoints, including 1,008 force publications. Startup, preparation, SCF, output
and teardown are timed. Metadata/iterations match exactly; E/F parity is <=1e-12.
Repeated fresh processes are not prepared warm starts.

The rebuilt installed products pass 24 CUDA and two CPU independent CLI E/F
publications. Maximum errors are 2.1316e-14
Hartree and 3.0209e-08 Hartree/Bohr,
against gates 1e-8 and 1e-6. Checks retain base/moved H2, Direct/DF RHF and Direct
PBE, explicit initial-guess policies, exact/one-byte-below preparation caps and
CUDA HF/LDA refusal. Installed C++ diagnostic copy/move/destruction and oracle
probes pass on both backends with PATH=/nonexistent. CUDA memcheck: zero errors.

The CPU Release rebuild passes 119/119 CTests and 24 targeted Python tests plus
25 subtests, including the warnings-as-errors CLI-defaults probe. CUDA 12.9.1
sm_120 Release builds all configured stationary manifests with fast compile OFF
and verified sccache 0.16.0; the preliminary native contract passes. All GPU
execution used Slurm job 14434 on n1/main/gpu:5090:1, finite time 01:30:00,
preserving assigned visibility 1.

## Reproduction and limits

Decode `samples.json.gz` with Python gzip/json for every timed result, CUDA
warmups, installed outputs/references, exact runner/probe inputs and build/test/
integrity receipts. The separate `build-receipts.json.gz` retains complete CPU/CUDA
configure/build/install logs. Adapt the recorded absolute paths, rebuild the recorded
source through a verified compiler cache, run the two paired CLI scripts, and
run the installed CLI/SDK probes with a finite srun allocation for CUDA. The
independent reference generator is retained with the original publication.

Preparation-byte caps are not full device/process admission. Independent
Fock/transfer/launch counts, full allocation/provider/RSS peaks, per-iteration
residual traces and cold compilation costs remain unavailable. Full performance
promotion and the broader R2/#2227 migration remain unqualified.
