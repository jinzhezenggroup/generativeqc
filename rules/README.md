# Fast native structural lint

These ast-grep rules are intentionally small and **syntax-only**. They cover
maintained `src/` and `include/` C/C++/CUDA code, not vendored sources,
generated archives, benchmarks, or frozen test oracles. They do not replace
NVCC, clang-tidy, CUDA correctness tests, or the existing clang-format hook.

The hooks use `language: python` plus the pinned
`additional_dependencies: [ast-grep-cli==0.50.0]`. Both prek (on GitHub
Actions) and plain pre-commit (on external pre-commit.ci) provision an
isolated hook environment: neither requires a preinstalled `uvx` command.
One scan evaluates all the rules on the files passed by the hook; staged
commits only scan changed files. The current GitHub pre-commit workflow
runs all tracked files, so it scans all maintained native sources. There
is no CUDA toolkit, compiler, or GPU dependency.

To inspect locally:

```sh
prek run ast-grep-native --all-files
prek run ast-grep-rule-tests --all-files
# Optional direct CLI usage for developers who already use uv:
uvx --from ast-grep-cli==0.50.0 ast-grep scan --config sgconfig.yml src include
uvx --from ast-grep-cli==0.50.0 ast-grep test --config sgconfig.yml --skip-snapshot-tests
```

* **error**: `cpp-no-gets`, `cuda-no-unchecked-runtime-call`,
  `cuda-no-assert-api-status` and `cuda-no-divergent-block-barrier`
  reject unsafe constructs (subject to ordinary code review of any hit).
* **warning**: `cpp-review-strcpy`, `cpp-review-format-string`,
  `cuda-review-device-sync`, `cuda-review-device-reset`, and
  `cuda-review-discarded-graph-launch` request human review. CUDA Graph
  device-tail relaunches intentionally ignore a status in some existing
  callers; confirm failure propagation before changing the code.
* The native lint hook uses ast-grep's GitHub annotation format and
  `verbose: true` so warnings are **visible even when CI passes**.
  In particular, a synchronization or reset may be required for lifetime
  and correctness. Warnings do **not** fail pre-commit.
* No rule has an automatic rewrite. Replacing these calls blindly can change
  numerical behavior, synchronization, or memory safety.

`.cu`/`.cuh` are mapped to the C++ Tree-sitter parser. Some CUDA-specific
constructs such as kernel launches can parse imperfectly; the rules only
target ordinary C-style function calls. Rule tests include both matching and
non-matching cases and intentionally skip output snapshots.
