# Fast native structural lint

These ast-grep rules are intentionally small and **syntax-only**. They cover
maintained `src/` and `include/` C/C++/CUDA code, not vendored sources,
generated archives, benchmarks, or frozen test oracles. They do not replace
NVCC, clang-tidy, CUDA correctness tests, or the existing clang-format hook.

The pre-commit hooks use the existing `uvx`/prek setup and pin
`ast-grep-cli==0.50.0`. One scan evaluates all the rules on the files
passed by pre-commit; staged commits only scan changed files. The current
GitHub pre-commit workflow runs all tracked files, so it scans all maintained
native sources. There is no CUDA toolkit, compiler, or GPU dependency.

To inspect locally:

```sh
prek run ast-grep-native --all-files
prek run ast-grep-rule-tests --all-files
uvx --from ast-grep-cli==0.50.0 ast-grep scan --config sgconfig.yml src include
uvx --from ast-grep-cli==0.50.0 ast-grep test --config sgconfig.yml --skip-snapshot-tests
```

* **error**: `cpp-no-gets` is forbidden.
* **warning**: `cpp-review-strcpy`, `cpp-review-format-string`,
  `cuda-review-device-sync`, and `cuda-review-device-reset` need human
  review. Warnings do **not** fail pre-commit. In particular, a synchronization
  or reset may be required for lifetime and correctness.
* No rule has an automatic rewrite. Replacing these calls blindly can change
  numerical behavior, synchronization, or memory safety.

`.cu`/`.cuh` are mapped to the C++ Tree-sitter parser. Some CUDA-specific
constructs such as kernel launches can parse imperfectly; the rules only
target ordinary C-style function calls. Rule tests include both matching and
non-matching cases and intentionally skip output snapshots.
