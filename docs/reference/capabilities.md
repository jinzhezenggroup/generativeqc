# Capability sources

Do not maintain handwritten copies of generated capability tables.

The canonical native ABI/provider registry is `manifests/public_methods.json`; tooling generates [the native ABI table](../public_methods.md). DFT scientific discovery instead comes from the compiler MethodIR catalog and generated pinned-Libxc metadata, with native lowerer gates deciding whether each RKS/UKS selector is executable.

Compiler shell/code-generation capability evidence is tracked in `docs/codegen_capabilities.json`. CUDA semantic ownership is tracked in `docs/cuda_ownership/`.

A low-level capability does not by itself prove end-to-end scientific qualification; user-facing support still requires the relevant validation and fail-closed execution contracts.

The generated [bulk Libxc import report](../libxc_bulk_import.md) and [Maple importer coverage](../libxc_maple_coverage.md) retain their registered paths.
