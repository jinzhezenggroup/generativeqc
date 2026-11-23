# Decision: expose the accepted CPU KS derivative owner to the SDK

Status: implemented
Date: 2026-11-22

The [CPU bridge](2026-11-12-native-cpu-hessian-bridge.md) already serves the
Python and CLI B3LYP Hessian/HVP frontends. Export its existing v1 wire layout
as a C-compatible SDK contract, with a native default initializer. Keep the
orbital-frame transport private. `Batch::prepare_named_ks` uses the native
catalog preparation ABI, including retained device admission if requested;
the Hessian and HVP wrappers then call the existing CPU bridge. The existing
CUDA methods keep their signatures and remain CUDA-only.

Do not derive catalog identity, restage an electronic-only calculation, or
rerun SCF in the SDK helper. Bind the derivative request to the original item
index and caller-expected elements, coordinates, charge and multiplicity. A
successful warm move makes the prior geometry stale. The native execution
continues to authenticate the live sources, count work, and publish only after
all numerical and resource checks pass. No scientific domain expands here.

The SDK test uses public headers and exercises a named, corrected CPU method,
raw dense/HVP consistency, a changed-geometry replay, and failure atomicity.
The parity test uses the same H₂ B3LYP-D3(BJ) RKS geometry, basis, grid and
direction across frontends. It compares total energy, HVP and D3 pair work;
the native host Hessian and Python tests retain independent derivative checks.
In local CPU qualification based on `aff6ad925`, six targeted CTest cases and
137 Python cases passed. The separately compiled installed C/C++ clients and
installed CLI also ran without Python or compiler discovery paths, with the
linked OpenBLAS provider on the library search path. This is not a new CUDA,
performance or broader-method qualification.

Refs #2227 (R3/R5/R6).
