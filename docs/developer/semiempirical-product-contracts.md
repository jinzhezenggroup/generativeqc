# Semiempirical product/derivative requirement contracts

Issue: [#875](https://github.com/jinzhezenggroup/generativeqc/issues/875)

The shared `SemiempiricalMethodIR` is an immutable scientific composition
graph, **not** an SCC solver, compiler-target admission policy, runtime
certificate, or proof that an analytic nuclear force is available.

## CPU-side contract

```python
from generativeqc_compiler.method.semiempirical import (
    ProductSpec, semiempirical_from_xtb,
)
from generativeqc_compiler.method.semiempirical_requirements import (
    semiempirical_product_requirements,
)
from generativeqc_compiler.method.xtb import resolve_xtb_method

method = semiempirical_from_xtb(
    resolve_xtb_method(
        "GFN2-xTB", requested_products=("energy", "nuclear-gradient")
    )
)
requirements = semiempirical_product_requirements(
    method, ProductSpec("nuclear-gradient", 1)
)

for node in requirements.nodes:
    # Bound by immutable parameter identity, not table names alone.
    print(node.node_id, node.parameter_resources, node.state_reads)
print(requirements.unresolved_global_obligations)
```

Each node binds its parameter **roles to immutable resource identities** and
its state reads/writes to typed `StateField` objects, including scope,
components, dtype and spin semantics. DAG order is inherited from the
canonical common IR. Unknown derivative mechanism names, ambiguous fixed-point
state writers, unrequested products, unknown product derivative semantics and
gradient requests without an energy product all fail closed.

The initially understood product contracts are `energy` (order 0) and
`nuclear-gradient` (order 1). New requested products can be represented
through the base IR, but require a separate explicitly reviewed scientific
contract before the requirements API will accept them.

## Local derivative route is not a complete force

Legacy GFN declarations are interpreted conservatively:

| Legacy primitive declaration | Typed mechanism |
| --- | --- |
| `energy` | `primal-energy` |
| `hamiltonian` | `hamiltonian` |
| `integral-adjoint` | `integral-adjoint` |
| `nuclear-gradient` | `explicit-coordinate-vjp` |

Future shared primitives may declare `state-adjoint` or
`explicit-coordinate-vjp`. Unrecognized declarations fail closed; no
untrusted family-private string is silently interpreted as qualified.

For a gradient, a node without a declared local route is reported in
`unresolved_local_derivative_nodes`. A fixed-point state graph adds
`stationary-state-response`; **every** total-gradient request adds
`total-gradient-assembly`. This deliberately prevents inference of a complete
gradient from per-node coordinate VJPs or integral adjoints alone. A scientific
owner must provide independent response, Pulay, energy/force and stationary
closure evidence outside this representation before enabling a public force.

`require_closed_derivative_contract()` rejects unresolved obligations.
Successful energy representation is likewise **not** proof of public runtime,
backend, force or performance qualification.

## Identity and family boundaries

The new requirements are a deterministic **derived view** of the existing
`semiempirical-method-ir-v1` graph. No original GFN v1 payload, legacy
`XtbMethodIR.identity`, alias handling or compiled cache key is changed.
Parameter revisions and resource digests change the derived identity;
display-name aliases do not.

GFN1 and GFN2 resolve through their existing audited xTB adapters; neither
runtime is redirected. Independent schematic DFTB and NDDO fixtures only show
that typed source/state/derivative obligations can be represented without a
family discriminator in common code. They are **not** production DFTB, AM1,
PM3, PM6, or GFN force validations. Family-specific physical equations and
numeric parameter data remain with family resolvers.

Run the CPU-only checks:

```bash
PYTHONPATH=python python -m pytest -q \
  tests/python/test_semiempirical_method_ir.py \
  tests/python/test_semiempirical_field_types.py \
  tests/python/test_semiempirical_product_requirements.py
```

No CUDA device is needed because this work only adds immutable requirement
extraction and CPU-side contract tests. Any future CUDA lowering or production
device-route change must undergo its own validation.
