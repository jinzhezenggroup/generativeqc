# Extending GenerativeQC

The public, versioned [programmable extension API](../extensions.md)
(`generativeqc.extensions`) lets advanced users **compose already-supported
scientific primitives**. Its XC and method builders share the canonical
scientific specs and MethodIR with built-in methods; the TensorIR subset also
supports explicitly requested, capability-checked CPU JIT compilation.

This is **not** a general third-party plug-in ABI. An extension's scientific
identity, representability, successful type check and actual molecular
execution qualification are separate. Constructing an extension does not
automatically make it runnable on CPU/CUDA or enable a force/Hessian endpoint.
Check the [capability contract](../../reference/capabilities.md) for the
fully resolved execution context.

## Extension boundaries

- [Custom electronic methods](custom-method.md) — compose audited primitives,
  resolve and verify MethodIR; arbitrary new method primitives are not exposed.
- [Custom XC functionals](custom-functional.md) — compose audited XC components;
  custom expression callbacks and device primitives remain unsupported.
- [Custom backends](custom-backend.md) — no public third-party backend
  registration interface; experimental adapters have narrower contracts.

The [programmable API guide](../extensions.md) is authoritative for
supported entry points and limitations. Undocumented internal Python/C++
classes are not a stable extension interface.

```{toctree}
:hidden:
:maxdepth: 1

custom-method
custom-functional
custom-backend
```
