# Custom XC functionals

**Available:** `generativeqc.extensions.xc` is a versioned public builder
for combinations of **already-audited XC components**. The `compose`,
`named` and `inspect` functions preserve exact coefficient and spin
identities. A declared component or exact-exchange fraction does not itself
establish backend, grid, derivative or molecular endpoint qualification.

**Not available:** arbitrary user-defined density expressions, Python
callbacks in device execution, an XC expression decorator, or a
custom-primitive device ABI. Some metadata can be represented without
being executable as a complete MethodIR composition.

Follow the [programmable extension API](../extensions.md) for the
supported composition example and the current AOT/JIT boundary. For
capability interpretation, see the [reference](../../reference/capabilities.md).
