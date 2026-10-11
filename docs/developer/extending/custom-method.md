# Custom electronic methods

**Available:** `generativeqc.extensions.method` publicly exposes
`compose`, `named`, `resolve`, `inspect` and `verify` for supported
scientific primitives and canonical MethodIR. Equivalent built-in and
user-defined compositions share a scientific identity; verification checks
an explicit backend capability rather than assuming runnable support.

**Not available:** a stable API for adding arbitrary scientific primitives,
user callbacks or unrestricted method/backend/derivative combinations.
Representable MethodIR and a successful type check are **not** proof of
a qualified molecular energy, force or response endpoint.

Use the [programmable extension API](../extensions.md) for the canonical
builder example, and the [Electronic Method IR](../electronic_method_ir.md)
for compiler ownership. The [capability reference](../../reference/capabilities.md)
explains how actual public execution support is determined.
