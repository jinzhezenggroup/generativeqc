# Custom backends

**Not available:** there is no stable public third-party backend plug-in
or backend-registration contract. The versioned public extension surface
covers audited scientific composition and a bounded TensorIR subset, not
an unrestricted CPU/GPU device interface.

The [provider-selection ownership](../provider_selection_boundaries.md)
describes internal scientific/execution responsibilities. The
[experimental OpenCL adapter](../opencl_backend.md) executes selected
generated scalar expressions but is **not** an OpenCL HF/DFT molecular
backend or an example of a supported third-party plug-in ABI.

Any future public backend interface will need explicit discovery,
supported operators and derivatives, resource/memory ownership, stream
and lifetime rules, error handling and independent numerical qualification.
