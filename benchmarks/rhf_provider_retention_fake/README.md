# RHF provider observer fake ELF harness

This directory is a diagnostic-only Linux harness for
`rhf_provider_retention_probe.cpp`. It does not use a CUDA device, production
library or scientific endpoint, and a PASS is not CUDA or scientific acceptance.

The fake provider exports each of the seven observed functions twice: once with
the exact production import version and once with the deliberately different
default `FAKE_PROVIDER_9.9` version. The driver imports only the production
versions. A baseline run without the observer and a run with `LD_PRELOAD` must
produce byte-identical JSON, including arguments, output sentinels, statuses and
provider-side call audits. This catches an observer that intercepts the right
version but forwards through the provider's unrelated default definition.

The harness also checks:

- one nested `cudaMemGetInfo@libcudart.so.12` call from cuBLAS is tagged
  `provider_internal`;
- provider and symbol-resolution failures remain visible;
- the journal is a fresh `O_EXCL` file with mode `0600`;
- the 512-event limit, dropped-event accounting and terminal counters;
- a single injected host write failure followed by a writable terminal; and
- the normal report reconstructs one complete 200-byte delta with a 250-byte
  source allowance.

Run it only in a fresh absolute result directory on Linux with CUDA SDK headers,
`readelf`, Python 3 and either `sccache >=0.16.0` or `ccache`:

```bash
benchmarks/rhf_provider_retention_fake/build_and_run.sh \
  /absolute/fresh/result-directory \
  <full-observer-commit>
```

The script fails before compilation when no approved compiler-cache launcher is
available. It records compiler, SDK-header, launcher, source, binary, case-exit,
symbol-version and runtime evidence under the result directory. The fixed
production a308 source/library/endpoint identities are provenance only; the fake
provider and driver have their own independent hashes in the result.
