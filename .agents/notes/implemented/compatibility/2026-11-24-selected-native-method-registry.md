# Decision: Require the selected native named method registry

Status: implemented
Date: 2026-11-24

Python ships the shared method manifest for public aliases and ABI constants.
Those constants cannot establish that the chosen binary assigns the same
canonical carrier to a numeric ID. The public capability query and `Calculator`
therefore require the installed named lookup to confirm the carrier before
querying capabilities on that same binary.

The former numeric-only route accepted an unverifiable name-to-ID mapping when
an older binary lacked the named ABI. Retiring it makes that mismatch a pairing
error rather than another discovery policy. The C numeric-ID API itself is
unchanged. Public alias presentation, explicit compiler/automatic Libxc
resolution, native KS graph discovery and contextual admission keep their
separate contracts; this change neither lowers graphs nor qualifies new science.

`tests/python/test_selected_native_method_capabilities.py` exercises the
installed carrier mapping and injects missing, failed and inconsistent native
publications without allowing alternate discovery or numeric-only queries.
Refs #2227.
