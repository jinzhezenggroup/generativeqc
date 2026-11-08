# Decision: Recover the same pinned CI tools through the official asset API

Status: implemented
Date: 2026-11-08

## Problem

The derivative-AD integration's required CI repeatedly failed before compilation
or tests while downloading ccache 4.14 and CPython 3.11.14. One failed-jobs rerun
still left seven ccache HTTP 504 failures and one CPython HTTP 504 failure. A
hosted-runner probe received four 504 responses and no bytes from the original
ccache release URL, then downloaded and executed the identical asset through
GitHub's official release asset API.

This was an observed transport failure, not evidence of a repository-wide
service incident or permission to skip build and scientific checks. Repeated
unchanged reruns did not repair the transport path.

## Decision

`.github/scripts/prepare_ci_tools.py` owns the two narrowly bounded recovery
paths. Existing workflow conditions, jobs, dependencies, matrices, timeouts,
cache expressions and subsequent build/test commands stay unchanged.

For ccache, attempt the original release URL first, with the existing finite
curl retry budget. Only the explicitly enumerated DNS, connection, partial
transfer, HTTP, timeout and related transport exit codes may reach the alternate
API path. Local I/O, usage, certificate and size-limit errors stay fatal.
Failure of the alternate path is also fatal.

For Python, first run the original `uv venv --python 3.11`. Recovery requires
exit status 2 and the exact pinned CPython URL. The eligible diagnostics are
the observed `504 Gateway Timeout`, or the complete observed cause chain
`client error (SendRequest)`, `http2 error`, and `stream error received: refused
stream before processing any application logic`. The HTTP/2 case was added
after two hosted attempts reproduced that exact pre-response failure. Matching
an arbitrary HTTP/2 error or an arbitrary uv exit status is insufficient.
Unrelated failures propagate. Refuse an existing `.venv`, including an
unexpected directory left by the failed normal path.

Both paths use the same pinned asset, not a substitute version or mirror:

| Tool | Official release asset ID | Bytes | SHA256 |
| --- | ---: | ---: | --- |
| ccache 4.14, Linux x86_64 glibc | 526043228 | 1,344,580 | `45a91165db7092e67c6208ada03f54700e684c4cd3735f9031de95669ed9272c` |
| CPython 3.11.14, 20260211, Linux x86_64, install_only_stripped | 354545489 | 31,215,121 | `1fb6a89daa750f7eff00eb65d901c3c236713812eb9c8c7e07ecdfcd78203c8e` |

Check the exact size and digest before extraction or execution. The download
also has a maximum file size. Check ccache's executable version before and
after installation. Keep the complete recovered Python installation under
`RUNNER_TEMP` until the job ends; the virtual environment depends on its
interpreter, standard library and shared libraries. Use the explicit recovered
executable with additional Python downloads disabled, then verify the exact
Python version and base identity. Do not copy only the executable or delete its
installation after creating the virtual environment.

Read-only `GH_TOKEN` is scoped to helper steps. Pass the API headers through
curl's standard input, outside command arguments, and remove the token from all
child environments. Require the HTTPS GitHub API host and HTTPS redirects;
never forward credentials with `--location-trusted`.

## Invariants

- Keep the normal download/environment path first and the fallback bounded.
- Download, extraction, identity or prerequisite failures must not skip a gate.
- Preserve every pinned upstream identity and executable/runtime check.
- Preserve compiler-cache contents, cache correctness and job-local interpreter
  lifetime. A workflow-content cache key can change naturally; its key and
  restore-prefix expressions are not bypassed or weakened.
- Keep scientific source identities, historical measurements, evidence limits
  and public qualification decisions independent of this CI transport repair.
- Exercise actual transport behavior in regression tests. The original
  `test_ci_ccache_download.py` discovered only inline workflow curl commands;
  moving those commands into the helper exposed that discovery assumption as a
  collection error. Tests must reach the executed installer path, retain finite
  budgets and fatal-error coverage, and avoid pinning helper source spelling.

## Rejected alternatives

Increasing retries indefinitely would mask an unavailable route and consume
runner time. Changing tool versions, taking an unverified archive, using an
unrelated mirror, removing integrity checks, clearing caches or suppressing
failed jobs would change contracts beyond the diagnosed transport problem.
A generic catch-all Python fallback would also hide real environment failures.

## Evidence and its limits

- [Hosted transport probe](https://github.com/njzjz-bot/vibeqc/actions/runs/42467479220):
  real ccache release-URL 504 responses, followed by a real API download with the
  pinned size/digest and successful execution of ccache 4.14.
- [Exact-candidate preflight](https://github.com/njzjz-bot/vibeqc/actions/runs/42469151298)
  checked production helper revision `4656b0f48ed349ca9a52dbccb292a04bc1b79665`.
  Initial 504 triggers were explicitly injected. API transfers, digest/size
  checks, extraction, ccache execution, full CPython execution, virtual
  environment creation and imports of `ssl`, `lzma`, `sqlite3` and `venv` were
  real. The normal uv path, retained interpreter lifetime, cache markers and
  original catalog/evidence checks also passed.
- [Production CodSpeed job](https://github.com/jinzhezenggroup/generativeqc/actions/runs/42469371650/job/128696321352):
  real original-URL 504 responses recovered through the ccache API with the
  exact pinned identity. The subsequent native build and job completed. Its
  Python environment used the normal uv download path.
- [Production Sphinx job](https://github.com/jinzhezenggroup/generativeqc/actions/runs/42469371654/job/128696320924):
  the normal uv path, dependency installation and complete documentation build
  passed. This is not an explanation for the separate external Read the Docs
  failure, whose detailed logs were unavailable.
- Both attempts of the later [download-contract preflight](https://github.com/njzjz-bot/vibeqc/actions/runs/42472314590)
  failed before tests while downloading that same pinned Python asset, with uv
  exit status 2 and the complete refused-HTTP/2-stream cause chain above. The
  first helper revision correctly rejected it under its 504-only boundary.
  Repeating the job once did not repair the route; this observed repetition
  justified adding that one explicit transport case to the same-asset recovery
  policy. It does not justify a catch-all uv failure handler.
- Python API recovery is verified by the explicitly scoped live preflight;
  these reviewed production logs do not establish an instance of that fallback.
  None of these transport checks is a new GPU endpoint or timing campaign.

## Revisit when

Revisit the pinned assets and tests when the selected ccache or uv/Python
version changes, when GitHub's asset API contract changes, or when observed
transport evidence shows this path is no longer necessary. Preserve equivalent
identity, lifetime and failure guarantees when simplifying the implementation.
Unknown failure diagnostics remain fatal. Expanding the accepted diagnostics
requires observed transport evidence and executable positive/negative cases.

Agent: /root (ChatGPT)
Model: GPT-6 Astra Pro
