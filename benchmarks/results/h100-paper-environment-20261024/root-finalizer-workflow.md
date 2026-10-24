# Root-only replacement-Job workflow

This document is a handoff, not permission to create a Job. The environment
writer does not submit, restart, or publish a real release marker. Preserve r1.

## Prepare before considering the single replacement

Freeze and independently review these staged files under task root
`/inspire/ssd/project/chemicalreaction/czxs25220150/h100-paper-environment-20261024`:

- `scripts/adapter-prepared-v5/qz_smoke.py`:
  `66bd120eea89abf44a5a4e4a77616afc91979c7fb67822eab1a58011e6bb7b66`.
- `scripts/gate-wait-v3/gate_wait.py`:
  `b35b3f18a054922f47e1700a2918544a9b2e63131f213eebd596e3b7bb1f865c`.
- `scripts/gate-wait-v3/finalize_release.py`:
  `76edfc8e70f90862c1c4fbff156d54c08a7fd8df46b49b21bafd4016e59b492b`.
- `scripts/gate-wait-v3/job-entry.sh`:
  `ea38ac6f410f8397fcbc7dd40f1d7684611b3d98a6501571f1913f4aabde4016`.
- `assets/prepared-smoke-plan-r2-v5.json`:
  `cae38b70227e56a66e7dcda455a83b0d95b832ebcc33270f4ed9786b45cc5941`.

The proposed output namespace is
`output/h100-smoke-gate-20261024-9dcc-r2`. It must not exist before entry. The
entry names `h100-paper-smoke-20261024-9dcc-r2`; root must keep the chosen Job name
consistent. The immutable prepared plan also names this exact expected Job; consistent renaming of all observed inputs is refused. The finite command is:

```sh
/usr/bin/timeout --kill-after=30 14400 /bin/bash /inspire/ssd/project/chemicalreaction/czxs25220150/h100-paper-environment-20261024/scripts/gate-wait-v3/job-entry.sh
```

Root alone rechecks quota/availability/budget, dry-runs and submits if authorized.
Request/stored priority scales are not assumed equivalent; retain actual readback.

## During the 300-second actual release window

Read the complete actual `preflight.json` and hash its file bytes. Verify one
H10080/SM90, runtime hostname, actual package/thread values and zero endpoints.
Query this exact Job's `status` and `instances` in `分布式训练空间` using account
`qz`. Keep both successful scrubbed JSON response envelopes byte-for-byte in
root-controlled files; do not synthesize missing fields.

The supported instances envelope contains `data.name` and one `data.items` row
with `status=instance_running`, `type=instance_type_worker`, `node`, `rank=0`.
Opaque Job UID and instance ID remain null/UNAVAILABLE. The node comes from the
platform row/status; the hostname comes from the unique actual runtime preflight.
Neither is mislabelled as an unexposed API field.

Create a separate query-context JSON receipt from the actual CLI invocation:

```json
{
  "account_alias": "qz",
  "workspace": "分布式训练空间",
  "job_name": "h100-paper-smoke-20261024-9dcc-r2",
  "scope_source": "root_cli_query_arguments",
  "status_sha256": "ACTUAL_STATUS_JSON_FILE_SHA256",
  "instances_sha256": "ACTUAL_INSTANCES_JSON_FILE_SHA256"
}
```

Supply those actual files, the preflight path/hash, a fresh root-control output
directory, and the actual observation UTC to the staged finalizer. Its CLI
requires `--adapter`, `--adapter-sha256`, `--prepared-plan`,
`--prepared-plan-sha256`, `--preflight`, `--preflight-sha256`, `--status`,
`--instances`, `--query-context`, `--observed-utc`, `--output`, and
`--release-marker`. The marker is `<actual r2 output>/release.json`.

The finalizer validates both successful envelopes, exact Job/project/group and
1 GPU/1 node/20 CPU/200 GiB/32 GiB shm/H10080 resources, query-scope bindings,
the unique running rank-0 worker, different node from `qb-prod-gpu095`, actual
hostname/preflight association and the zero-endpoint 300-second gate. It keeps
original envelopes separately, writes normalized proof with explicit unavailable
IDs, validates the unchanged final plan, and atomically publishes a no-clobber
marker. It checks for expiry/terminal/old marker both before validation and
immediately before publication. It does not query an API or change resources.

If proof is incomplete, or the window expires, preserve failure without a late
marker, limit increase, or automatic retry. A device-only preflight is not E/F.

## After root release

Exactly three fresh endpoint processes may run: native HF, GPy HF, CBS C2.
The original source/scientific functions, 32 GiB CBS budget, fixed source/library
hashes and stopping caps remain unchanged. Collect complete outputs and original
numeric bytes, then run CPU integrity and independent-reference replay. Keep
unavailable C2 displaced originals explicit and exclude setup times from timing
cohorts. No publication/merge completion follows from a process exit alone.
