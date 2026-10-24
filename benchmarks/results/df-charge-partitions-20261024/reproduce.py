"""Authenticate and requalify every retained DF charge-partition endpoint."""
import copy
from collections import Counter
import hashlib
import json
import lzma
import statistics as st
from pathlib import Path

from benchmarks import readme_defaults_report as reader
from benchmarks.readme_defaults_qualification import _comparison, qualify

ROOT = Path(__file__).resolve().parent
if not __debug__:
    raise SystemExit("Evidence verification requires assertions; do not use -O.")


def unpack(path):
    return json.loads(lzma.decompress(path.read_bytes()))


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


packet = unpack(ROOT / "endpoints.json.xz")
provenance = unpack(ROOT / "provenance.json.xz")
previous = unpack(ROOT.parent / "xc-bounded-cross-20261024/provenance.json.xz")
base_raw = previous["source_manifests"]["guarded"]
base = json.loads(base_raw)
manifests = {}
for name, delta in provenance["source_manifest_deltas"].items():
    assert digest(base_raw) == delta["base_manifest_sha256"]
    files = dict(base["files"])
    for removed in delta["removed"]:
        del files[removed]
    files.update(delta["changed"])
    manifest = dict(source_revision=delta["source_revision"], files=dict(sorted(files.items())))
    assert digest(json.dumps(manifest, indent=2) + "\n") == delta["manifest_sha256"]
    manifests[name] = manifest
for name, raw in provenance["support"].items():
    assert digest(raw) == provenance["support_sha256"][name], name
support = provenance["support"]
build = manifests["stage25-source-inputs.json"]
tested = manifests["stage25-test-source-inputs-v2.json"]
verified = json.loads(support["stage25-source-verification.json"])
assert verified["source_revision"] == build["source_revision"]
assert verified["files_verified"] == len(build["files"]) and not verified["mismatches"]
assert set(build["files"]) == set(tested["files"])
changes = {p for p in build["files"] if build["files"][p] != tested["files"][p]}
assert changes == {"tests/native/df_md_j_probe.cu", "tests/python/test_df_md_j_cuda.py"}
verified_test = json.loads(support["stage25-test-source-verification-v2.json"])
assert verified_test["production_inputs_unchanged"] and not verified_test["mismatches"]
assert verified_test["files_verified"] == len(tested["files"])
assert verified_test["build_revision"] == build["source_revision"]
assert verified_test["test_revision"] == tested["source_revision"]

refs = ROOT.parent / "readme-final-20261022"
oracles = reader.load_records([refs / "oracles.json.xz"], schema="generativeqc.readme-defaults-oracle.v1")
reuse = json.loads((refs / "reference-reuse.json").read_text())
# The production reader's README size list remains unchanged. This archive also
# qualifies the repository's existing 108-atom fixture with its retained oracle.
reader.SIZES = (*reader.SIZES, 108)
extra_oracle = json.loads(support["stage25-larger/oracle.json"])
identities = {
    "baseline": ("1ccc53419acd4c22d9496178a65fffd83f26d6b1", "4555f81fe4e277565cec634d642b41af99aa8c99ae108b0997ea78dec47e43db"),
    "automatic": ("f01d512f9165d68e2aceb2c86ea2d8d4c3a99987", "399a8486b6cc5d930e1d60dc51c0074a9db3eeb7690cec22fcdefdf306046df4"),
}
records = {}
calls = 0
for path, raw in sorted(packet["records"].items()):
    assert digest(raw) == packet["raw_record_sha256"][path], path
    record = json.loads(raw)
    assert record["status"] == "complete", path
    arm = record.get("df_charge_policy_control", {}).get("selection", "automatic")
    if record["arm"] != "gpu4pyscf":
        assert (record["source_revision"], record["native_build"]["library_sha256"]) == identities[arm], path
        if path.startswith("stage25-cold-nsys/"):
            # Profiling uses the unmodified complete-endpoint driver; its
            # authenticated launch recipe retains the GEMM environment.
            assert "export GENERATIVEQC_SCF_VALIDATION_GEMM=1" in support["stage25-nsys.sh"]
        else:
            assert record["validation_gemm_control"]["selection"] == "1"
    view = copy.deepcopy(record)
    if path.startswith("stage25-moved-"):
        ref = packet["moved_references"][record["method"] + "/" + record["mode"]]
        assert record["moved_geometry_bohr"] == ref["geometry_bohr"]
        for key in ("method", "mode", "orbital_basis", "basis_representation", "charge", "spin", "auxiliary_input_sha256"):
            assert record.get(key) == ref.get(key), (path, key)
        assert [s["phase"] for s in record["samples"]] == ["cold", "warm", "moved", "moved-warm"]
        assert all(_comparison(s, ref["samples"][0], 96)["passed"] for s in record["samples"][2:])
        view["samples"] = view["samples"][:2]
        view["warm_repeats"] = 1
    q = qualify([view], oracles + [extra_oracle], reference_reuse=packet["extra_reference_reuse"])[0]
    assert q["status"] == "qualified", (path, q)
    records[path] = record
    calls += len(record["samples"])
assert calls == packet["complete_calls"] == 242 and len(records) == 83
for campaign, count in (("screen-df",6), ("screen-direct",6), ("pairs-df",30),
                        ("gemm-matrix-df",14), ("gemm-matrix-direct",14),
                        ("moved-df",4), ("moved-direct",4), ("larger",3), ("cold-nsys",2)):
    assert sum(path.startswith("stage25-" + campaign + "/") for path in records) == count

# Counts alone cannot establish method/mode/arm coverage: a duplicated valid
# record can otherwise replace a missing population even after rehashing.
methods = ("pbe", "pbe0", "b3lyp", "r2scan", "r2scan-3c", "m06-2x", "wb97m-v")
expected = {}
for mode in ("df", "direct"):
    expected["stage25-screen-" + mode] = Counter({
        (method, mode, "standard", policy): 1
        for method in ("pbe", "pbe0", "r2scan-3c")
        for policy in identities
    })
    expected["stage25-gemm-matrix-" + mode] = Counter({
        (method, mode, arm, "automatic"): 1
        for method in methods for arm in ("standard", "fast")
    })
    expected["stage25-moved-" + mode] = Counter({
        (method, mode, arm, "automatic"): 1
        for method in ("pbe", "r2scan-3c") for arm in ("standard", "fast")
    })
expected["stage25-pairs-df"] = Counter({
    (method, "df", "standard", policy): 5
    for method in ("pbe", "pbe0", "r2scan-3c") for policy in identities
})
expected["stage25-larger"] = Counter({
    ("r2scan-3c", "df", "standard", "baseline"): 1,
    ("r2scan-3c", "df", "standard", "automatic"): 1,
    ("r2scan-3c", "df", "gpu4pyscf", "automatic"): 1,
})
expected["stage25-cold-nsys"] = Counter({
    ("r2scan-3c", mode, "standard", "automatic"): 1 for mode in ("df", "direct")
})
actual = {}
for path, record in records.items():
    campaign = path.split("/", 1)[0]
    actual.setdefault(campaign, Counter())[
        (record["method"], record["mode"], record["arm"],
         record.get("df_charge_policy_control", {}).get("selection", "automatic"))
    ] += 1
assert actual == expected, "Incomplete or duplicated endpoint population"

for method in ("pbe", "pbe0", "r2scan-3c"):
    arms = {arm: {} for arm in identities}
    for path, record in records.items():
        if path.startswith("stage25-pairs-df/") and record["method"] == method:
            pair = int(path.split("-pair-")[1].split("-")[0])
            arms[record["df_charge_policy_control"]["selection"]][pair] = record
    assert all(set(values) == set(range(5)) for values in arms.values())
    for pair in range(5):
        b, c = [arms[arm][pair] for arm in ("baseline", "automatic")]
        for key in ("protocol", "geometry_bohr", "orbital_basis", "basis_representation", "slurm_job_id", "thread_controls", "auxiliary_input_sha256"):
            assert b.get(key) == c.get(key), (method, pair, key)
        assert {k:v for k,v in b["accelerator"].items() if k != "nvidia_smi"} == {k:v for k,v in c["accelerator"].items() if k != "nvidia_smi"}
        for x,y in zip(b["samples"],c["samples"],strict=True):
            assert x["settings"] == y["settings"] and _comparison(x,y,96)["passed"]
            assert (x["iterations"],x["fock_builds"]) == (y["iterations"],y["fock_builds"])
    for phase in ("cold", "warm"):
        values = {arm:[st.median(s["seconds"] for s in rows[p]["samples"] if s["phase"] == phase) for p in range(5)] for arm,rows in arms.items()}
        medians = {arm:st.median(v) for arm,v in values.items()}
        mads = {arm:st.median(abs(x-medians[arm]) for x in v) for arm,v in values.items()}
        gain = 1 - medians["automatic"] / medians["baseline"]
        gate = max(.02, 2 * sum(mads[k]/medians[k] for k in arms))
        print("paired", method, phase, medians, "gain", gain, "gate", gate, "pass", gain > gate)
print("qualified calls", calls, "records", len(records))
