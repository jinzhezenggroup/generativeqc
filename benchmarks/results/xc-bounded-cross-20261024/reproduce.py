"""Recheck complete guarded-XC experiments, preserving every measured arm."""
import copy
import hashlib
import json
import lzma
import statistics as st
from pathlib import Path
from benchmarks.readme_defaults_qualification import qualify, _comparison
from benchmarks.readme_defaults_report import load_records

ROOT = Path(__file__).resolve().parent
if not __debug__:
    raise SystemExit("Evidence verification requires assertions; do not use -O.")
packet = json.loads(lzma.decompress((ROOT / "endpoints.json.xz").read_bytes()))
provenance = json.loads(lzma.decompress((ROOT / "provenance.json.xz").read_bytes()))
def verify_source_identities(provenance):
    """Check recorded compact identities, not unavailable historical Git objects."""
    sources = provenance["source_identities"]
    assert hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == provenance["source_identities_sha256"]
    assert set(sources) == {"baseline", "guarded"}
    assert sources["baseline"]["tree"] == "79b2a331c383bc45bc7640bb2916c2c7ed514676"
    assert sources["baseline"]["source_patch"] is None
    assert sources["guarded"]["tree"] is None  # Historical commit object was not recovered.
    assert sources["guarded"]["baseline_revision"] == sources["baseline"]["revision"]
    assert len(sources["guarded"]["changed_files"]) == 4
    for arm, source in sources.items():
        assert source["manifest_sha256"] == provenance["source_manifest_sha256"][arm]
        assert source["manifest_bytes"] == (1643599 if arm == "baseline" else 1643737)
        receipt_name = "stage24-source-verification.json" if arm == "guarded" else "validation-gemm-pr-source-verification.json"
        verified = json.loads(provenance["support"][receipt_name])
        assert verified["source_revision"] == source["revision"] and not verified["mismatches"]
        assert verified["files_verified"] == source["files_verified"] == (12425 if arm == "baseline" else 12426)


verify_source_identities(provenance)
for name, raw in provenance["support"].items():
    assert hashlib.sha256(raw.encode()).hexdigest() == provenance["support_sha256"][name]
refs = ROOT.parent / "readme-final-20261022"
oracles = load_records([refs / "oracles.json.xz"], schema="generativeqc.readme-defaults-oracle.v1")
reuse = json.loads((refs / "reference-reuse.json").read_text())
moved_refs = packet["moved_references"]
identities = {
    "baseline": ("f088d3fcd30435c125a5b92422e4d91088884977", "639ec53fd7786b909b53be2539724b3c1dc9ae6a85eb94578e4a29ec39bb1288"),
    "guarded": ("1ccc53419acd4c22d9496178a65fffd83f26d6b1", "4555f81fe4e277565cec634d642b41af99aa8c99ae108b0997ea78dec47e43db"),
}
report = {"qualification": {}, "timing": {}, "work": {}, "qualified_calls": 0, "records": 0}
for campaign in ("screen", "pairs", "gemm-matrix", "moved"):
    for mode in ("df", "direct"):
        folder = f"stage24-{campaign}-{mode}"
        records = {}
        for name, raw in sorted(packet["cohorts"][folder].items()):
            path = folder + "/" + name
            assert hashlib.sha256(raw.encode()).hexdigest() == packet["raw_record_sha256"][path]
            record = json.loads(raw)
            if record.get("schema") != "generativeqc.readme-defaults.v2":
                continue
            assert record["status"] == "complete", path
            arm = record.get("xc_cross_guard_control", {}).get("selection", "guarded")
            assert (record["source_revision"], record["native_build"]["library_sha256"]) == identities[arm], path
            assert record["validation_gemm_control"]["selection"] == "1"
            view = copy.deepcopy(record)
            extra = []
            if campaign == "moved":
                ref = moved_refs[record["method"] + "/" + mode]
                assert record["moved_geometry_bohr"] == ref["geometry_bohr"]
                for key in ("method", "mode", "orbital_basis", "basis_representation", "charge", "spin", "auxiliary_input_sha256"):
                    assert record.get(key) == ref.get(key), (path, key)
                assert [s["phase"] for s in record["samples"]] == ["cold", "warm", "moved", "moved-warm"]
                extra = [_comparison(s, ref["samples"][0], 96) for s in record["samples"][2:]]
                assert all(x["passed"] for x in extra), (path, extra)
                view["samples"] = view["samples"][:2]
                view["warm_repeats"] = 1
            q = qualify([view], oracles, reference_reuse=reuse)[0]
            assert q["status"] == "qualified", (path, q)
            key = folder + "/" + name
            report["qualification"][key] = {"qualification": q, "moved_comparisons": extra}
            report["work"][key] = [(s["phase"], s["iterations"], s["fock_builds"]) for s in record["samples"]]
            report["qualified_calls"] += len(record["samples"])
            report["records"] += 1
            records[name] = record
            if campaign == "gemm-matrix" and record["arm"] == "standard":
                pass  # Full work counts remain in the authenticated raw records.
        if campaign != "pairs":
            expected = {"screen": 6, "gemm-matrix": 14, "moved": 4}[campaign]
            assert len(records) == expected, (folder, len(records), expected)
            continue
        for method in ("pbe", "pbe0", "r2scan-3c"):
            arms = {arm: {} for arm in identities}
            for name, record in records.items():
                if record["method"] == method:
                    pair = int(name.split("-pair-")[1].split("-")[0])
                    arms[record["xc_cross_guard_control"]["selection"]][pair] = record
            if any(set(values) != set(range(5)) for values in arms.values()):
                raise AssertionError((method, mode, "incomplete pairs"))
            for pair in range(5):
                b, c = [arms[arm][pair] for arm in ("baseline", "guarded")]
                for key in ("protocol", "geometry_bohr", "orbital_basis", "basis_representation", "slurm_job_id", "thread_controls", "auxiliary_input_sha256"):
                    assert b.get(key) == c.get(key), (method, pair, key)
                assert {k:v for k,v in b["accelerator"].items() if k != "nvidia_smi"} == {k:v for k,v in c["accelerator"].items() if k != "nvidia_smi"}
                for x,y in zip(b["samples"], c["samples"], strict=True):
                    assert x["settings"] == y["settings"] and _comparison(x,y,96)["passed"]
            for phase in ("cold", "warm"):
                values = {arm: [st.median(s["seconds"] for s in records[p]["samples"] if s["phase"] == phase) for p in range(5)] for arm, records in arms.items()}
                medians = {arm: st.median(v) for arm,v in values.items()}
                mads = {arm: st.median(abs(x - medians[arm]) for x in v) for arm,v in values.items()}
                gain = 1 - medians["guarded"] / medians["baseline"]
                gate = max(.02, 2 * sum(mads[k]/medians[k] for k in arms))
                report["timing"][f"{method}/{mode}/{phase}"] = dict(values=values, medians=medians, mads=mads, gain=gain, gate=gate, passed=gain > gate)
                print("paired", method, mode, phase, medians, "gain", gain, "gate", gate, "pass", gain > gate)
assert report["qualified_calls"] == packet["complete_calls"] == 352
assert report["records"] == 108
assert len(report["timing"]) == 12
print("qualified calls", report["qualified_calls"], "records", report["records"])
