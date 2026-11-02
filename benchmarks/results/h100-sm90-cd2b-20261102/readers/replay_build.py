"""Recover and verify terminal build metadata; no molecular/science acceptance."""
from pathlib import Path
import hashlib
import json
import shlex
import subprocess
import sys
import tarfile

TASK = Path(__file__).resolve().parent
REVISION = "cd2b4514be0a38c4920934481b1f02d157a694b4"
TREE = "c76f5bd3cfa62416255c5d6914cf701d34e10a8d"
PROFILES = ["pbe_rks", "pbe0_rks", "b3lyp_rks", "r2scan_rks", "m06_2x_rks", "wb97mv_rks", "pbe_uks", "pbe0_uks"]


def check(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    check(sys.version_info >= (3,12), "Python3.12 filtered extraction required")
    manifest_path = TASK / "production-build-v1-download-manifest.json"
    manifest = read(manifest_path)
    check(manifest["status"] == "COMPLETE_TERMINAL_PRODUCTION_BUILD_METADATA_ARCHIVE" and manifest["source_revision"] == REVISION, "Wrong archive identity")
    archive = TASK / manifest["archive"]
    check(archive.parent == TASK and archive.stat().st_size == manifest["bytes"] and sha(archive) == manifest["sha256"], "Archive bytes differ")
    roots = {"production-build-v1", "production-build-v1-executed-producers", "production-build-v1-unverified-producers", "production-build-v1-available-inputs", "production-build-v1-archival-producers", "production-build-v1-archival-admissions"}
    with tarfile.open(archive) as stream:
        members = stream.getmembers()
        check(len(members) == len(manifest["files"]) and {m.name for m in members} == set(manifest["files"]), "Duplicate/missing archive member")
        for member in members:
            name = Path(member.name)
            check(member.isfile() and not name.is_absolute() and ".." not in name.parts and name.parts[0] in roots and not (TASK / name).exists() and member.size == manifest["files"][member.name]["bytes"], "Unsafe/existing target")
        stream.extractall(TASK, filter="data")
    check(all((TASK / n).stat().st_size == v["bytes"] and sha(TASK / n) == v["sha256"] for n,v in manifest["files"].items()), "Recovered original differs")
    out = TASK / "production-build-v1"
    receipt_path = out / "build-receipt.json"
    receipt = read(receipt_path)
    check(sha(receipt_path) == manifest["build_receipt_sha256"], "Build receipt differs")
    observed_path = TASK / "production-build-v1-archival-admissions/observed-platform-status.json"
    observed = read(observed_path)
    check(sha(observed_path) == manifest["platform_status_sha256"] and observed["success"] is True and observed["data"]["name"] == receipt["job"] and observed["data"]["status"] == manifest["platform_status"], "Platform response differs")
    # A failed/partial build is preserved above, then rejected as an admission.
    check(manifest["platform_status"] == "SUCCEEDED" and manifest["recorded_provenance_errors"] == [] and receipt["status"] == "PASS_SM90_PRODUCTION33_ARTIFACT_BUILD_ONLY_NOT_SCIENTIFIC_ACCEPTANCE", "Failed/incomplete/unbound build")
    check(receipt["source_revision"] == REVISION and receipt["source_tree"] == TREE and receipt["profiles"] == PROFILES, "Source/profile inventory differs")
    repo = REPO
    check(subprocess.check_output(["git", "-C", str(repo), "rev-parse", REVISION + "^{tree}"], text=True).strip() == TREE, "Frozen Git tree differs")
    check(set(receipt["scripts"]) == {"build-production.py", "build-production-job.sh"}, "Build producer inventory differs")
    for name,digest in receipt["scripts"].items():
        check(sha(TASK / name) == digest == sha(TASK / "production-build-v1-executed-producers" / name), "Executed producer differs")
    stage_path = TASK / "production-build-v1-available-inputs/source-staging-v1.json"
    stage = read(stage_path)
    prep_path = TASK / "production-build-v1-available-inputs/preparation-v1.json"
    check(sha(stage_path) == receipt["source_stage_sha256"] and stage["source_revision"] == REVISION and stage["source_tree"] == TREE and stage["preparation_sha256"] == sha(prep_path), "Staging/preparation differs")
    check(all(stage["files"][n]["sha256"] == h for n,h in receipt["scripts"].items()), "Staged/actual producer differs")
    check(receipt["allocation_cgroups"] == {"cpu.max":"2000000 100000", "memory.max":"214748364800"} and "H100" in receipt["device"] and ", 9.0," in receipt["device"], "Allocation/device differs")
    check(receipt["cache_version"] == "sccache 0.16.0" and [r["stage"] for r in receipt["stages"]] == ["configure", "build", "aot-audit"] and all(r["status"] == "COMPLETE" and r["returncode"] == 0 for r in receipt["stages"]), "Cache/stage result differs")
    commands_path = out / "compiler-commands.txt"
    check(sha(commands_path) == receipt["compiler_commands_sha256"], "Compiler commands differ")
    compiles = [shlex.split(line) for line in commands_path.read_text().splitlines() if " -c " in line and ("/g++-12 " in line or "/nvcc " in line)]
    check(len(compiles) == receipt["planned_compiler_actions"] > 0 and all(any(p.endswith("/env/bin/sccache") for p in row) for row in compiles), "Compiler launcher/planned inventory differs")
    check(not any("GENERATIVEQC_TEST_HOOKS" in p or "--use_fast_math" in p or "-ffast-math" in p for row in compiles for p in row), "Test hooks/fast math in production")
    cache = {}
    for line in (out / "configured-build/CMakeCache.txt").read_text().splitlines():
        if line and not line.startswith(("#", "//")) and "=" in line and ":" in line.split("=",1)[0]:
            key,value = line.split("=",1)
            cache[key.split(":",1)[0]] = value
    required = {"CMAKE_BUILD_TYPE":"Release", "CMAKE_CUDA_ARCHITECTURES":"90", "GENERATIVEQC_CUDA_ARCHITECTURES":"90", "GENERATIVEQC_AOT_PROFILE":"portable_cuda", "GENERATIVEQC_BUILD_TESTS":"OFF", "GENERATIVEQC_ENABLE_AOT_SHELLS":"ON", "GENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT":"ON", "GENERATIVEQC_STATIONARY_AOT_PROFILES":";".join(PROFILES)}
    check(all(cache.get(k) == v for k,v in required.items()), "Production CMake policy differs")
    mandatory = {"production-build-v1/" + n for n in ["build-receipt.json","configured-build/CMakeCache.txt","configured-build/build.ninja","compiler-commands.txt","aot-audit.json","sccache-before.txt","sccache-after.txt","nvcc-version.txt","compiler-version.txt","production-exports.txt","production-ldd.txt"]} | {"production-build-v1-executed-producers/" + n for n in ["build-production.py","build-production-job.sh"]} | {"production-build-v1-available-inputs/" + n for n in ["source-staging-v1.json","preparation-v1.json"]} | {"production-build-v1-archival-admissions/observed-platform-status.json","production-build-v1-archival-producers/archive-build-v3.py"} | {"production-build-v1/aot-manifests/generativeqc_stationary_" + profile + suffix + ".json" for profile in PROFILES for suffix in ["","_spd"]}
    mandatory |= {"production-build-v1/"+stage+suffix for stage in ["configure","build","aot-audit"] for suffix in [".stdout",".stderr"]}
    check(mandatory <= set(manifest["files"]), "Required consumed metadata/logs omitted from archive inventory")
    names = {"libgenerativeqc.so"} | {n for profile in PROFILES for suffix in ["", "_spd"] for n in [f"generativeqc_stationary_{profile}{suffix}.json", f"libgenerativeqc_stationary_{profile}{suffix}.so"]}
    check(set(receipt["artifacts"]) == names and len(names) == 33, "33 artifact inventory differs")
    audit_path = out / "aot-audit.json"
    check(sha(audit_path) == receipt["aot_audit_sha256"], "AOT audit differs")
    audit = read(audit_path)["primary"]
    expected_domains = {profile + suffix for profile in PROFILES for suffix in ["", "_spd"]}
    check(audit["architecture"] == "sm_90" and len(audit["artifacts"]) == 16 and {r["name"] for r in audit["artifacts"]} == expected_domains, "16 AOT domains differ")
    for row in audit["artifacts"]:
        name = "generativeqc_stationary_" + row["name"] + ".json"
        path = out / "aot-manifests" / name
        metadata = read(path)
        check(sha(path) == receipt["artifacts"][name] and metadata["binary_sha256"] == row["binary_sha256"] == receipt["artifacts"]["libgenerativeqc_stationary_" + row["name"] + ".so"], "AOT manifest/binary identity differs")
        check(metadata["architectures"] == ["sm_90"] and metadata["compile_architectures"] == ["90"] and metadata["compile_contract"] == {"fmad":False,"fp64":True,"relaxed_constexpr":True}, "AOT target/FP64 contract differs")
        check("cubin" in row["code_kinds"] and row["driver_ptx_jit_required"] is False, "Native SM90 cubin path missing")
        intrinsic = {"schema":"generativeqc.stationary-cuda-aot.manifest-integrity.v1", "manifest":{k:v for k,v in metadata.items() if k != "manifest_integrity_sha256"}}
        check(metadata["manifest_integrity_sha256"] == hashlib.sha256(json.dumps(intrinsic,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest(), "AOT manifest integrity differs")
    result = {"status":"PASS_RECEIVER_SM90_PRODUCTION33_BUILD_METADATA_ONLY", "source_revision":REVISION, "source_tree":TREE,
              "build_receipt_sha256":sha(receipt_path), "archive_sha256":sha(archive), "artifacts":receipt["artifacts"],
              "planned_compiler_actions":len(compiles), "scope":"Exact metadata recovery, source/producer/configuration/cache-launcher/16 AOT manifest replay. Binary hashes remain bound to actual build and later loader observations; libraries are not downloaded here. No native execution, molecular E/F, sanitizer, resource/lifetime, final majority or performance acceptance."}
    target = TASK / "production-build-v1-local-replay.json"
    check(not target.exists(), "Inspect previous receiver result")
    target.write_text(json.dumps(result,indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "artifacts"}))


if __name__ == "__main__":
    main()
