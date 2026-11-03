"""Recover one immutable terminal packet; refuse existing or unbound targets."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import tarfile

TASK = Path(__file__).resolve().parent


def check(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    args = parser.parse_args()
    check(sys.version_info >= (3,12), "Python3.12 filtered extraction required")
    manifest_path = TASK / (args.cohort + "-download-manifest.json")
    manifest = json.loads(manifest_path.read_text())
    check(manifest["cohort"] == args.cohort and manifest["status"] == "COMPLETE_IMMUTABLE_NOTEBOOK_PARENT_ARCHIVE", "Wrong archive identity")
    check(manifest["workload"] == "notebook" and manifest["platform_status"] == "RUNNING" and manifest["all_parent_records_terminal"] is True, "Notebook allocation/terminal parent records differ")
    archive = TASK / manifest["archive"]
    check(archive.parent == TASK and archive.stat().st_size == manifest["bytes"] and sha(archive) == manifest["sha256"], "Archive bytes differ")
    with tarfile.open(archive) as stream:
        members = stream.getmembers()
        check(len(members) == len(manifest["files"]) and {m.name for m in members} == set(manifest["files"]), "Duplicate or missing archive member")
        for member in members:
            name = Path(member.name)
            check(member.isfile() and not name.is_absolute() and ".." not in name.parts and name.parts[0] in [args.cohort,args.cohort + "-executed-producers",args.cohort + "-unverified-producers",args.cohort + "-archival-producers",args.cohort + "-archival-admissions",args.cohort + "-available-producers",args.cohort + "-available-admissions",args.cohort + "-reference-inputs"] and not (TASK / name).exists() and member.size == manifest["files"][member.name]["bytes"], "Unsafe or existing extraction target")
        stream.extractall(TASK, filter="data")
    check(all((TASK / n).stat().st_size == v["bytes"] and sha(TASK / n) == v["sha256"] for n,v in manifest["files"].items()), "Recovered bytes differ")
    check(sha(TASK / args.cohort / "campaign.json") == manifest["campaign_sha256"], "Campaign differs")
    result = {"status":"PASS_EXACT_NOTEBOOK_PARENT_BYTE_RECOVERY_ONLY", "archive_sha256":sha(archive), "manifest_sha256":sha(manifest_path), "files":len(manifest["files"])}
    target = TASK / (args.cohort + "-download.json")
    check(not target.exists(), "Inspect existing download receipt")
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
