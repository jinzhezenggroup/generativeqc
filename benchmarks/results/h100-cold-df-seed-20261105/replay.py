"""Offline replay of the source-bound H100 DF cold evidence; no chemistry runs."""

import argparse
import base64
import contextlib
import hashlib
import importlib.util
import io
import json
import lzma
import os
import subprocess
import sys
import tempfile
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check(value: object, message: str) -> None:
    if not value:
        raise ValueError(message)


def compact(result: dict) -> dict:
    keys = [
        "status",
        "scope",
        "source_revision",
        "native_source_identity",
        "core_sha256",
        "device_uuid",
        "campaign_sha256",
        "qualified_populations",
        "returned_EF",
        "changed_default_science_pass",
        "changed_default_performance_pass",
    ]
    return {
        **{k: result[k] for k in keys},
        "studies": [
            {
                k: row[k]
                for k in [
                    "id",
                    "producer_status",
                    "all_declared_pairs_qualified",
                    "summary",
                ]
            }
            for row in result["studies"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=BUNDLE.parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repository = args.repository.resolve()
    check(not sys.flags.optimize, "Use unoptimized Python for integrity checks")
    check(args.output is None or not args.output.exists(), "Preserve existing output")
    cohort = json.loads((BUNDLE / "cohort.json").read_text())
    compressed = (BUNDLE / "records.json.xz").read_bytes()
    check(
        sha(compressed) == cohort["records"]["sha256"]
        and len(compressed) == cohort["records"]["bytes"],
        "Published records differ",
    )
    check(
        sha((BUNDLE / "analyze.py").read_bytes()) == cohort["analyzer_sha256"],
        "Frozen historical reader differs",
    )
    payload = json.loads(lzma.decompress(compressed))
    check(
        payload["schema"] == "generativeqc.h100-cold-receipt-files.v1",
        "Unknown receipt container",
    )
    entries = payload["files"]
    check(
        len(entries) == cohort["records"]["files"]
        and len({e["path"] for e in entries}) == len(entries),
        "Member inventory differs",
    )
    with tempfile.TemporaryDirectory(prefix="gqc-h100-cold-readback-") as name:
        task = Path(name)
        for entry in entries:
            relative = Path(entry["path"])
            check(
                not relative.is_absolute()
                and all(
                    part not in {"", ".", ".."} for part in entry["path"].split("/")
                )
                and "\\" not in entry["path"]
                and ":" not in entry["path"],
                "Unsafe receipt path",
            )
            check(entry["encoding"] in ["utf8", "base64"], "Unknown member encoding")
            data = (
                entry["data"].encode("utf8")
                if entry["encoding"] == "utf8"
                else base64.b64decode(entry["data"], validate=True)
            )
            check(
                len(data) == entry["bytes"] and sha(data) == entry["sha256"],
                "Receipt member differs: " + entry["path"],
            )
            target = task / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(data)
        spec = importlib.util.spec_from_file_location(
            "source_bound_h100_analysis", BUNDLE / "analyze.py"
        )
        analysis = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(analysis)
        output = task / "readback.json"
        previous = sys.argv
        try:
            sys.argv = [
                str(BUNDLE / "analyze.py"),
                "--task",
                str(task),
                "--repository",
                str(repository),
                "--output",
                str(output),
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                analysis.main()
        finally:
            sys.argv = previous
        # This is a historical opt-in evidence reader, not a source-spelling CI
        # gate. Refuse a later scientific reducer instead of silently changing
        # the registered gates/semantics. A pinned source checkout can be passed.
        env = {
            **os.environ,
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_ALLOW_PROTOCOL": "",
            "GIT_TERMINAL_PROMPT": "0",
        }
        for key, module in list(sys.modules.items()):
            if key == "benchmarks" or key.startswith("benchmarks."):
                filename = getattr(module, "__file__", None)
                if filename is None:
                    continue
                path = Path(filename).resolve()
                relative = path.relative_to(repository).as_posix()
                frozen = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(repository),
                        "show",
                        cohort["measured_source"]["revision"] + ":" + relative,
                    ],
                    env=env,
                    capture_output=True,
                    check=True,
                ).stdout
                check(
                    frozen == path.read_bytes(),
                    "Imported scientific reader differs from measured source: "
                    + relative
                    + "; pass --repository with the pinned source checkout",
                )
        result = json.loads(output.read_text())
        summary = compact(result)
        for key in ["source_revision", "native_source_identity", "core_sha256"]:
            measured_key = "revision" if key == "source_revision" else key
            check(
                result[key] == cohort["measured_source"][measured_key],
                "Cohort source identity differs",
            )
        check(result["device_uuid"] == cohort["device_uuid"], "Cohort device differs")
        expected = json.loads(
            lzma.decompress((BUNDLE / "summary.json.xz").read_bytes())
        )
        check(summary == expected, "Recomputed public result differs")
        check(
            result["status"] == "PASS_CHANGED_DF_DEFAULT_SCIENCE_AND_COLD_PERFORMANCE",
            "Changed-default acceptance failed",
        )
        if args.output:
            with args.output.open("x") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write("\n")
        print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
