"""Verify or rerun the frozen native prototype, never the preliminary-SCF API.

Use --verify-only for a read-only publication check. Scientific reruns require
an explicit clean source checkout, a prebuilt library and a new scratch output.
Historical sources are restored byte-for-byte from the JSON input captures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGNS = ("root", "coarse", "cluster", "hf-preliminary", "hf-lda-preliminary")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def verify_bundle(validator_source: Path) -> tuple[dict, dict, list[dict]]:
    sys.path[:0] = [str(validator_source / "python"), str(validator_source)]
    from tools.generativeqc_validation.publication import validate_publication
    from tools.generativeqc_validation.record import load_record

    publication = json.loads((HERE / "publication.json").read_text())
    files = {
        entry["path"]: (HERE / entry["path"]).read_bytes()
        for entry in publication["files"]
    }
    validate_publication(publication, files)
    captures = {}
    for campaign in CAMPAIGNS:
        capture = json.loads((HERE / "inputs" / (campaign + ".json")).read_text())
        for name, text in capture["sources"].items():
            actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if actual != capture["original_file_sha256"][name]:
                raise ValueError(
                    f"historical source checksum mismatch: {campaign}/{name}"
                )
        if (
            capture["original_file_sha256"]["staged.cpp"]
            != capture["protocol"]["harness_sha256"]
        ):
            raise ValueError(f"protocol/bridge mismatch: {campaign}")
        if (
            capture["original_file_sha256"]["basis.inc"]
            != capture["protocol"]["basis_sha256"]
        ):
            raise ValueError(f"protocol/basis mismatch: {campaign}")
        captures[campaign] = capture
    samples = load_record(HERE / "evidence.json")["timings"]
    if len(samples) != 224:
        raise ValueError("incomplete historical sample set")
    return json.loads((HERE / "provenance.json").read_text()), captures, samples


def process_schedule(samples: list[dict], campaign: str) -> list[tuple[int, str, str]]:
    # Collapse only the second endpoint in the same native process. Preserve the
    # original process order rather than synthesizing a new A/B experiment.
    schedule = []
    for sample in samples:
        row = sample["diagnostics"]
        if row["campaign"] != campaign:
            continue
        key = (row["repeat"], row["case"], row["mode"])
        if key not in schedule:
            schedule.append(key)
    return schedule


def compile_command(args: argparse.Namespace, source: Path, output: Path) -> list[str]:
    return [
        args.ccache,
        args.cxx,
        "-std=c++20",
        "-O2",
        "-I" + str(args.source_root / "include"),
        "-I" + str(args.source_root / "src"),
        str(source),
        str(args.library),
        "-Wl,-rpath," + str(args.library.parent),
        "-o",
        str(output),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check inputs and print commands, without writes/builds/solves",
    )
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--campaign", choices=CAMPAIGNS, action="append")
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="Also repeat the water spot-check; requires an existing PySCF environment",
    )
    parser.add_argument("--ccache", default="ccache")
    parser.add_argument("--cxx", default="c++")
    parser.add_argument(
        "--timeout",
        type=float,
        help="Override historical per-process limit (90 s root/coarse; 75 s others)",
    )
    args = parser.parse_args()
    validator_source = (args.source_root or HERE.parents[2]).resolve()
    provenance, captures, samples = verify_bundle(validator_source)
    if args.verify_only:
        print(
            "Standard publication/envelope validation passed; exact source captures and 224 samples verified."
        )
        return 0
    if not all((args.source_root, args.library, args.output)):
        parser.error("rerun/dry-run requires --source-root, --library and --output")
    args.source_root = args.source_root.resolve(strict=True)
    args.library = args.library.resolve(strict=True)
    from benchmarks._retention import raw_output_path

    args.output = raw_output_path(
        args.output, repository_root=args.source_root
    ).resolve()
    raw_output_path(args.output, repository_root=HERE.parents[2])
    if args.output.is_relative_to(HERE) or args.output.exists():
        parser.error("output must be a new scratch directory outside this publication")
    if args.timeout is not None and args.timeout <= 0:
        parser.error("--timeout must be positive")
    revision = git(args.source_root, "rev-parse", "HEAD")
    tree = git(args.source_root, "rev-parse", "HEAD^{tree}")
    dirty = bool(git(args.source_root, "status", "--porcelain", "--untracked-files=no"))
    if tree != provenance["measured_tree"] or dirty:
        parser.error(
            "use a clean checkout of the pinned public-equivalent source tree; new API code is outside this experiment"
        )
    campaigns = args.campaign or list(CAMPAIGNS)
    if len(set(campaigns)) != len(campaigns):
        parser.error("do not repeat --campaign values")
    if args.oracle and "root" not in campaigns:
        parser.error("--oracle requires the root campaign")
    commands = {
        campaign: compile_command(
            args,
            args.output / campaign / "staged.cpp",
            args.output / campaign / "staged",
        )
        for campaign in campaigns
    }
    if args.oracle:
        commands["oracle-grid"] = compile_command(
            args,
            args.output / "root" / "oracle_grid.cpp",
            args.output / "root" / "oracle-grid",
        )
    metadata = {
        "source_revision": revision,
        "source_tree": tree,
        "source_dirty": dirty,
        "library_sha256": digest(args.library),
        "historical_library_sha256": provenance["library_sha256"],
        "platform": platform.platform(),
        "python": sys.version,
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "campaigns": campaigns,
        "commands": commands,
        "process_order": {c: process_schedule(samples, c) for c in campaigns},
        "scope": "New reproduction of frozen prototype, not new API qualification or replay of historical timings",
        "compiler_flags_scope": "Portable C++20/-O2 bridge build, not an assertion about unrecorded historical bridge flags",
        "timeout_override_seconds": args.timeout,
    }
    if args.dry_run:
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return 0
    # A missing cache is a prerequisite failure, never an uncached fallback.
    metadata["ccache_version"] = subprocess.check_output(
        [args.ccache, "--version"], text=True
    )
    metadata["compiler_version"] = subprocess.check_output(
        [args.cxx, "--version"], text=True
    )
    env = os.environ | {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
        "CCACHE_BASEDIR": str(args.source_root),
    }
    args.output.mkdir(parents=True)
    write_json(args.output / "reproduction.json", metadata)
    failures = 0
    for campaign in campaigns:
        target = args.output / campaign
        target.mkdir()
        for name, text in captures[campaign]["sources"].items():
            (target / name).write_bytes(text.encode("utf-8"))
        write_json(target / "historical-protocol.json", captures[campaign]["protocol"])
        subprocess.run(commands[campaign], check=True, env=env)
        timeout = args.timeout or (90 if campaign in {"root", "coarse"} else 75)
        with (target / "records.jsonl").open("w") as stream:
            for repeat, case, mode in metadata["process_order"][campaign]:
                started = time.perf_counter()
                try:
                    process = subprocess.run(
                        [str(target / "staged"), case, mode],
                        capture_output=True,
                        text=True,
                        env=env,
                        timeout=timeout,
                        check=False,
                    )
                    wall = time.perf_counter() - started
                    if process.returncode:
                        emitted = [
                            {
                                "failure": f"exit {process.returncode}",
                                "stdout": process.stdout,
                                "stderr": process.stderr,
                            }
                        ]
                        failures += 1
                    else:
                        emitted = [
                            json.loads(line) for line in process.stdout.splitlines()
                        ]
                        if not emitted:
                            raise ValueError(
                                "successful process produced no endpoint records"
                            )
                except subprocess.TimeoutExpired as error:
                    wall = time.perf_counter() - started

                    def decode(text: str | bytes | None) -> str | None:
                        return (
                            text.decode(errors="replace")
                            if isinstance(text, bytes)
                            else text
                        )

                    emitted = [
                        {
                            "failure": "timeout",
                            "timeout_s": timeout,
                            "stdout": decode(error.stdout),
                            "stderr": decode(error.stderr),
                        }
                    ]
                    failures += 1
                for row in emitted:
                    row.update(case=case, mode=mode, repeat=repeat, process_wall_s=wall)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
    if args.oracle:
        if failures:
            raise RuntimeError(
                "campaign failures retained; inspect before running the independent oracle"
            )
        target = args.output / "root"
        subprocess.run(commands["oracle-grid"], check=True, env=env)
        with (target / "oracle-grid.txt").open("w") as stream:
            subprocess.run(
                [str(target / "oracle-grid")], stdout=stream, check=True, env=env
            )
        subprocess.run(
            [sys.executable, str(target / "check_oracle.py")], check=True, env=env
        )
    print(f"Retained new raw records in {args.output}; failing processes: {failures}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
