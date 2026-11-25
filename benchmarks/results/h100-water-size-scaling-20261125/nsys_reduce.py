"""Convert retained Nsight capture streams and produce the CUDA kernel census."""

import argparse
import hashlib
import json
import os
import shutil
import signal
import sqlite3
import tempfile
from itertools import product
from pathlib import Path

from acquire import METHODS, MODES, SIZES, digest, interrupt, save
from nsys_probe import CAPTURE_SHA256, kernel_census, run

IMPORTER_SHA256 = "8b4cc3e19d8d33b07fc8d4d09fc49d6a0b5a80543f993df38e4d28eaf8718361"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--importer", type=Path, required=True)
    parser.add_argument("--nsys", type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    capture = args.capture.resolve()
    importer = args.importer.resolve()
    nsys = args.nsys.resolve()
    if digest(importer) != IMPORTER_SHA256 or digest(nsys) != CAPTURE_SHA256:
        raise ValueError("unexpected Nsight capture conversion tools")
    manifest = (capture / "nsys.json").read_bytes()
    names = [record["case"] for record in json.loads(manifest)["cases"]]
    allowed = {
        f"{atoms}-{method}-{mode}"
        for atoms, method, mode in product(SIZES, METHODS, MODES)
    }
    if any(not isinstance(name, str) or name not in allowed for name in names):
        raise ValueError("capture includes a case outside the measured protocol")
    if len(set(names)) != len(names):
        raise ValueError("capture cases must be unique")
    output = args.output.resolve()
    output.mkdir()
    report = {
        "capture_sha256": hashlib.sha256(manifest).hexdigest(),
        "importer_sha256": IMPORTER_SHA256,
        "exporter_sha256": CAPTURE_SHA256,
        "cases": [],
    }
    destination = output / "conversion.json"
    save(destination, report)
    for name in names:
        source = capture / name / "capture.qdstrm"
        path = output / name
        path.mkdir()
        row = {
            "case": name,
            "qdstrm_sha256": digest(source),
            "qdstrm_bytes": source.stat().st_size,
        }
        report["cases"].append(row)
        rep = path / "capture.nsys-rep"
        row["import"] = {"status": "starting"}
        save(destination, report)
        try:
            with tempfile.TemporaryDirectory(dir=path) as directory:
                working = Path(directory) / "working.qdstrm"
                shutil.copyfile(source, working)
                run(
                    [importer, "--input-file", working, "--output-file", rep],
                    path / "import.txt",
                    path,
                    os.environ,
                    180,
                    row["import"],
                )
        finally:
            save(destination, report)
        if digest(source) != row["qdstrm_sha256"]:
            raise ValueError(f"original capture changed: {name}")
        if row["import"]["status"] != "complete":
            continue
        if not rep.is_file():
            row["conversion_error"] = "missing Nsight report"
            save(destination, report)
            continue
        row["report_sha256"] = digest(rep)
        sqlite = path / "capture.sqlite"
        row["export"] = {"status": "starting"}
        save(destination, report)
        try:
            run(
                [
                    nsys,
                    "export",
                    "--type=sqlite",
                    "--lazy=false",
                    f"--output={sqlite}",
                    rep,
                ],
                path / "export.txt",
                path,
                os.environ,
                180,
                row["export"],
            )
        finally:
            save(destination, report)
        if row["export"]["status"] != "complete":
            continue
        if not sqlite.is_file():
            row["census_error"] = "missing SQLite output"
            save(destination, report)
            continue
        row["sqlite_sha256"] = digest(sqlite)
        save(destination, report)
        try:
            save(path / "kernels.json", kernel_census(sqlite))
        except sqlite3.Error as error:
            row["census_error"] = f"{type(error).__name__}: {error}"
            save(destination, report)


if __name__ == "__main__":
    main()
