"""Convert retained Nsight capture streams and produce the CUDA kernel census."""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from acquire import digest, save
from nsys_probe import CAPTURE_SHA256, kernel_census, run

IMPORTER_SHA256 = "8b4cc3e19d8d33b07fc8d4d09fc49d6a0b5a80543f993df38e4d28eaf8718361"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--importer", type=Path, required=True)
    parser.add_argument("--nsys", type=Path, required=True)
    args = parser.parse_args()
    capture = args.capture.resolve()
    importer = args.importer.resolve()
    nsys = args.nsys.resolve()
    if digest(importer) != IMPORTER_SHA256 or digest(nsys) != CAPTURE_SHA256:
        raise ValueError("unexpected Nsight capture conversion tools")
    output = args.output.resolve()
    output.mkdir()
    report = {
        "capture_sha256": digest(capture / "nsys.json"),
        "importer_sha256": IMPORTER_SHA256,
        "exporter_sha256": CAPTURE_SHA256,
        "cases": [],
    }
    recorded = json.loads((capture / "nsys.json").read_text())
    destination = output / "conversion.json"
    for record in recorded["cases"]:
        name = record["case"]
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
        with tempfile.TemporaryDirectory(dir=path) as directory:
            working = Path(directory) / "working.qdstrm"
            shutil.copyfile(source, working)
            row["import"] = run(
                [importer, "--input-file", working, "--output-file", rep],
                path / "import.txt",
                path,
                os.environ,
                180,
            )
        if digest(source) != row["qdstrm_sha256"]:
            raise ValueError(f"original capture changed: {name}")
        save(destination, report)
        if row["import"]["returncode"] != 0:
            continue
        row["report_sha256"] = digest(rep)
        sqlite = path / "capture.sqlite"
        row["export"] = run(
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
        )
        save(destination, report)
        if row["export"]["returncode"] == 0:
            row["sqlite_sha256"] = digest(sqlite)
            save(path / "kernels.json", kernel_census(sqlite))
            save(destination, report)


if __name__ == "__main__":
    main()
