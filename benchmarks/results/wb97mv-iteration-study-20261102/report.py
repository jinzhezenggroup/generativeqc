"""Audit the same-binary WB97M-V iteration experiments without ranking failures."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import lzma
from pathlib import Path

from benchmarks._exact_fp64_evidence import load, require
from benchmarks.readme_defaults_accuracy import errors
from benchmarks.readme_defaults_qualification import qualify

ROOT = Path(__file__).resolve().parents[3]


def replay(folder: Path) -> dict:
    """Check exact owner bytes, independent observables and completed allocations."""
    root = ROOT if (ROOT / "benchmarks").exists() else Path.cwd()
    module_path = root / "benchmarks/results/automatic-ks-tiles-20261102/report.py"
    spec = importlib.util.spec_from_file_location("automatic_tiles_report", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bundle, files = load(folder)
    for name, digest in bundle["references"].items():
        require(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name)
    oracles = json.loads(lzma.decompress((root / "benchmarks/results/readme-refresh-20261026/oracles.json.xz").read_bytes()))
    references = json.loads(lzma.decompress((root / "benchmarks/results/independent-force-tiles-20261031/receipts.json.xz").read_bytes()))["a100"]["analysis"]["reference_reuse"]
    results = []
    for name, text in sorted(files.items()):
        if not name.endswith("-owner.json"):
            continue
        owner = json.loads(text)
        prefix = name.rsplit("/", 1)[0]
        label = Path(name).name.removesuffix("-owner.json")
        receipt_name = prefix + "/" + label + ".json"
        job = owner["scheduler"]["LSB_JOBID"]
        controller = files["lsf-" + job + ".txt"]
        begin, end = module.lsf_lifetime(controller, job)
        require(begin <= owner["start_epoch"] <= owner["end_epoch"] < end, "owner lifetime")
        receipt_text = files.get(receipt_name)
        require(receipt_text is not None, "retained failed or complete receipt")
        require(hashlib.sha256(receipt_text.encode()).hexdigest() == owner["receipt_sha256"], "owner receipt hash")
        record = json.loads(receipt_text)
        require(owner["driver_sha256"] == record["runner_sha256"] == hashlib.sha256(files[prefix + "/wb-study-driver.py"].encode()).hexdigest(), "driver bytes")
        require(all(record["scheduler"].get(k) == v for k, v in owner["scheduler"].items()), "allocated visibility")
        successful = "Done successfully." in controller and files[prefix + "/exit-code.txt"].strip() == "0"
        row = dict(job=job, label=label, receipt=receipt_name, status=record["status"], allocation_successful=successful, timing_qualified=False, samples=[])
        if record["status"] != "complete" or owner["returncode"] != 0:
            row["failure"] = files.get(prefix + "/" + label + ".log", "")[-1800:]
            results.append(row)
            continue
        proof = json.loads(files[prefix + "/identity.json"])
        require(record["native_build"]["source_identity"] == proof["source_identity"] == bundle["source_identity"], "source identity")
        require(record["native_build"]["library_sha256"] == proof["binaries"]["libgenerativeqc.so"], "native binary")
        require(owner["start_epoch"] <= owner["receipt_mtime"] <= owner["end_epoch"], "receipt lifetime")
        gate = qualify([record], oracles, reference_reuse=references)[0]
        require(gate["status"] == "qualified", "independent energy/force gates")
        control = json.loads(files[prefix + "/default-huckel.json"])["samples"][0]
        for sample in record["samples"]:
            paired = errors(sample["energy"], sample["forces"], control, 96)
            require(paired["energy_abs_per_atom_hartree"] * 96 <= 2e-8 and paired["force_max_abs_hartree_per_bohr"] <= 2e-7, "paired target parity")
            require(sample["converged"] and sample["precision"]["mixed_stage_fock_builds"] == 0, "strict target")
            require(sample["settings"]["energy_tolerance"] == 1e-10 and sample["settings"]["density_tolerance"] == 1e-8, "unchanged final tolerances")
            diagnostic = sample["ks_diagnostic"]
            work = diagnostic["nonlocal_work"]
            require(work["provisional_focks"] + work["target_focks"] == sample["iterations"], "semantic iteration census")
            row["samples"].append(dict(phase=sample["phase"], seconds=sample["seconds"], iterations=sample["iterations"], nonlocal_work=work, initial_guess=sample["initial_guess"], paired_errors=paired, history=diagnostic["history"]))
        row["qualification"] = gate
        row["timing_qualified"] = successful
        results.append(row)
    return dict(source_identity=bundle["source_identity"], observations=results, limitations=["One cold and one warm observation per variant; no statistical speedup promotion.", "Failed allocations retain complete preceding numerical observations but do not rank timings.", "The ordinary outer-solver opt-out also disables automatic LR residency; the second experiment explicitly holds its bounded cache policy fixed.", "MINAO is not admitted for public DF owners; the rejected attempt does not change that API contract.", "Normalized history diagnostics are not the AO maximum used for stopping."])


if __name__ == "__main__":
    print(json.dumps(replay(Path(__file__).resolve().parent), indent=2))
