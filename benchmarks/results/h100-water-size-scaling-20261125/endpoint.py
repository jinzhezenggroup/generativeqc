"""Cold-only calls through the pinned paper drivers on the new geometries."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
from pathlib import Path
from time import monotonic

from benchmarks import readme_defaults as paper
from benchmarks._support import cuda_accelerator_metadata


def mark(stage):
    print("STAGE " + json.dumps({"stage": stage, "monotonic": monotonic()}), flush=True)


def staged(method, name):
    def call(*args, **kwargs):
        mark(name)
        return method(*args, **kwargs)

    return call


class NativeOwner:
    def __init__(self, owner):
        self.owner = owner

    def __getattr__(self, name):
        return getattr(self.owner, name)

    def execute(self, *args, **kwargs):
        return staged(self.owner.execute, "native_scf_and_forces")(*args, **kwargs)


class NativeCalculator:
    def __init__(self, calculator):
        self.calculator = calculator

    def __getattr__(self, name):
        return getattr(self.calculator, name)

    def prepare_batch(self, *args, **kwargs):
        return NativeOwner(
            staged(self.calculator.prepare_batch, "native_prepare")(*args, **kwargs)
        )


def observe_native():
    original = paper.native_calculator
    paper.native_calculator = lambda *a, **kw: NativeCalculator(original(*a, **kw))


def observe_reference(engine_type, mode):
    def gradient_with_marker(method):
        def gradient(*args, **kwargs):
            mark("reference_gradient_setup")
            owner = method(*args, **kwargs)
            owner.kernel = staged(owner.kernel, "reference_forces")
            return owner

        return gradient

    if mode == "direct":
        engine_type.kernel = staged(engine_type.kernel, "reference_scf")
        engine_type.nuc_grad_method = gradient_with_marker(engine_type.nuc_grad_method)
        return

    original = engine_type.density_fit

    def fit(engine, *args, **kwargs):
        mark("reference_df_setup")
        fitted = original(engine, *args, **kwargs)
        fitted.kernel = staged(fitted.kernel, "reference_scf")
        fitted.nuc_grad_method = gradient_with_marker(fitted.nuc_grad_method)
        return fitted

    engine_type.density_fit = fit


def cpu_oracle(args, geometry, record, source):
    spec = importlib.util.find_spec("benchmarks.readme_defaults_accuracy")
    if spec is None or spec.origin is None:
        raise ValueError("cannot locate the pinned CPU oracle driver")
    digest = paper.digest(Path(spec.origin))
    record["cpu_oracle_driver_sha256"] = digest
    if digest != source["cpu_oracle_driver_sha256"]:
        raise ValueError("CPU oracle driver differs from the frozen source manifest")
    from pyscf import lib

    from benchmarks.readme_defaults_accuracy import solve

    model = next(
        line.split(":", 1)[1].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.startswith("model name")
    )
    record["cpu"] = {
        "model": model,
        "threads": lib.num_threads(),
        "affinity_count": len(os.sched_getaffinity(0)),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
    }
    if "Xeon" not in model or "9462" not in model:
        raise RuntimeError("expected Xeon CPU Max 9462 for this oracle allocation")
    for level in (5,) if args.method == "hf" else (4, 5):
        mark(f"cpu_reference_level_{level}")
        record["samples"].append(
            solve(args.method, geometry, backend="cpu", level=level)
        )
        paper.save(args.output, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("standard", "gpu4pyscf", "cpu"))
    parser.add_argument("method", choices=("hf", "pbe", "r2scan", "wb97m-v"))
    parser.add_argument("mode", choices=("direct", "df"))
    parser.add_argument("atoms", type=int)
    parser.add_argument("output", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.arm == "cpu" and args.mode != "direct":
        parser.error("the independent CPU oracle uses Direct SCF")
    args.inputs = args.root / "inputs"
    args.force_tile_points = None
    args.warm_repeats = 0
    source = json.loads((args.root / "source.json").read_text())
    geometry_bytes = (args.root / "geometries.json").read_bytes()
    geometry_digest = hashlib.sha256(geometry_bytes).hexdigest()
    if geometry_digest != source["geometry_sha256"]:
        raise ValueError("geometry differs from the frozen source manifest")
    geometry = json.loads(geometry_bytes)["cases"][str(args.atoms)]["geometry_bohr"]
    if args.output.exists():
        raise FileExistsError(args.output)
    record = {
        "status": "running",
        "arm": args.arm,
        "method": args.method,
        "mode": args.mode,
        "atoms": args.atoms,
        "source_revision": source["revision"],
        "paper_driver_sha256": paper.digest(Path(paper.__file__)),
        "geometry_sha256": geometry_digest,
        "orbital_basis": "def2-SVP",
        "basis_representation": "spherical",
        "charge": 0,
        "spin": 0,
        "packages": {"pyscf": importlib.metadata.version("pyscf")},
        "python": platform.python_version(),
        "samples": [],
    }
    paper.save(args.output, record)

    def retain(sample):
        mark("complete_endpoint")
        record["samples"].append(sample)
        paper.save(args.output, record)

    try:
        if args.arm == "cpu":
            cpu_oracle(args, geometry, record, source)
        else:
            if args.mode == "df":
                relative = f"inputs/{args.method}-auxiliary.json"
                digest = paper.digest(args.root / relative)
                record["auxiliary_input_sha256"] = digest
                if digest != source["inputs_sha256"][relative]:
                    raise ValueError("DF auxiliary basis differs from the frozen input")
            import cupy as cp

            record["allocation"] = paper.allocation_metadata(
                os.environ["GQC_QZ_JOB_NAME"]
            )
            record["accelerator"] = cuda_accelerator_metadata(cp)
            record["packages"]["cupy"] = cp.__version__
            if args.arm == "standard":
                observe_native()
                mark("native_setup")
                paper.run_native(args, geometry, record, retain)
            else:
                import gpu4pyscf
                from gpu4pyscf.dft import rks
                from gpu4pyscf.scf import hf

                version = importlib.metadata.version("gpu4pyscf-cuda12x")
                record["packages"]["gpu4pyscf"] = version
                if version != "1.9.0" or cp.__version__ != "13.4.1":
                    raise ValueError(
                        "reference must resolve GPU4PySCF 1.9.0 and CuPy 13.4.1"
                    )
                record["gpu4pyscf_path"] = gpu4pyscf.__file__
                observe_reference(hf.RHF if args.method == "hf" else rks.RKS, args.mode)
                mark("reference_setup")
                paper.run_reference(args, geometry, record, retain)
        record["status"] = "complete"
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        paper.save(args.output, record)


if __name__ == "__main__":
    main()
