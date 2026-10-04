"""Reject incomplete or inaccurate cold/changed/warm points before reporting."""

import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import median

from tools.render_omol25_benchmarks import validate


def duration(value: object, name: str) -> int | float:
    """Require a measured duration before closure or cold-boundary comparisons."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"invalid {name}: expected a finite nonnegative duration")
    return value


point = Path(sys.argv[1])
reference_path = point / "reference.json"
reference = json.loads(reference_path.read_text())
identity = json.loads((point / "source-identity.json").read_text())
rows = {}
for variant in ("reference", "none", "lda16"):
    raw = json.loads((point / f"{variant}.json").read_text())
    outcome = json.loads((point / f"{variant}.outcome").read_text())
    if (
        outcome["exit_code"] != 0
        or raw["status"] != "measured"
        or raw["stage"] != "complete"
    ):
        raise ValueError(f"{variant}: incomplete endpoint")
    validate(raw, reference)
    if raw["protocol"]["reference_fock_policy"] != "full-density-rebuild":
        raise ValueError("reference Fock policy differs")
    if variant == "reference":
        if any(
            r["reference_xc_backend"]["backend"] != "cuda-libxc" for r in raw["records"]
        ):
            raise ValueError("reference semilocal XC is not entirely on GPU")
    else:
        if (
            raw["reference_sha256"]
            != hashlib.sha256(reference_path.read_bytes()).hexdigest()
        ):
            raise ValueError("different independent reference bytes")
        if (
            raw["native_build"]["probe"]["source_identity"]
            != identity["source_identity"]
        ):
            raise ValueError("native source/binary mismatch")
        seed = raw["preliminary_density"]
        if seed["requested"] != variant or seed["source_reference_density_used"]:
            raise ValueError("wrong source policy")
        if (
            seed["target_basis_identity"]["basis_identity"]
            != raw["protocol"]["basis_identity"]
        ):
            raise ValueError("wrong target basis identity")
        wrapper = duration(raw["preliminary_wrapper_seconds"], "preliminary wrapper")
        if variant == "lda16":
            if not seed["selected"] or not seed["source_converged"]:
                raise ValueError("LDA source was not admitted")
            names = (
                "source_construct_seconds",
                "source_prepare_seconds",
                "source_solve_seconds",
                "export_seconds",
                "import_seconds",
                "source_destroy_seconds",
                "source_bookkeeping_seconds",
            )
            phases = [duration(seed[name], name) for name in names]
            complete_source = duration(
                seed["complete_source_seconds"], "complete source"
            )
            if abs(sum(phases) - complete_source) > 1e-9:
                raise ValueError("source phases do not close")
            if complete_source > wrapper:
                raise ValueError("source lifetime escaped cold timer")
            cold = raw["records"][0]
            if not cold["warm_start_used"] or cold["warm_start_fallback"]:
                raise ValueError("target did not use admitted density")
        elif seed["selected"] or seed["complete_source_seconds"] is not None:
            raise ValueError("unmeasured source reported as measured")
        for row in raw["records"]:
            work = row["native_scf_ao_work"]
            if not (
                work["requested"] == work["selected"] == 1
                and work["xc_evaluations"] > 0
            ):
                raise ValueError("native target AO maps were not used")
            if not 0 < work["point_ao_square_sum"] <= work["dense_point_ao_square_sum"]:
                raise ValueError("invalid work-domain counters")
    rows[variant] = {
        "seconds": {
            phase: median(
                r["complete_seconds"] for r in raw["records"] if r["phase"] == phase
            )
            for phase in ("cold", "warm", "moved", "moved-warm")
        },
        "iterations": {
            phase: [r["iterations"] for r in raw["records"] if r["phase"] == phase]
            for phase in ("cold", "warm", "moved", "moved-warm")
        },
        "max_energy_error": max(r["energy_error"] for r in raw["records"]),
        "max_force_error": max(r["force_error"] for r in raw["records"]),
        "preliminary_density": raw.get("preliminary_density"),
        "sha256": hashlib.sha256((point / f"{variant}.json").read_bytes()).hexdigest(),
    }
print(
    json.dumps(
        {"protocol": reference["protocol"], "identity": identity, "variants": rows},
        indent=2,
    )
)
