import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.dft_mp_v1 import qualify_capacity

ROOT = Path(__file__).resolve().parents[2]
SOURCE_SHA = "f" * 40
SPD_CONTRACT_FILES = (
    "src/molecule/basis.cpp",
    "src/dft/ao_grid.cpp",
    "src/dft/bridge.cpp",
    "python/vibeqc/_stationary_cuda.py",
)


def report(*, aot_directory: Path | None = None) -> dict:
    return qualify_capacity._build_report(
        ROOT,
        source_sha=SOURCE_SHA,
        aot_directory=aot_directory,
    )


def spd_contract_tree(tmp_path: Path) -> None:
    for relative in SPD_CONTRACT_FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            (ROOT / relative).read_text(encoding="utf-8"), encoding="utf-8"
        )


def test_frozen_capacity_report_uses_actual_basis_and_grid_identities() -> None:
    result = report()

    assert result["schema"] == "vibeqc.dft-mp-v1.stationary-capacity.v1"
    assert result["source"]["sha"] == SOURCE_SHA
    assert result["contract"] == {
        "id": "DFT-MP-v1",
        "version": "1.0.0",
        "sha256": "a9d439261a08007a296862e8869d35c1b127e6725e7bd73e294fa9ccc9b602a0",
    }
    assert result["basis"]["manifest_ao_counts_match"] is True
    assert result["basis"]["basis_pack_sha256_match"] is True
    assert result["basis"]["packed_capacity_definition"] == (
        "np.empty(3 * self.natom + 2 * self.nprimitive + 16 * self.nao)"
    )
    assert result["basis"]["numeric_capacity_definition"] == (
        "2 * self.packed.nbytes + 32 * self.natom + "
        "32 * len(self.shells) + 16 * self.nprimitive"
    )
    assert result["basis"]["spd_expansion_contract_sha256"] == (
        "f0d9be746f30067f6dba8293bcc35a9dc06db74037a6c76d322d21051bc61334"
    )
    assert result["basis"]["sparse_spherical_component_terms"] == {
        "s": 1,
        "p": 3,
        "d": 8,
    }
    assert result["basis"]["ao_packer_contract_sha256"] == (
        "07858ba7f9a78fe6348bbcb9430eb4f8321db8774ea3ce1ecef495629abe2a1c"
    )
    assert result["basis"]["ao_pack_bridge_contract_sha256"] == (
        "c5c8a0181075e7d171e1d189c875d5cc9e69467cb069b13267f91e73b1e1dd7e"
    )
    assert result["basis"]["native_ao_constructor_contract_sha256"] == (
        "1cf236f40a51bdad066635e3eaaaab3d8fc4c7c094653b73898b8dc32bd89e08"
    )
    assert result["basis"]["stationary_layout_contract_sha256"] == (
        "89568c04b3b5f91bec27a391ca5e819f279f3f0385e1712f33f522f7265fdb62"
    )

    cases = {item["id"]: item for item in result["cases"]}
    assert set(cases) == {
        "ace_glygly_nme",
        "benzene",
        "caffeine",
        "o2",
        "oh",
        "water",
        "water_dimer",
        "water8",
        "water16",
        "water32",
    }
    assert cases["water8"]["shape"] == {
        "atom_count": 24,
        "ao_count_spherical": 192,
        "shell_count": 96,
        "basis_primitive_count": 176,
        "component_primitive_sum": 328,
        "grid_points": 1_990_656,
    }
    assert cases["water16"]["shape"]["ao_count_spherical"] == 384
    assert cases["water32"]["shape"]["ao_count_spherical"] == 768
    assert cases["caffeine"]["shape"]["ao_count_spherical"] == 246
    assert cases["ace_glygly_nme"]["shape"]["ao_count_spherical"] == 247
    assert cases["water32"]["identities"]["grid"] == (
        "462361d06e44c372a4b116599ecde2e4bf6b6bd4416f26fea2f3b47fa6cb0ece"
    )
    assert cases["water8"]["identities"]["changed_input_sha256"] == (
        "5c2813a5e647040a55cc6e4fc7f4879759c896c906e53bed49472e73dd7e5d69"
    )
    assert cases["water8"]["identities"]["changed_grid"] == (
        "f0f51a6ba5dd355d91f95b0bdcd5d0354b889b3a257a7970c39aaf7c3ce36c0d"
    )
    assert all(item["identities"]["basis"] for item in cases.values())


def test_report_exposes_exact_first_gate_and_all_losing_work() -> None:
    result = report()
    cases = {item["id"]: item for item in result["cases"]}

    assert result["admission_limits"]["small_domain"] == {
        "atom_count": 32,
        "ao_count": 128,
    }
    assert result["admission_limits"]["basis_primitive_count"] == 4096
    assert result["admission_limits"]["primitive_records"] == 16_000_000
    assert result["admission_limits"]["primitive_records_scope"] == (
        "whole_force_cumulative"
    )
    assert result["admission_limits"]["primitive_records_definition"] == (
        "(1 + int(has_exchange)) * primitive_sum ** 4 + "
        "(na + 2) * primitive_sum ** 2 + na * (na - 1) // 2"
    )
    assert result["admission_limits"]["grid_pair_visits_definition"] == (
        "(1 + 2 * len(state.grid.points)) * na * (na - 1) // 2"
    )
    assert result["admission_limits"]["source_bytes_definition"].startswith(
        "8 * (22 * primitive_tile"
    )
    assert result["admission_limits"]["host_bound_definition"].startswith(
        "grid_plan.host_bytes + 8 * (34 * primitive_tile"
    )
    assert result["admission_limits"]["available_device_bytes_definition"] == (
        "max_device_bytes - grid_plan.peak_bytes - source_bytes"
    )
    assert result["admission_limits"]["additional_device_admission"] == (
        "additional_device_peak_bound < additional_device_budget"
    )
    assert result["admission_limits"]["gate_predicates"] == {
        "primitive_records": "records > max_primitive_records",
        "grid_points": "len(state.grid.points) > max_grid_points",
        "grid_pair_visits": "pair_visits > max_grid_pair_visits",
        "additional_device": "available <= 0",
        "additional_host": "host_bound > max_host_bytes",
    }
    assert result["admission_limits"]["tile_points"] == 256
    assert result["admission_limits"]["primitive_tile"] == 4096
    assert result["admission_limits"]["integral_terms"] == 32
    assert result["admission_limits"]["grid_points"] == 1_000_000
    assert result["admission_limits"]["grid_pair_visits"] == 100_000_000

    assert cases["water8"]["requirements"]["primitive_records"] == 11_577_114_516
    assert cases["water16"]["requirements"]["primitive_records"] == 185_210_590_824
    assert cases["water32"]["requirements"]["primitive_records"] == 2_963_193_862_608
    assert cases["caffeine"]["requirements"]["grid_pair_visits"] == 1_098_842_388
    assert cases["ace_glygly_nme"]["requirements"]["grid_pair_visits"] == 1_401_753_925

    assert cases["water8"]["admission"]["first_blocker"]["gate"] == (
        "small_domain_atom_ao_cap"
    )
    assert cases["water8"]["admission"]["first_blocker"]["exceeded"] == ["ao_count"]
    assert cases["water16"]["admission"]["first_blocker"]["exceeded"] == [
        "atom_count",
        "ao_count",
    ]
    assert cases["water32"]["admission"]["first_blocker"]["exceeded"] == [
        "atom_count",
        "ao_count",
    ]
    assert cases["caffeine"]["admission"]["first_blocker"]["exceeded"] == ["ao_count"]
    assert cases["ace_glygly_nme"]["admission"]["first_blocker"]["exceeded"] == [
        "ao_count"
    ]
    assert cases["benzene"]["admission"]["first_blocker"]["gate"] == (
        "primitive_work_budget"
    )
    assert cases["water_dimer"]["admission"]["first_blocker"]["gate"] == (
        "primitive_work_budget"
    )
    for sentinel in ("water", "oh", "o2"):
        assert cases[sentinel]["admission"]["outcome"] == (
            "passes_static_stationary_caps"
        )
        assert cases[sentinel]["admission"]["first_blocker"] is None

    # Losing gates remain visible after the first failure; the report is not a
    # pass/fail truncation that hides the production-scale work.
    water32_gates = [item["gate"] for item in cases["water32"]["admission"]["failures"]]
    assert water32_gates == [
        "small_domain_atom_ao_cap",
        "primitive_work_budget",
        "grid_point_work_budget",
        "grid_pair_work_budget",
    ]


def test_report_covers_every_required_semilocal_fp64_force_row_and_aot_route(
    tmp_path: Path,
) -> None:
    result = report()
    rows = result["rows"]

    assert len(rows) == 22
    assert {row["method"] for row in rows} == {"lda", "pbe", "r2scan"}
    assert {row["product"] for row in rows} == {"energy+analytic_forces"}
    assert all(row["required"] is True for row in rows)
    assert all(row["public_capability"]["forces"] is True for row in rows)
    assert all(
        row["public_route"]["scientific_runtime_compilation_required"] is False
        for row in rows
    )
    assert all(row["packaged_aot"]["source_package_declared"] is True for row in rows)
    assert all(
        row["packaged_aot"]["binary_verification"]["status"] == "not_checked"
        for row in rows
    )

    pbe_uks = next(row for row in rows if row["id"] == "pbe/uks/oh/fp64_energy_forces")
    assert pbe_uks["packaged_aot"]["name"] == "pbe_uks_spd"
    assert pbe_uks["packaged_aot"]["contract_identity"]
    assert pbe_uks["stationary_plan"]["source_names"] == [
        "one_electron",
        "coulomb",
        "xc_ao",
        "xc_grid",
        "xc_weight",
        "overlap_pulay",
        "nuclear",
    ]
    water32 = next(
        row for row in rows if row["id"] == "pbe/rks/water32/fp64_energy_forces"
    )
    assert water32["resource_requirements"]["additional_device_peak_bound"] == (
        353_705_216
    )
    assert water32["resource_requirements"]["additional_host_numeric_bound"] == (
        192_105_088
    )
    assert water32["resource_requirements"]["additional_device_budget"] == 512 << 20
    assert water32["resource_requirements"]["additional_host_budget"] == 256 << 20

    missing = report(aot_directory=tmp_path)
    missing_rows = missing["rows"]
    assert all(
        row["packaged_aot"]["binary_verification"]["status"] == "missing_or_invalid"
        for row in missing_rows
    )
    assert all(
        "missing packaged stationary CUDA artifact"
        in row["packaged_aot"]["binary_verification"]["detail"]
        for row in missing_rows
    )


def test_machine_readable_report_round_trips_without_nonfinite_values() -> None:
    result = report()
    encoded = json.dumps(result, allow_nan=False, sort_keys=True)
    assert json.loads(encoded) == result


def test_report_rejects_a_repository_other_than_its_import_checkout(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="checkout containing this tool"):
        qualify_capacity.build_report(tmp_path)


def test_public_report_binds_the_clean_git_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "e" * 40
    monkeypatch.setattr(qualify_capacity, "_clean_git_sha", lambda _: revision)

    result = qualify_capacity.build_report(ROOT)

    assert result["source"]["sha"] == revision


def test_primitive_budget_scope_fails_closed_when_whole_force_gate_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/vibeqc/_stationary_cuda.py").read_text(encoding="utf-8")
    whole_force_gate = (
        "    if records > max_primitive_records:\n"
        '        raise ValueError("primitive work budget exceeded")\n'
    )
    assert whole_force_gate in source
    target = tmp_path / "python/vibeqc/_stationary_cuda.py"
    target.parent.mkdir(parents=True)
    target.write_text(source.replace(whole_force_gate, ""), encoding="utf-8")

    with pytest.raises(RuntimeError, match="whole-force cumulative"):
        qualify_capacity._source_limits(tmp_path)


def test_primitive_budget_scope_fails_closed_when_work_definition_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/vibeqc/_stationary_cuda.py").read_text(encoding="utf-8")
    old = "(1 + int(has_exchange)) * primitive_sum**4"
    assert old in source
    target = tmp_path / "python/vibeqc/_stationary_cuda.py"
    target.parent.mkdir(parents=True)
    target.write_text(source.replace(old, "primitive_sum**3", 1), encoding="utf-8")

    with pytest.raises(RuntimeError, match="primitive-record definition"):
        qualify_capacity._source_limits(tmp_path)


def test_memory_bounds_fail_closed_when_production_definition_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/vibeqc/_stationary_cuda.py").read_text(encoding="utf-8")
    old = "22 * primitive_tile"
    assert old in source
    target = tmp_path / "python/vibeqc/_stationary_cuda.py"
    target.parent.mkdir(parents=True)
    target.write_text(source.replace(old, "23 * primitive_tile", 1), encoding="utf-8")

    with pytest.raises(RuntimeError, match="source-bytes definition"):
        qualify_capacity._source_limits(tmp_path)


def test_memory_bounds_fail_closed_when_host_gate_moves(tmp_path: Path) -> None:
    source = (ROOT / "python/vibeqc/_stationary_cuda.py").read_text(encoding="utf-8")
    old = "if host_bound > max_host_bytes:"
    assert old in source
    target = tmp_path / "python/vibeqc/_stationary_cuda.py"
    target.parent.mkdir(parents=True)
    target.write_text(source.replace(old, "if host_bound >= max_host_bytes:", 1))

    with pytest.raises(RuntimeError, match="additional-host predicate"):
        qualify_capacity._source_limits(tmp_path)


def test_basis_numeric_bound_fails_closed_when_production_definition_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/vibeqc_compiler/dft/ao.py").read_text(encoding="utf-8")
    old = "2 * self.packed.nbytes"
    assert old in source
    target = tmp_path / "python/vibeqc_compiler/dft/ao.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        source.replace(old, "3 * self.packed.nbytes", 1), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="numeric capacity definition"):
        qualify_capacity._basis_layout_contract(tmp_path)


def test_spherical_component_count_fails_closed_when_d_expansion_moves(
    tmp_path: Path,
) -> None:
    spd_contract_tree(tmp_path)
    target = tmp_path / "src/molecule/basis.cpp"
    source = target.read_text(encoding="utf-8")
    old = ", {{0, 2, 0}, -root_three_over_two}"
    assert old in source
    target.write_text(source.replace(old, "", 1), encoding="utf-8")

    with pytest.raises(RuntimeError, match="s/p/d expansion contract changed"):
        qualify_capacity._spd_expansion_contract(tmp_path)


def test_spherical_component_count_fails_closed_when_cartesian_generator_moves(
    tmp_path: Path,
) -> None:
    spd_contract_tree(tmp_path)
    target = tmp_path / "src/molecule/basis.cpp"
    source = target.read_text(encoding="utf-8")
    old = "components.push_back({static_cast<unsigned>(lx), ly, lz});"
    assert old in source
    target.write_text(source.replace(old, "components.push_back({0, ly, lz});", 1))

    with pytest.raises(RuntimeError, match="s/p/d expansion contract changed"):
        qualify_capacity._spd_expansion_contract(tmp_path)


def test_spherical_component_count_fails_closed_when_ao_packer_moves(
    tmp_path: Path,
) -> None:
    spd_contract_tree(tmp_path)
    target = tmp_path / "src/dft/ao_grid.cpp"
    source = target.read_text(encoding="utf-8")
    old = "record[3] = expansion.size();"
    assert old in source
    target.write_text(source.replace(old, "record[3] = 1;", 1))

    with pytest.raises(RuntimeError, match="packed-AO contract changed"):
        qualify_capacity._spd_expansion_contract(tmp_path)


def test_spherical_component_count_fails_closed_when_layout_parser_moves(
    tmp_path: Path,
) -> None:
    spd_contract_tree(tmp_path)
    target = tmp_path / "python/vibeqc/_stationary_cuda.py"
    source = target.read_text(encoding="utf-8")
    old = "for term in range(int(row[3]))"
    assert old in source
    target.write_text(source.replace(old, "for term in range(1)", 1))

    with pytest.raises(RuntimeError, match="stationary layout contract changed"):
        qualify_capacity._spd_expansion_contract(tmp_path)


def test_spherical_component_count_fails_closed_when_basis_creation_moves(
    tmp_path: Path,
) -> None:
    spd_contract_tree(tmp_path)
    target = tmp_path / "src/dft/bridge.cpp"
    source = target.read_text(encoding="utf-8")
    old = "dimensions[2] = basis->nao;"
    assert old in source
    target.write_text(source.replace(old, "dimensions[2] = 0;", 1))

    with pytest.raises(RuntimeError, match="packed-AO contract changed"):
        qualify_capacity._spd_expansion_contract(tmp_path)


def test_module_import_binds_helpers_to_the_tool_checkout(tmp_path: Path) -> None:
    environment = {
        key: value for key, value in os.environ.items() if key != "PYTHONPATH"
    }
    environment["PYTHONPATH"] = str(ROOT)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import inspect; "
                "from tools.dft_mp_v1 import qualify_capacity as q; "
                "print(inspect.getfile(q.Atom))"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()).resolve().is_relative_to(ROOT / "python")


def test_report_rejects_a_helper_imported_outside_the_tool_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qualify_capacity, "Atom", Path)
    with pytest.raises(RuntimeError, match="outside the tool checkout"):
        qualify_capacity.build_report(ROOT)


def test_malformed_optional_aot_manifest_is_reported_not_raised(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def malformed_loader(*_: object, **__: object) -> object:
        raise KeyError("source_identity")

    monkeypatch.setattr(
        qualify_capacity, "load_stationary_aot_artifact", malformed_loader
    )
    result = qualify_capacity._artifact_verification(
        tmp_path,
        functional=0,
        spin="unpolarized",
        plan=object(),
    )

    assert result == {
        "status": "missing_or_invalid",
        "detail": "missing AOT manifest field: source_identity",
    }


def test_non_object_optional_aot_manifest_is_reported_not_raised(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def non_object_loader(*_: object, **__: object) -> object:
        raise AttributeError("'list' object has no attribute 'get'")

    monkeypatch.setattr(
        qualify_capacity, "load_stationary_aot_artifact", non_object_loader
    )
    result = qualify_capacity._artifact_verification(
        tmp_path,
        functional=0,
        spin="unpolarized",
        plan=object(),
    )

    assert result == {
        "status": "missing_or_invalid",
        "detail": "invalid AOT manifest schema: 'list' object has no attribute 'get'",
    }


def test_unreadable_optional_aot_artifact_is_reported_not_raised(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def unreadable_loader(*_: object, **__: object) -> object:
        raise PermissionError("access denied")

    monkeypatch.setattr(
        qualify_capacity, "load_stationary_aot_artifact", unreadable_loader
    )
    result = qualify_capacity._artifact_verification(
        tmp_path,
        functional=0,
        spin="unpolarized",
        plan=object(),
    )

    assert result == {
        "status": "missing_or_invalid",
        "detail": "access denied",
    }


def test_device_budget_requires_a_positive_remainder() -> None:
    limits = {
        "small_domain": {"atom_count": 32, "ao_count": 128},
        "basis_primitive_count": 4096,
        "primitive_records": 16_000_000,
        "grid_points": 1_000_000,
        "grid_pair_visits": 100_000_000,
        "additional_device_bytes": 512,
        "additional_host_bytes": 256,
    }
    failures = qualify_capacity._case_failures(
        {
            "atom_count": 1,
            "ao_count_spherical": 1,
            "basis_primitive_count": 1,
        },
        {"primitive_records": 1, "grid_points": 1, "grid_pair_visits": 1},
        {
            "additional_device_peak_bound": 512,
            "additional_host_numeric_bound": 1,
        },
        limits,
    )

    assert [item["gate"] for item in failures] == ["additional_device_budget"]
