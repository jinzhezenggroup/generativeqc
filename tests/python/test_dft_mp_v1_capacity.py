import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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
PUBLIC_ROUTE_FILES = (
    "python/vibeqc/calculator.py",
    "python/vibeqc/batch.py",
)
GRID_CONTRACT_FILES = (
    "python/vibeqc/ks.py",
    "python/vibeqc_compiler/dft/grid.py",
    "python/vibeqc_compiler/xc/quadrature_cuda.py",
    "src/dft/cuda_quadrature.cu",
    "src/dft/grid.hpp",
    "src/methods/dft_method.cpp",
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


def copy_contract_files(tmp_path: Path, files: tuple[str, ...]) -> None:
    for relative in files:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            (ROOT / relative).read_text(encoding="utf-8"), encoding="utf-8"
        )


def stationary_contract_tree(tmp_path: Path, source: str) -> None:
    copy_contract_files(tmp_path, ("src/dft/stationary_gradient_cuda.cuh",))
    target = tmp_path / "python/vibeqc/_stationary_cuda.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")


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
        "c08f40375765a126782325dd4d03ded0ea9bf3caa25f23953a8a5cdc5c75c01b"
    )
    assert result["basis"]["stationary_layout_contract_sha256"] == (
        "2f1bb49d43cbfd93e65f69c769ec26c9d04b84bfe5e4be2d705b1262a386b030"
    )
    assert result["basis"]["native_spherical_ao_count_contract_sha256"] == (
        "23785e9e006f9a100b4fecc690e6936a348581beba073507c154b185564832c6"
    )
    assert result["grid"] == {
        "source_only_molecular_grid_sha256": (
            "03a43444cd793167823c0c30c0b66b51c2a464d8f65946118dd781813dc7f0a4"
        ),
        "public_grid_abi_sha256": (
            "b64b0ca7be75b9425c32220476d5eda9b5f013168bd68c856a448fdd320a679e"
        ),
        "native_grid_abi_sha256": (
            "0fa29ffe02ff05df801d8986d06ba03cd492f73d6f621543a96c3f8fc79700af"
        ),
        "generated_quadrature_layout_sha256": (
            "7ad4c84286cce70329233f7aa2dcaf2b934e2e7cf46137cc3ed32cc6076754c3"
        ),
        "native_cuda_grid_sha256": (
            "eed5f5bff7c67622c75fd0d21448b66502b581c102637036a748b459a288a41b"
        ),
        "native_cuda_grid_route_sha256": (
            "cc639c77e261810ff35a30f3bf4967a398b6408e72f86446f94a4d5e760e1a42"
        ),
        "native_grid_point_count_sha256": (
            "92cd50078b7a96f371ed8d4fcdb77930b8c472134bd1e97bba803ac445d85867"
        ),
        "point_count_definition": (
            "atom_count * radial_points * angular_polar * angular_azimuth"
        ),
    }
    assert result["public_route"] == {
        "semilocal_force_predicate_sha256": (
            "fba0a84cb3d993919caf6e6d10391239598ef876cda41123d683479fccf767e0"
        ),
        "force_capability_promotion_sha256": (
            "3ea6ef6ce2c0d8ea5849161ef4ccd706f987261e2ceacdd13c7cfb185525d2d8"
        ),
        "cuda_force_method_sha256": (
            "1d0df874a38441e94168f329e8055f9b9d27e7f26649e15e106db9ab7694c79e"
        ),
        "prepared_aot_selection_sha256": (
            "ed21f18ca4a41d861f0e96310d6a85ea56b03b46a3343fe8741b73cd0182434b"
        ),
    }

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
        "ao_primitive_count_peak": 5,
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
    assert result["admission_limits"]["primitive_records_scope"] == "per_native_page"
    assert result["admission_limits"]["primitive_logical_metric_limit"] == (2**64 - 1)
    assert result["admission_limits"]["primitive_page_budget_bindings"] == [
        "max_primitive_records",
        "max_primitive_records",
    ]
    assert result["admission_limits"]["fixed_task_capacity_definition"] == (
        "min(integral_terms, primitive_tile)"
    )
    assert result["admission_limits"]["primitive_page_execution_definition"] == (
        "task_executor.execute_pages(domain, submit_page)"
    )
    assert result["admission_limits"]["primitive_page_contract_sha256"] == {
        "initializer_sha256": (
            "6ae30e757b7dd4d8df4729d3431d5631db0e53434b58cd2c886eefb8190ab2c6"
        ),
        "flush_sha256": (
            "1c2e0bb83a12eed7113825855cbe2164f53366b6bb270dd6c1247b498737c77b"
        ),
        "bulk_sha256": (
            "d5f2d214d89a6c714edc52d81b6909e14b5c1b962234c6892c91c9d076e9d63b"
        ),
        "scalar_sha256": (
            "c5b8ef983462f6c56ebfe6bd6eb8d5cf98f92f36f3e5425504b596730846205f"
        ),
        "component_integral_sha256": (
            "c0eb9658bb707083073c9ea57b5021825691c35c0cc2ff6e2afa326c09159dc3"
        ),
        "nuclear_sha256": (
            "1e86737d8732ef8637378ab925f829dfe229bcf049219c2705a0a4fbf7afdb85"
        ),
        "component_mode_sha256": (
            "8d9819961d3014d161aff8c5c798f926fe6f1d9de54b84a725fdf2f6694b76bb"
        ),
        "executor_sha256": (
            "71bac6eddd844fcd29830994ad9528bda276557c45efeb12f6dd80ee1fe1146b"
        ),
        "submit_page_sha256": (
            "2fcd280569106fe3cbcf1256a02693aa3532e7db0fa62f7c97d7319454a62a48"
        ),
        "nuclear_pair_loop_sha256": (
            "5a69bf4fd85d28b137e1ae35bce4a1d32134375bbaca9f66f60c9377a0c8f935"
        ),
        "native_owner_sha256": (
            "cb5d69c2486d3566af7bb61f42eabc51df3d0a514b1ee8a00e1e6a74a0339a9a"
        ),
        "native_allocation_sha256": (
            "e680ab29f69ce35c9758e4f3ebd916e889d3f553dc9816dd07e9b7b740624544"
        ),
        "native_create_sha256": (
            "9aee878f0f32fae3f756934074fca0ea57062658e38b32deba9b6af98c94ab26"
        ),
        "native_reset_sha256": (
            "ea2a7df22edca3c3e1f6afad7185dfa7d06ddda5fec0610f8fd31f19ba585a9b"
        ),
        "native_tasks_sha256": (
            "e05af602b22c92dc71b056b0339910a8ac9f7dcc9816b2175d889620ef1024a0"
        ),
        "native_nuclear_sha256": (
            "f5106ec4c238357ece702013e10044c945705b5c9435aae1078ebfa85847c1c2"
        ),
    }
    assert result["admission_limits"]["primitive_records_definition"] == (
        "(1 + int(has_exchange)) * primitive_sum ** 4 + "
        "(na + 2) * primitive_sum ** 2 + na * (na - 1) // 2"
    )
    assert result["admission_limits"]["grid_pair_visits_definition"] == (
        "(1 + 2 * len(state.grid.points)) * na * (na - 1) // 2"
    )
    assert result["admission_limits"]["grid_derivative_order_definition"] == (
        "functional != 0"
    )
    assert result["admission_limits"]["method_ir_definition"] == (
        "state._source.method_ir"
    )
    assert result["admission_limits"]["functional_lowering_definition"] == (
        "_native_semilocal_family(method)"
    )
    assert result["admission_limits"]["grid_plan_definition"] == (
        "plan_tiles(basis, backend='cuda', order=2 if needs_first else 1, "
        "tile_points=tile_points, active_ao_capacity=n, "
        "budget_bytes=max_device_bytes)"
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
    assert result["admission_limits"]["gate_order"] == [
        "small_domain_atom_ao_cap",
        "primitive_topology_cap",
        "primitive_logical_metric_range",
        "grid_point_work_budget",
        "grid_pair_work_budget",
        "additional_device_budget",
        "additional_host_budget",
        "primitive_descriptor_page_budget",
    ]
    assert result["admission_limits"]["gate_predicates"] == {
        "primitive_metric_range": "records > np.iinfo(np.uint64).max",
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
    assert all(
        case["requirements"]["primitive_descriptor_peak_records"] == 625
        for case in cases.values()
    )
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
        "grid_pair_work_budget"
    )
    for sentinel in ("water", "oh", "o2", "water_dimer"):
        assert cases[sentinel]["admission"]["outcome"] == (
            "passes_static_stationary_caps"
        )
        assert cases[sentinel]["admission"]["first_blocker"] is None

    # Losing gates remain visible after the first failure; the report is not a
    # pass/fail truncation that hides the production-scale work.
    water32_gates = [item["gate"] for item in cases["water32"]["admission"]["failures"]]
    assert water32_gates == [
        "small_domain_atom_ao_cap",
        "grid_point_work_budget",
        "grid_pair_work_budget",
    ]


def test_report_covers_every_required_semilocal_fp64_force_row_and_aot_route(
    tmp_path: Path,
) -> None:
    result = report()
    rows = result["rows"]

    assert len(rows) == 22
    assert result["summary"] == {
        "required_semilocal_fp64_force_rows": 22,
        "statically_blocked_rows": 12,
        "rows_passing_static_stationary_caps": 10,
        "scientific_qualification": "NOT_RUN",
    }
    assert {row["method"] for row in rows} == {"lda", "pbe", "r2scan"}
    assert {row["product"] for row in rows} == {"energy+analytic_forces"}
    assert all(row["required"] is True for row in rows)
    assert all(row["public_capability"]["forces"] is True for row in rows)
    assert {
        row["public_capability"]["selector_contract"]["selector"] for row in rows
    } == set(qualify_capacity.SEMILOCAL_ABI_IDS)
    assert all(
        row["public_capability"]["selector_contract"]["stationary_plan_identity"]
        == row["stationary_plan"]["identity"]
        for row in rows
    )
    assert all(
        row["public_route"]["scientific_runtime_compilation_required"] is False
        for row in rows
    )
    assert all(row["packaged_aot"]["source_package_declared"] is True for row in rows)
    assert result["stationary_aot_source_package"] == {
        "cmake_contract_sha256": (
            "c35064b2f437a6fb5bbce301b90e92c806c718b469d9ab539d8ae83237b8cd49"
        ),
        "profiles": [
            "lda_rks",
            "lda_uks",
            "pbe_rks",
            "pbe_uks",
            "r2scan_rks",
            "r2scan_uks",
        ],
        "component_domain": "spd",
    }
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
        192_187_488
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


def test_each_row_uses_its_own_method_memory_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = qualify_capacity._method_resources

    def method_resources(
        basis: object,
        *,
        atom_count: int,
        functional: int,
        spin: str,
        limits: dict,
    ) -> tuple[dict, object]:
        memory, plan = original(
            basis,
            atom_count=atom_count,
            functional=functional,
            spin=spin,
            limits=limits,
        )
        memory = dict(memory)
        if functional == qualify_capacity.SEMILOCAL_FUNCTIONALS["pbe"]:
            memory["additional_device_peak_bound"] = limits["additional_device_bytes"]
        return memory, plan

    monkeypatch.setattr(qualify_capacity, "_method_resources", method_resources)
    rows = {row["id"]: row for row in report()["rows"]}

    assert rows["lda/rks/water/fp64_energy_forces"]["admission"]["outcome"] == (
        "passes_static_stationary_caps"
    )
    assert rows["r2scan/rks/water/fp64_energy_forces"]["admission"]["outcome"] == (
        "passes_static_stationary_caps"
    )
    pbe = rows["pbe/rks/water/fp64_energy_forces"]["admission"]
    assert pbe["outcome"] == "blocked"
    assert pbe["first_blocker"]["gate"] == "additional_device_budget"


def test_public_selector_contract_rejects_changed_semilocal_coefficients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = qualify_capacity.resolve_ks_options

    def changed(selector: str) -> SimpleNamespace:
        options = original(selector)
        return SimpleNamespace(
            coefficients=(0.5, 1.0, 0.0),
            execution_plan=options.execution_plan,
        )

    monkeypatch.setattr(qualify_capacity, "resolve_ks_options", changed)
    plan = qualify_capacity._qualified_aot_plan(1, "unpolarized")

    with pytest.raises(RuntimeError, match="semilocal coefficients"):
        qualify_capacity._public_selector_contract(
            "pbe-rks",
            expected_functional=1,
            expected_spin="unpolarized",
            stationary_plan=plan,
        )


def test_public_selector_contract_rejects_removed_native_eligibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qualify_capacity.generated_methods,
        "NATIVE_DFT_METHOD_IDS",
        qualify_capacity.generated_methods.NATIVE_DFT_METHOD_IDS - {7},
    )
    plan = qualify_capacity._qualified_aot_plan(1, "unpolarized")

    with pytest.raises(RuntimeError, match="native DFT eligibility"):
        qualify_capacity._public_selector_contract(
            "pbe-rks",
            expected_functional=1,
            expected_spin="unpolarized",
            stationary_plan=plan,
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
    monkeypatch.setattr(qualify_capacity, "_clean_git_sha", lambda _, **__: revision)

    result = qualify_capacity.build_report(ROOT)

    assert result["source"]["sha"] == revision


def test_public_report_cannot_exempt_an_arbitrary_dirty_path() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument 'output_path'"):
        qualify_capacity.build_report(  # type: ignore[call-arg]
            ROOT,
            output_path=ROOT / "python/vibeqc/calculator.py",
        )


def test_source_contract_hashes_do_not_use_version_dependent_ast_dump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_ast_dump(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("source contracts must not depend on ast.dump formatting")

    monkeypatch.setattr(qualify_capacity.ast, "dump", reject_ast_dump)

    basis = qualify_capacity._basis_layout_contract(ROOT)
    expansion = qualify_capacity._spd_expansion_contract(ROOT)

    assert basis["native_ao_constructor_contract_sha256"] == (
        qualify_capacity.NATIVE_AO_CONSTRUCTOR_CONTRACT_SHA256
    )
    assert expansion["stationary_layout_contract_sha256"] == (
        qualify_capacity.STATIONARY_LAYOUT_CONTRACT_SHA256
    )


def test_clean_git_sha_ignores_only_the_requested_report_output(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )

    git("init")
    git("config", "user.name", "Capacity Test")
    git("config", "user.email", "capacity@example.invalid")
    (repository / "tracked.json").write_text("{}\n", encoding="utf-8")
    git("add", "tracked.json")
    git("commit", "-m", "fixture")
    revision = git("rev-parse", "HEAD").stdout.strip()

    output = repository / "capacity-report.json"
    output.write_text("first run\n", encoding="utf-8")
    assert qualify_capacity._report_output_exemption(repository, output) == (
        output.resolve()
    )
    assert qualify_capacity._clean_git_sha(repository, ignored_path=output) == revision

    with pytest.raises(ValueError, match="must not replace a tracked file"):
        qualify_capacity._report_output_exemption(
            repository, repository / "tracked.json"
        )
    with pytest.raises(ValueError, match="must be a JSON file"):
        qualify_capacity._report_output_exemption(
            repository, repository / "capacity-report.py"
        )
    aot_directory = repository / "aot"
    aot_directory.mkdir()
    with pytest.raises(ValueError, match="must not overlap AOT evidence"):
        qualify_capacity._report_output_exemption(
            repository,
            aot_directory / "pbe_rks_spd.json",
            aot_directory=aot_directory,
        )

    (repository / "unrelated.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        qualify_capacity._clean_git_sha(repository, ignored_path=output)


def test_primitive_budget_scope_fails_closed_when_whole_force_gate_returns(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/vibeqc/_stationary_cuda.py").read_text(encoding="utf-8")
    whole_force_gate = (
        "    if records > max_primitive_records:\n"
        '        raise ValueError("primitive work budget exceeded")\n'
    )
    assert whole_force_gate not in source
    marker = "    pair_visits = (1 + 2 * len(state.grid.points))"
    assert marker in source
    stationary_contract_tree(
        tmp_path, source.replace(marker, whole_force_gate + marker, 1)
    )

    with pytest.raises(RuntimeError, match="restored a whole-force primitive cap"):
        qualify_capacity._source_limits(tmp_path)


def test_primitive_budget_scope_fails_closed_when_page_contract_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/vibeqc/_stationary_cuda.py").read_text(encoding="utf-8")
    old = "if np.any(primitive_work > self.page_work_budget):"
    assert old in source
    stationary_contract_tree(
        tmp_path,
        source.replace(old, "if np.any(primitive_work >= self.page_work_budget):", 1),
    )

    with pytest.raises(RuntimeError, match="bulk page contract changed"):
        qualify_capacity._source_limits(tmp_path)


def test_primitive_budget_scope_fails_closed_when_component_work_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/vibeqc/_stationary_cuda.py").read_text(encoding="utf-8")
    old = "primitive_work *= int(row[2])"
    assert old in source
    stationary_contract_tree(
        tmp_path, source.replace(old, "primitive_work *= int(row[2]) + 1", 1)
    )

    with pytest.raises(RuntimeError, match="component_integral page contract changed"):
        qualify_capacity._source_limits(tmp_path)


def test_primitive_page_gate_order_fails_closed_when_execution_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/vibeqc/_stationary_cuda.py").read_text(encoding="utf-8")
    original = (
        "            execution = task_executor.execute_pages(domain, submit_page)"
    )
    host_gate = "    if host_bound > max_host_bytes:"
    assert original in source and host_gate in source
    moved = source.replace(original, "            execution = None", 1)
    moved = moved.replace(
        host_gate,
        "    task_executor.execute_pages(domain, submit_page)\n" + host_gate,
        1,
    )
    stationary_contract_tree(tmp_path, moved)

    with pytest.raises(RuntimeError, match="primitive descriptor page order changed"):
        qualify_capacity._source_limits(tmp_path)


def test_primitive_page_gate_fails_closed_when_callback_bypasses_producer(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/vibeqc/_stationary_cuda.py").read_text(encoding="utf-8")
    old = "sources.integral_page("
    assert source.count(old) == 2
    stationary_contract_tree(tmp_path, source.replace(old, "sources.integral(", 1))

    with pytest.raises(RuntimeError, match="submit-page contract changed"):
        qualify_capacity._source_limits(tmp_path)


def test_primitive_page_gate_fails_closed_when_budget_initialization_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/vibeqc/_stationary_cuda.py").read_text(encoding="utf-8")
    old = "self.page_work_budget = int(page_work_budget)"
    assert old in source
    stationary_contract_tree(
        tmp_path,
        source.replace(old, "self.page_work_budget = 2 * int(page_work_budget)", 1),
    )

    with pytest.raises(RuntimeError, match="initializer page contract changed"):
        qualify_capacity._source_limits(tmp_path)


def test_primitive_page_gate_fails_closed_when_native_consumer_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/vibeqc/_stationary_cuda.py").read_text(encoding="utf-8")
    stationary_contract_tree(tmp_path, source)
    target = tmp_path / "src/dft/stationary_gradient_cuda.cuh"
    native = target.read_text(encoding="utf-8")
    old = "size_t(work) > p->max_page_primitive_work"
    assert old in native
    target.write_text(
        native.replace(old, "size_t(work) >= p->max_page_primitive_work", 1),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="native_tasks_sha256 contract changed"):
        qualify_capacity._source_limits(tmp_path)


def test_nuclear_pair_work_fails_closed_when_python_consumer_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/vibeqc/_stationary_cuda.py").read_text(encoding="utf-8")
    old = '        self.flush()\n        kind = self.kinds["nuclear", ()]'
    assert old in source
    stationary_contract_tree(
        tmp_path,
        source.replace(old, '        kind = self.kinds["nuclear", ()]', 1),
    )

    with pytest.raises(RuntimeError, match="nuclear page contract changed"):
        qualify_capacity._source_limits(tmp_path)


def test_nuclear_pair_work_fails_closed_when_endpoint_loop_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/vibeqc/_stationary_cuda.py").read_text(encoding="utf-8")
    old = "            for other in range(atom):\n                sources.nuclear("
    assert old in source
    stationary_contract_tree(
        tmp_path,
        source.replace(
            old,
            "            for other in range(atom + 1):\n                sources.nuclear(",
            1,
        ),
    )

    with pytest.raises(RuntimeError, match="nuclear-pair loop contract changed"):
        qualify_capacity._source_limits(tmp_path)


def test_nuclear_pair_work_fails_closed_when_native_consumer_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/vibeqc/_stationary_cuda.py").read_text(encoding="utf-8")
    stationary_contract_tree(tmp_path, source)
    target = tmp_path / "src/dft/stationary_gradient_cuda.cuh"
    native = target.read_text(encoding="utf-8")
    old = "p->check_page_primitive_work(1);"
    assert native.count(old) == 1
    target.write_text(
        native.replace(old, "p->check_page_primitive_work(2);", 1),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="native_nuclear_sha256 contract changed"):
        qualify_capacity._source_limits(tmp_path)


def test_memory_bounds_fail_closed_when_native_allocation_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/vibeqc/_stationary_cuda.py").read_text(encoding="utf-8")
    stationary_contract_tree(tmp_path, source)
    target = tmp_path / "src/dft/stationary_gradient_cuda.cuh"
    native = target.read_text(encoding="utf-8")
    old = "(579 + 3 * stationary_source_count) * na"
    assert old in native
    target.write_text(
        native.replace(old, "(580 + 3 * stationary_source_count) * na", 1),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="native_allocation_sha256 contract changed"):
        qualify_capacity._source_limits(tmp_path)


def test_primitive_budget_scope_fails_closed_when_work_definition_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/vibeqc/_stationary_cuda.py").read_text(encoding="utf-8")
    old = "(1 + int(has_exchange)) * primitive_sum**4"
    assert old in source
    stationary_contract_tree(tmp_path, source.replace(old, "primitive_sum**3", 1))

    with pytest.raises(RuntimeError, match="primitive-record definition"):
        qualify_capacity._source_limits(tmp_path)


def test_memory_bounds_fail_closed_when_production_definition_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/vibeqc/_stationary_cuda.py").read_text(encoding="utf-8")
    old = "22 * primitive_tile"
    assert old in source
    stationary_contract_tree(tmp_path, source.replace(old, "23 * primitive_tile", 1))

    with pytest.raises(RuntimeError, match="source-bytes definition"):
        qualify_capacity._source_limits(tmp_path)


def test_grid_memory_fails_closed_when_production_plan_inputs_move(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/vibeqc/_stationary_cuda.py").read_text(encoding="utf-8")
    old = "order=2 if needs_first else 1"
    first = source.index(old)
    production = source.index(old, first + len(old))
    stationary_contract_tree(
        tmp_path,
        source[:production] + "order=1" + source[production + len(old) :],
    )

    with pytest.raises(RuntimeError, match="grid-plan input definition changed"):
        qualify_capacity._source_limits(tmp_path)


def test_grid_memory_fails_closed_when_functional_lowering_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/vibeqc/_stationary_cuda.py").read_text(encoding="utf-8")
    old = "functional = _native_semilocal_family(method)"
    assert old in source
    stationary_contract_tree(tmp_path, source.replace(old, "functional = 0", 1))

    with pytest.raises(RuntimeError, match="functional-family lowering"):
        qualify_capacity._source_limits(tmp_path)


def test_public_selector_contract_rejects_changed_functional_lowering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qualify_capacity, "_native_semilocal_family", lambda _: 0)
    plan = qualify_capacity._qualified_aot_plan(1, "unpolarized")

    with pytest.raises(RuntimeError, match="native functional-family lowering"):
        qualify_capacity._public_selector_contract(
            "pbe-rks",
            expected_functional=1,
            expected_spin="unpolarized",
            stationary_plan=plan,
        )


def test_memory_bounds_fail_closed_when_host_gate_moves(tmp_path: Path) -> None:
    source = (ROOT / "python/vibeqc/_stationary_cuda.py").read_text(encoding="utf-8")
    old = "if host_bound > max_host_bytes:"
    assert old in source
    stationary_contract_tree(
        tmp_path, source.replace(old, "if host_bound >= max_host_bytes:", 1)
    )

    with pytest.raises(RuntimeError, match="additional-host predicate"):
        qualify_capacity._source_limits(tmp_path)


def test_admission_gate_order_fails_closed_when_leading_gates_move(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/vibeqc/_stationary_cuda.py").read_text(encoding="utf-8")
    small = (
        "    if not 1 <= na <= 32 or not 1 <= n <= 128:\n"
        '        raise ValueError("CUDA diagnostic small-domain atom/AO cap exceeded")\n'
    )
    primitives = (
        "    if not 1 <= basis.nprimitive <= 4096:\n"
        '        raise ValueError("CUDA diagnostic primitive-topology cap exceeded")\n'
    )
    assert small + primitives in source
    stationary_contract_tree(
        tmp_path, source.replace(small + primitives, primitives + small, 1)
    )

    with pytest.raises(RuntimeError, match="admission gate order changed"):
        qualify_capacity._source_limits(tmp_path)


def test_admission_gate_order_fails_closed_when_memory_gates_move(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/vibeqc/_stationary_cuda.py").read_text(encoding="utf-8")
    device = (
        "    if available <= 0:\n"
        '        raise ValueError("stationary additional-device budget exceeded")\n'
    )
    host = (
        "    if host_bound > max_host_bytes:\n"
        '        raise ValueError("stationary additional-host byte budget exceeded")\n'
    )
    assert device in source and host in source
    swapped = source.replace(device, "    # swapped-memory-gate\n", 1)
    swapped = swapped.replace(host, device, 1)
    swapped = swapped.replace("    # swapped-memory-gate\n", host, 1)
    stationary_contract_tree(tmp_path, swapped)

    with pytest.raises(RuntimeError, match="admission gate order changed"):
        qualify_capacity._source_limits(tmp_path)


def test_packaged_aot_claim_fails_closed_when_cmake_wiring_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "cmake/VibeQCCuda.cmake").read_text(encoding="utf-8")
    old = "          --component-domain spd"
    assert old in source
    target = tmp_path / "cmake/VibeQCCuda.cmake"
    target.parent.mkdir(parents=True)
    target.write_text(
        source.replace(old, "          --component-domain sp", 1), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="packaged-AOT CMake contract changed"):
        qualify_capacity._source_package_inventory(tmp_path)


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


def test_spherical_ao_count_fails_closed_when_native_count_moves(
    tmp_path: Path,
) -> None:
    spd_contract_tree(tmp_path)
    target = tmp_path / "src/molecule/basis.cpp"
    source = target.read_text(encoding="utf-8")
    old = "? 2 * static_cast<std::size_t>(shell.angular_momentum) + 1"
    assert old in source
    target.write_text(source.replace(old, old[:-1] + "2", 1), encoding="utf-8")

    with pytest.raises(RuntimeError, match="spherical AO count contract changed"):
        qualify_capacity._spd_expansion_contract(tmp_path)


def test_public_capability_fails_closed_when_complete_predicate_moves(
    tmp_path: Path,
) -> None:
    copy_contract_files(tmp_path, PUBLIC_ROUTE_FILES)
    target = tmp_path / "python/vibeqc/calculator.py"
    source = target.read_text(encoding="utf-8")
    old = "self._ks_options.coefficients == (1.0, 1.0, 0.0)"
    first = source.index(old)
    semilocal = source.index(old, first + len(old))
    target.write_text(
        source[:semilocal] + "False" + source[semilocal + len(old) :],
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="semilocal force predicate changed"):
        qualify_capacity._source_public_route(tmp_path)


def test_public_capability_fails_closed_when_promotion_condition_moves(
    tmp_path: Path,
) -> None:
    copy_contract_files(tmp_path, PUBLIC_ROUTE_FILES)
    target = tmp_path / "python/vibeqc/calculator.py"
    source = target.read_text(encoding="utf-8")
    old = "and self._method in _method_manifest.NATIVE_DFT_METHOD_IDS"
    assert old in source
    target.write_text(source.replace(old, "and False", 1), encoding="utf-8")

    with pytest.raises(RuntimeError, match="force capability promotion changed"):
        qualify_capacity._source_public_route(tmp_path)


def test_public_cuda_force_fails_closed_when_packaged_route_moves(
    tmp_path: Path,
) -> None:
    copy_contract_files(tmp_path, PUBLIC_ROUTE_FILES)
    target = tmp_path / "python/vibeqc/batch.py"
    source = target.read_text(encoding="utf-8")
    old = "and not state._source.method_ir.full_range_exact_exchange"
    assert old in source
    target.write_text(source.replace(old, "and False", 1), encoding="utf-8")

    with pytest.raises(RuntimeError, match="public CUDA force route changed"):
        qualify_capacity._source_public_route(tmp_path)


def test_prepared_aot_route_fails_closed_when_selection_moves(
    tmp_path: Path,
) -> None:
    relative = "python/vibeqc/_stationary_cuda.py"
    copy_contract_files(tmp_path, (relative,))
    target = tmp_path / relative
    source = target.read_text(encoding="utf-8")
    old = "if aot_directory is None or ecp"
    assert old in source
    target.write_text(source.replace(old, "if True", 1), encoding="utf-8")

    with pytest.raises(RuntimeError, match="AOT selection contract changed"):
        qualify_capacity._prepared_aot_route_contract(tmp_path)


def test_grid_count_fails_closed_when_native_cuda_shape_moves(
    tmp_path: Path,
) -> None:
    copy_contract_files(tmp_path, GRID_CONTRACT_FILES)
    target = tmp_path / "src/dft/cuda_quadrature.cu"
    source = target.read_text(encoding="utf-8")
    old = "q::product(system.atoms.size(), per_atom)"
    assert old in source
    target.write_text(
        source.replace(old, "q::product(system.atoms.size() + 1, per_atom)", 1),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="grid point-count contract changed"):
        qualify_capacity._grid_count_contract(tmp_path)


def test_grid_count_fails_closed_when_source_only_shape_moves(
    tmp_path: Path,
) -> None:
    copy_contract_files(tmp_path, GRID_CONTRACT_FILES)
    target = tmp_path / "python/vibeqc_compiler/dft/grid.py"
    source = target.read_text(encoding="utf-8")
    old = "len(atoms) * len(r) * len(angular)"
    assert old in source
    target.write_text(source.replace(old, "len(r) * len(angular)", 1), encoding="utf-8")

    with pytest.raises(RuntimeError, match="grid point-count contract changed"):
        qualify_capacity._grid_count_contract(tmp_path)


def test_grid_count_fails_closed_when_generated_layout_moves(
    tmp_path: Path,
) -> None:
    copy_contract_files(tmp_path, GRID_CONTRACT_FILES)
    target = tmp_path / "python/vibeqc_compiler/xc/quadrature_cuda.py"
    source = target.read_text(encoding="utf-8")
    old = "l.points = points;"
    assert old in source
    target.write_text(
        source.replace(old, "l.points = points + 1;", 1), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="grid point-count contract changed"):
        qualify_capacity._grid_count_contract(tmp_path)


def test_grid_count_fails_closed_when_native_backend_route_moves(
    tmp_path: Path,
) -> None:
    copy_contract_files(tmp_path, GRID_CONTRACT_FILES)
    target = tmp_path / "src/methods/dft_method.cpp"
    source = target.read_text(encoding="utf-8")
    old = "return dft::MolecularGrid::from_cuda(system, spec, device);"
    assert old in source
    target.write_text(
        source.replace(old, "return dft::MolecularGrid(system, spec);", 1),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="grid point-count contract changed"):
        qualify_capacity._grid_count_contract(tmp_path)


def test_grid_count_fails_closed_when_public_abi_lowering_moves(
    tmp_path: Path,
) -> None:
    copy_contract_files(tmp_path, GRID_CONTRACT_FILES)
    target = tmp_path / "python/vibeqc/ks.py"
    source = target.read_text(encoding="utf-8")
    old = "        grid.radial_points,"
    assert old in source
    target.write_text(
        source.replace(old, "        grid.radial_points + 1,", 1),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="grid point-count contract changed"):
        qualify_capacity._grid_count_contract(tmp_path)


def test_grid_count_fails_closed_when_native_abi_lowering_moves(
    tmp_path: Path,
) -> None:
    copy_contract_files(tmp_path, GRID_CONTRACT_FILES)
    target = tmp_path / "src/methods/dft_method.cpp"
    source = target.read_text(encoding="utf-8")
    old = "grid.radial_points = input.radial_points;"
    assert old in source
    target.write_text(
        source.replace(old, "grid.radial_points = input.radial_points + 1;", 1),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="grid point-count contract changed"):
        qualify_capacity._grid_count_contract(tmp_path)


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


def test_report_rejects_same_checkout_helper_source_changed_since_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _ = qualify_capacity._IMPORTED_HELPER_SOURCES["Atom"]
    monkeypatch.setitem(
        qualify_capacity._IMPORTED_HELPER_SOURCES,
        "Atom",
        (path, "0" * 64),
    )

    with pytest.raises(RuntimeError, match="helper source changed since import: Atom"):
        qualify_capacity._assert_local_imports()


def test_report_requires_a_fresh_interpreter_after_checkout_head_moves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qualify_capacity, "_IMPORTED_TOOL_HEAD", "0" * 40)

    with pytest.raises(RuntimeError, match="start a fresh interpreter"):
        qualify_capacity._assert_local_imports()


def test_report_reloads_basis_data_instead_of_reusing_a_stale_cache() -> None:
    baseline = next(case for case in report()["cases"] if case["id"] == "water")
    qualify_capacity._basis_pack.cache_clear()
    qualify_capacity._named_basis_record.cache_clear()
    pack = qualify_capacity._basis_pack()
    bases = pack["bases"]
    assert isinstance(bases, dict)
    shells = bases["def2-svp"]["elements"]["1"]
    coefficients = shells[0]["coefficients"]
    original = coefficients[0]
    coefficients[0] = "999.0"
    try:
        result = report()
    finally:
        coefficients[0] = original
        qualify_capacity._basis_pack.cache_clear()
        qualify_capacity._named_basis_record.cache_clear()

    water = next(case for case in result["cases"] if case["id"] == "water")
    assert water["identities"]["basis"] == baseline["identities"]["basis"]


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
        "primitive_logical_metric_limit": 2**64 - 1,
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
        {
            "primitive_records": 1,
            "primitive_descriptor_peak_records": 1,
            "grid_points": 1,
            "grid_pair_visits": 1,
        },
        {
            "additional_device_peak_bound": 512,
            "additional_host_numeric_bound": 1,
        },
        limits,
    )

    assert [item["gate"] for item in failures] == ["additional_device_budget"]


def test_primitive_descriptor_budget_is_page_local_and_ordered_after_host() -> None:
    limits = {
        "small_domain": {"atom_count": 32, "ao_count": 128},
        "basis_primitive_count": 4096,
        "primitive_records": 16_000_000,
        "primitive_logical_metric_limit": 2**64 - 1,
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
        {
            "primitive_records": 16_000_001,
            "primitive_descriptor_peak_records": 16_000_001,
            "grid_points": 1,
            "grid_pair_visits": 1,
        },
        {
            "additional_device_peak_bound": 1,
            "additional_host_numeric_bound": 1,
        },
        limits,
    )

    assert [item["gate"] for item in failures] == ["primitive_descriptor_page_budget"]


def test_logical_primitive_metric_retains_uint64_range_gate() -> None:
    limits = {
        "small_domain": {"atom_count": 32, "ao_count": 128},
        "basis_primitive_count": 4096,
        "primitive_records": 16_000_000,
        "primitive_logical_metric_limit": 2**64 - 1,
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
        {
            "primitive_records": 2**64,
            "primitive_descriptor_peak_records": 1,
            "grid_points": 1,
            "grid_pair_visits": 1,
        },
        {
            "additional_device_peak_bound": 1,
            "additional_host_numeric_bound": 1,
        },
        limits,
    )

    assert [item["gate"] for item in failures] == ["primitive_logical_metric_range"]
