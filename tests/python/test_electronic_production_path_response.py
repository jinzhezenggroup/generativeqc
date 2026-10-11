"""Response/HVP path anchors must not imply an unqualified resident cutover."""

from tools.check_electronic_production_paths import DEFAULT_LEDGER, load_and_validate


def _rows() -> dict:
    payload, errors = load_and_validate(DEFAULT_LEDGER)
    assert errors == []
    assert payload["coverage"] == "pilot"
    return {row["id"]: row for row in payload["rows"]}


def test_cpu_and_cuda_lambda_keep_one_scientific_owner() -> None:
    rows = _rows()
    cpu = rows["cc-response-cpu-rccsd-lambda"]
    cuda = rows["cc-response-cuda-rccsd-lambda"]
    assert cpu["scientific_owner"] == cuda["scientific_owner"]
    assert cpu["provider_owner"] == cuda["provider_owner"]
    assert cpu["domain"] == cuda["domain"]
    assert cuda["exception"]["owner"] == "src/cc/lambda_response_cuda.cu"
    assert (
        "Default conventional and DF Lambda retain host vectors"
        in cuda["exception"]["missing_capability"]
    )
    assert "opt-in" in cuda["exception"]["missing_capability"]
    assert "DF residency" in cuda["exception"]["missing_capability"]
    assert "before default promotion" in cuda["exception"]["retirement_condition"]
    assert "diagonal policy" in cuda["exception"]["retirement_condition"]
    assert cuda["evidence_levels"]["public"]["state"] == "missing"


def test_resident_cphf_tool_is_not_public_derivative_promotion() -> None:
    row = _rows()["hf-response-cuda-resident-cphf"]
    assert row["status"] == "qualification-only"
    assert row["scientific_owner"] == "python/generativeqc/response_operator.py"
    assert row["execution_owner"] == "tools/generativeqc_response/resident_cuda.py"
    assert row["evidence_levels"]["public"]["state"] == "missing"
    assert row["evidence_levels"]["derivative"]["state"] == "missing"
    assert row["exception"]["retirement_condition"]


def test_cuda_cpks_and_rhf_frame_do_not_hide_host_krylov() -> None:
    rows = _rows()
    for identifier in ("dft-response-cuda-semilocal-cpks", "hf-response-cuda-frame-z"):
        row = rows[identifier]
        assert row["status"] == "production"
        assert "Krylov" in row["exception"]["missing_capability"]
        assert row["exception"]["owner"] == row["execution_owner"]
        assert row["evidence_levels"]["public"]["state"] == "missing"


def test_uks_tooling_is_not_relabelled_as_installed_rks_ownership() -> None:
    rows = _rows()
    rks = rows["dft-response-cuda-semilocal-cpks"]
    uks = rows["dft-response-cuda-semilocal-uks-cpks"]
    assert rks["domain"] == "lda-pbe-rks-fixed-grid"
    assert uks["domain"] == "lda-pbe-uks-fixed-grid"
    assert rks["provider_owner"] == "python/generativeqc/rks_response.py"
    assert uks["provider_owner"] == "tools/generativeqc_response/native_ks.py"
    assert uks["status"] == "qualification-only"
    assert uks["evidence_levels"]["public"]["state"] == "missing"


def test_mp2_bindings_keep_science_but_expose_cuda_host_block_seams() -> None:
    rows = _rows()
    cpu = rows["mp2-force-cpu-conventional-response"]
    cuda = rows["mp2-force-cuda-conventional-response"]
    assert cpu["scientific_owner"] == cuda["scientific_owner"]
    assert cpu["execution_owner"] == cuda["execution_owner"]
    assert cpu["domain"] == cuda["domain"]
    assert "host MO blocks" in cuda["exception"]["missing_capability"]
    assert "stationarity" in cuda["exception"]["retirement_condition"]


def test_cuda_hvp_missing_sources_remain_explicit_and_fail_closed() -> None:
    rows = _rows()
    cpu = rows["dft-hvp-cpu-semilocal-rks"]
    cuda = rows["dft-hvp-cuda-semilocal-rks"]
    assert cpu["scientific_owner"] == cuda["scientific_owner"]
    assert cpu["execution_owner"] == cuda["execution_owner"]
    assert cuda["status"] == "unsupported"
    assert "#1905" in cuda["blocker"]
    assert "Becke" in cuda["blocker"]
    assert cuda["evidence_levels"]["represented"]["state"] == "present"
    for level in (
        "compiled-cuda",
        "device-executed",
        "domain-qualified",
        "molecular",
        "derivative",
        "public",
    ):
        assert cuda["evidence_levels"][level]["state"] == "missing"
        assert cuda["evidence_levels"][level]["evidence"] == []


def test_gfn2_stationary_d4_is_not_external_dft_or_hvp_support() -> None:
    rows = _rows()
    for backend in ("cpu", "cuda"):
        row = rows[f"gfn2-force-{backend}-stationary-d4"]
        assert row["product"] == "force"
        assert row["scientific_owner"] == "src/xtb/native/src/model/gfn2/force.cpp"
        assert "self-consistent-d4" in row["domain"]
        assert "not external DFT D4" in row["exception"]["missing_capability"]
