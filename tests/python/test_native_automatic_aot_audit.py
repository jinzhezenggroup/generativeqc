"""Offline binary/resource evidence cannot borrow a different execution receipt."""

from copy import deepcopy

import pytest

from tools.audit_native_automatic_aot import audit_receipt


def inputs() -> tuple[dict, dict, dict]:
    """Minimal explicit transport; real cubin inspection is an offline gate."""
    binary = {"sha256": "a" * 64, "bytes": 123}
    receipt = {
        "force_completed": True,
        "fallback_reason": "none",
        "point_artifact": binary.copy(),
        "force_artifact": binary.copy(),
        "force_profile": "libxc_196792_rks",
        "identifier": "GGA_X_APBE",
    }
    counters = {"REG": 32, "STACK": 96, "LOCAL": 0, "SHARED": 0}
    resources = {
        "automatic_gga_x_apbe_point_provider_kernel": counters.copy(),
        "generativeqc_native_libxc_196792_rks_geometry_kernel": counters.copy(),
    }
    return receipt, binary, resources


def test_receipt_joins_actual_binary_and_both_executed_scopes() -> None:
    receipt, binary, resources = inputs()
    joined = audit_receipt(receipt, binary, resources)
    assert joined["profile"] == receipt["force_profile"]
    assert all(joined["scopes"].values())


def test_linked_zero_register_helper_does_not_replace_kernel_evidence() -> None:
    """cuobjdump also reports auxiliary device functions after device linking."""
    receipt, binary, resources = inputs()
    helper = "generativeqc_native_libxc_196792_rks_stationary_evaluate_point"
    resources[helper] = dict.fromkeys(("REG", "STACK", "LOCAL", "SHARED"), 0)
    assert helper in audit_receipt(receipt, binary, resources)["scopes"]["force"]
    del resources["generativeqc_native_libxc_196792_rks_geometry_kernel"]
    with pytest.raises(ValueError, match="no kernel resource"):
        audit_receipt(receipt, binary, resources)


@pytest.mark.parametrize(
    "damage",
    (
        "binary",
        "point-binary",
        "incomplete",
        "fallback",
        "missing-point",
        "missing-force",
        "missing-counter",
        "register-overflow",
    ),
)
def test_audit_fails_closed_on_stale_or_missing_evidence(damage: str) -> None:
    receipt, binary, resources = deepcopy(inputs())
    if damage == "binary":
        binary["sha256"] = "b" * 64
    elif damage == "point-binary":
        receipt["point_artifact"]["bytes"] += 1
    elif damage == "incomplete":
        receipt["force_completed"] = False
    elif damage == "fallback":
        receipt["fallback_reason"] = "force-not-executed-at-current-token"
    elif damage.startswith("missing-") and damage != "missing-counter":
        resources.pop(list(resources)[0 if damage == "missing-point" else 1])
    elif damage == "missing-counter":
        next(iter(resources.values())).pop("STACK")
    else:
        next(iter(resources.values()))["REG"] = 256
    with pytest.raises(ValueError):
        audit_receipt(receipt, binary, resources)


@pytest.mark.parametrize(
    "damage",
    (None, "callback", "point-key", "consumer-key", "tile", "launches", "points"),
)
@pytest.mark.parametrize("worker_prefix", ("executable_", "parameter_family_"))
def test_shared_audit_requires_actual_executables_and_accounted_point_work(
    damage: str | None,
    worker_prefix: str,
) -> None:
    receipt, binary, resources = inputs()
    receipt.update(
        schema="generativeqc.native-automatic-artifact.v2",
        point_executable_artifact=binary.copy(),
        force_executable_artifact=binary.copy(),
        polarized_executable_identity="b" * 64,
        restricted_executable_identity="c" * 64,
        force_consumer_profile="shared_" + "d" * 24,
        force_consumer_identity="e" * 64,
        staged_point_work={"tile_bytes": 4096, "launches": 2, "points": 64},
    )
    point, force = resources.values()
    resources = {
        worker_prefix + "b" * 64 + "_point_kernel": point,
        "generativeqc_native_shared_" + "d" * 24 + "_geometry_kernel": force,
    }
    if damage == "callback":
        receipt["force_executable_artifact"]["sha256"] = "f" * 64
    elif damage == "point-key":
        receipt["polarized_executable_identity"] = ""
    elif damage == "consumer-key":
        receipt["force_consumer_identity"] = ""
    elif damage:
        receipt["staged_point_work"]["tile_bytes" if damage == "tile" else damage] = 0
    if damage:
        with pytest.raises(ValueError):
            audit_receipt(receipt, binary, resources)
    else:
        assert all(audit_receipt(receipt, binary, resources)["scopes"].values())
