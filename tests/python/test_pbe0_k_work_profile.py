"""Fail-closed J/K attribution for class-resolved PBE0 cold CUDA traces."""

from __future__ import annotations

import sqlite3

import pytest

from benchmarks.pbe0_k_work_profile import summarize_classes


def trace_database() -> sqlite3.Connection:
    """Use distinct J/K times so mistaken all-value attribution is visible."""
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
    connection.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
        "(start INTEGER, end INTEGER, demangledName INTEGER, deviceId INTEGER, contextId INTEGER, streamId INTEGER)"
    )
    names = [
        "generated_sm120_psss_shell_class_fock_rhf_streaming_kernel",
        "generated_sm120_ddds_shell_class_fock_rhf_streaming_kernel",
        "void bounded_direct_dddd_streaming_kernel<(DirectScreeningPurpose)0>()",
        "generated_sm120_psss_shell_class_fock_rhf_work_streaming_kernel",
    ]
    connection.executemany("INSERT INTO StringIds VALUES (?, ?)", enumerate(names))
    for pass_index in range(4):
        for shell_class in range(3):
            started = (3 * pass_index + shell_class) * 1000
            name = 3 if pass_index % 2 and shell_class == 0 else shell_class
            duration = (shell_class + 1) * (100 if pass_index % 2 else 10)
            connection.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, 0, 1, 7)",
                (started, started + duration, name),
            )
    return connection


def test_separates_k_and_includes_native_high_angular_class() -> None:
    with trace_database() as connection:
        report = summarize_classes(connection, 2)
    assert report["K_seconds"] == pytest.approx(1200 / 1e9)
    assert report["high_angular_K_fraction"] == pytest.approx(5 / 6)
    assert report["psss_psps_K_fraction"] == pytest.approx(1 / 6)
    assert report["primitive_work_counts"] is None
    classes = {item["angular_class"]: item for item in report["classes"]}
    assert classes["dddd"]["K_seconds"] == pytest.approx(600 / 1e9)
    assert classes["psss"]["work_buckets"]
    assert classes["ddds"]["J_launches"] == 2


@pytest.mark.parametrize("corruption", ["missing", "order", "stream", "work-J"])
def test_refuses_ambiguous_dispatches(corruption: str) -> None:
    with trace_database() as connection:
        if corruption == "missing":
            connection.execute(
                "DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start = 11000"
            )
        elif corruption == "order":
            connection.execute(
                "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET demangledName = 0 WHERE start = 1000"
            )
        elif corruption == "stream":
            connection.execute(
                "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET streamId = 8 WHERE start = 1000"
            )
        else:
            connection.execute(
                "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET demangledName = 3 WHERE start = 0"
            )
        with pytest.raises(ValueError):
            summarize_classes(connection, 2)


def test_refuses_missing_fock_work() -> None:
    with trace_database() as connection, pytest.raises(ValueError):
        summarize_classes(connection, 0)
