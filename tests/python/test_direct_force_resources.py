"""Protect architecture, image, and completeness boundaries of force receipts."""

import json
import subprocess
from pathlib import Path

import pytest

from benchmarks import direct_force_resources
from benchmarks.direct_force_resources import final_binary_kind, linked_resources


def image(architecture: str, fields: str, symbol: str = "bounded_force") -> str:
    """Create the relevant CUOBJDump ELF structure, not a device measurement."""
    return (
        "Fatbin elf code:\n================\n"
        f"arch = {architecture}\nResource usage:\n"
        f" Function {symbol}:\n  {fields}\n"
    )


def test_images_and_architectures_are_not_coalesced() -> None:
    output = "".join(
        image(architecture, f"REG:{registers} STACK:84056 SHARED:4108 LOCAL:0")
        for architecture, registers in [("sm_120", 255), ("sm_120", 128), ("sm_90", 64)]
    )
    rows = linked_resources(output, "bounded_force")
    assert [row["image_index"] for row in rows] == [0, 1, 2]
    assert [row["architecture"] for row in rows] == ["sm_120", "sm_120", "sm_90"]
    assert [row["registers_per_thread"] for row in rows] == [255, 128, 64]
    assert all(row["stack_bytes_per_thread"] == 84056 for row in rows)
    assert all(row["static_shared_bytes"] == 4108 for row in rows)
    assert all(row["local_bytes_per_thread"] == 0 for row in rows)


def test_ptx_is_not_linked_resource_evidence() -> None:
    ptx = "Fatbin ptx code:\narch = sm_120\n Function bounded_force:\n  REG:1\n"
    elf = image("sm_120", "REG:255 STACK:64 SHARED:0 LOCAL:0")
    assert len(linked_resources(ptx + elf + ptx, "bounded_force")) == 1
    with pytest.raises(ValueError, match="no matching"):
        linked_resources(ptx, "bounded_force")


@pytest.mark.parametrize(
    "output, message",
    [
        (image("sm_120", "REG:255 STACK:0 SHARED:0"), "incomplete"),
        (
            image("sm_120", "REG:255 STACK:0 SHARED:0 LOCAL:0").replace(
                "arch = sm_120\n", ""
            ),
            "architecture",
        ),
        (
            image("sm_120", "REG:255 STACK:0 SHARED:0 LOCAL:0", "unrelated"),
            "no matching",
        ),
        ("", "no matching"),
    ],
)
def test_missing_evidence_is_not_reported_as_zero_work(
    output: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        linked_resources(output, "bounded_force")


def test_duplicate_symbols_within_one_image_are_rejected() -> None:
    output = image("sm_120", "REG:255 STACK:0 SHARED:0 LOCAL:0")
    output += " Function bounded_force:\n  REG:128 STACK:0 SHARED:0 LOCAL:0\n"
    with pytest.raises(ValueError, match="duplicate"):
        linked_resources(output, "bounded_force")


@pytest.mark.parametrize("missing_line", ["", "\n", "\n\n"])
def test_missing_selected_resource_line_cannot_hide_behind_a_valid_entry(
    missing_line: str,
) -> None:
    output = image("sm_120", "REG:255 STACK:0 SHARED:0 LOCAL:0")
    output += f" Function bounded_force_missing:{missing_line}"
    with pytest.raises(ValueError, match="incomplete"):
        linked_resources(output, "bounded_force")


def test_next_function_header_is_not_a_missing_entry_resource_line() -> None:
    output = image("sm_120", "REG:255 STACK:0 SHARED:0 LOCAL:0")
    output += " Function bounded_force_missing:\n"
    output += " Function __cuda_helper:\n  REG:1 STACK:0 SHARED:0 LOCAL:0\n"
    with pytest.raises(ValueError, match="incomplete"):
        linked_resources(output, "bounded_force")


def test_unrelated_local_device_functions_do_not_invalidate_receipt() -> None:
    output = image("sm_120", "REG:255 STACK:0 SHARED:0 LOCAL:0")
    output += " Function __cuda_helper:\n  REG:1 STACK:0 SHARED:0 LOCAL:0\n" * 2
    assert len(linked_resources(output, "bounded_force")) == 1


@pytest.mark.parametrize("byte_order", ["little", "big"])
@pytest.mark.parametrize(
    "elf_type, kind", [(2, "executable"), (3, "shared-library-or-pie")]
)
def test_final_binary_type_is_not_inferred_from_filename(
    tmp_path: Path, byte_order: str, elf_type: int, kind: str
) -> None:
    header = bytearray(20)
    header[:4] = b"\x7fELF"
    header[4] = 2
    header[5] = 1 if byte_order == "little" else 2
    header[16:18] = elf_type.to_bytes(2, byte_order)
    header[18:20] = (62).to_bytes(2, byte_order)
    binary = tmp_path / "misleading.o"
    binary.write_bytes(header)
    assert final_binary_kind(binary) == kind


@pytest.mark.parametrize("elf_type, machine", [(1, 62), (0, 62), (2, 190)])
def test_unlinked_objects_and_cubins_are_not_final_receipts(
    tmp_path: Path, elf_type: int, machine: int
) -> None:
    header = bytearray(20)
    header[:4] = b"\x7fELF"
    header[4:6] = bytes([2, 1])
    header[16:18] = elf_type.to_bytes(2, "little")
    header[18:20] = machine.to_bytes(2, "little")
    binary = tmp_path / "misleading.so"
    binary.write_bytes(header)
    with pytest.raises(ValueError, match="not a final host binary"):
        final_binary_kind(binary)


@pytest.mark.parametrize("content", [b"!<arch>\n", b"", b"\x7fELF" + b"\x00" * 16])
def test_malformed_or_non_elf_input_is_rejected(tmp_path: Path, content: bytes) -> None:
    binary = tmp_path / "not-a-library.so"
    binary.write_bytes(content)
    with pytest.raises(ValueError, match="final host ELF"):
        final_binary_kind(binary)


def test_cli_refuses_reviewed_results_before_running_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The raw collector must not replace the evidence publication/review gate."""
    destination = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/results"
        / f"{tmp_path.name}-forbidden-resource.json"
    )

    def unexpected_command(*arguments: object, **options: object) -> None:
        raise AssertionError("Inspection ran before validating its output boundary")

    monkeypatch.setattr(direct_force_resources.subprocess, "run", unexpected_command)
    monkeypatch.setattr(
        "sys.argv",
        [
            "direct_force_resources.py",
            "--binary",
            str(tmp_path / "missing-library.so"),
            "--output",
            str(destination),
        ],
    )
    with pytest.raises(SystemExit) as failure:
        direct_force_resources.main()
    assert failure.value.code == 2
    error = capsys.readouterr().err
    assert "invalid raw_output_path value" in error
    assert "benchmarks/results/" in error
    assert not destination.exists()


def test_cli_retains_linked_reservations_in_allowed_raw_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Synthetic parser fixtures are not genuine linked or device evidence."""
    header = bytearray(20)
    header[:4] = b"\x7fELF"
    header[4:6] = bytes([2, 1])
    header[16:18] = (3).to_bytes(2, "little")
    header[18:20] = (62).to_bytes(2, "little")
    binary = tmp_path / "library.so"
    binary.write_bytes(header)
    destination = tmp_path / "raw/resources.json"
    resource_output = image("sm_120", "REG:255 STACK:90680 SHARED:4108 LOCAL:0")

    def command(
        arguments: list[str], **options: object
    ) -> subprocess.CompletedProcess[str]:
        assert options["check"] and options["capture_output"] and options["text"]
        output = "bounded_force\n" if arguments[0] == "c++filt" else resource_output
        return subprocess.CompletedProcess(arguments, 0, stdout=output, stderr="")

    monkeypatch.setattr(direct_force_resources.subprocess, "run", command)
    monkeypatch.setattr(
        "sys.argv",
        [
            "direct_force_resources.py",
            "--binary",
            str(binary),
            "--kernel-pattern",
            "bounded_force",
            "--output",
            str(destination),
        ],
    )
    direct_force_resources.main()
    report = json.loads(destination.read_text())
    assert report["binary_sha256"] == direct_force_resources.file_digest(binary)
    assert report["rows"][0]["static_shared_bytes"] == 4108
    assert report["rows"][0]["stack_bytes_per_thread"] == 90680
    assert report["rows"][0]["local_bytes_per_thread"] == 0
