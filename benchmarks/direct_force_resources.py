"""Record linked Direct-force resources without loading CUDA or claiming speedups.

Use the final shared library, not PTX or an unlinked object: device callees affect
the linked register and stack reservations. Runtime traffic, occupancy, stalls,
and complete endpoint times need separate Slurm-scheduled device measurements.
The linked SHARED value may include driver-reserved storage that runtime kernel
attributes exclude. LOCAL is kept separate from STACK; a zero LOCAL field is
not evidence of zero thread-local state or executed local-memory traffic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from benchmarks._retention import raw_output_path
from tools.generativeqc_validation.f_shell import cuobjdump_resources


def linked_resources(output: str, kernel_pattern: str) -> list[dict]:
    """Keep each ELF image distinct, including repeated architectures/symbols.

    The existing strict resource parser rejects incomplete records and duplicate
    symbols within an image. PTX has no final linked resource reservation and is
    deliberately excluded. An empty match is an error, not a zero-work receipt.
    Preserve tool-reported reservations without inventing a driver-reservation
    subtraction or substituting STACK for LOCAL.
    """
    pattern = re.compile(kernel_pattern)
    rows = []
    for image_index, section in enumerate(output.split("Fatbin elf code:")[1:]):
        image = section.split("Fatbin ptx code:", 1)[0]
        architecture = re.search(r"^arch = (\S+)$", image, re.MULTILINE)
        if architecture is None:
            raise ValueError("ELF resource image has no architecture")
        # Linked images may repeat unrelated local CUDA helper symbols. Check
        # completeness and uniqueness of the selected kernel inventory only.
        selected = []
        for record in re.finditer(r"^ Function (\w+):(?:\n|$)", image, re.MULTILINE):
            if not pattern.search(record[1]):
                continue
            resource_line = image[record.end() :].split("\n", 1)[0]
            if not resource_line.strip():
                raise ValueError("incomplete CUOBJDump kernel resources")
            selected.append(f" Function {record[1]}:\n{resource_line}\n")
        for symbol, resources in cuobjdump_resources("".join(selected)).items():
            rows.append(
                {
                    "image_index": image_index,
                    "architecture": architecture[1],
                    "symbol": symbol,
                    "registers_per_thread": resources["REG"],
                    "stack_bytes_per_thread": resources["STACK"],
                    "static_shared_bytes": resources["SHARED"],
                    "local_bytes_per_thread": resources["LOCAL"],
                }
            )
    if not rows:
        raise ValueError("no matching linked kernel resources")
    return rows


def file_digest(path: Path) -> str:
    """Hash large libraries incrementally; do not load the production runtime."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_binary_kind(path: Path) -> str:
    """Reject relocatable objects/archives and standalone cubins as final receipts.

    ELF type does not certify CUDA device-link correctness; the build receipt
    must still identify the actual production link. This check prevents a raw
    compiler object from being mislabeled as a final shared-library inspection.
    """
    with path.open("rb") as source:
        header = source.read(20)
    if (
        len(header) != 20
        or header[:4] != b"\x7fELF"
        or header[4] not in (1, 2)
        or header[5] not in (1, 2)
    ):
        raise ValueError("resource receipt requires a final host ELF binary")
    byte_order = "little" if header[5] == 1 else "big"
    elf_type = int.from_bytes(header[16:18], byte_order)
    machine = int.from_bytes(header[18:20], byte_order)
    if elf_type not in (2, 3) or machine == 190:
        raise ValueError(
            "unlinked object, archive, or standalone cubin is not a final host binary"
        )
    return "shared-library-or-pie" if elf_type == 3 else "executable"


def main() -> None:
    """Seal the inspected binary, tool output, and demangled resource inventory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--cuobjdump", default="cuobjdump")
    parser.add_argument("--demangler", default="c++filt")
    parser.add_argument(
        "--kernel-pattern", default="bounded_direct_shell_quartet_kernel"
    )
    parser.add_argument(
        "--output",
        type=raw_output_path,
        required=True,
        help="Raw receipt path; reviewed results require tools/evidence.py publish",
    )
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    binary_digest = file_digest(binary)
    binary_kind = final_binary_kind(binary)
    command = [args.cuobjdump, "--dump-resource-usage", str(binary)]
    result = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=300
    )
    if file_digest(binary) != binary_digest:
        raise RuntimeError("binary changed during resource inspection")
    rows = linked_resources(result.stdout, args.kernel_pattern)
    names = subprocess.run(
        [args.demangler],
        input="".join(row["symbol"] + "\n" for row in rows),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.splitlines()
    if len(names) != len(rows):
        raise ValueError("demangler did not preserve the kernel inventory")
    for row, name in zip(rows, names):
        row["name"] = name
    report = {
        "schema_version": 1,
        "scope": "linked static resources, not executed work or performance",
        "binary": str(binary),
        "binary_kind": binary_kind,
        "binary_sha256": binary_digest,
        "cuobjdump_command": command,
        "cuobjdump_stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
        "kernel_pattern": args.kernel_pattern,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
