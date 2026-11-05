"""Generate the offline Ni/H basis input from independently pinned BSE data.

This generator imports neither GenerativeQC nor an electronic-structure
provider. Source file hashes match the primary BSE repository at the declared
commit; generated payload hashes are basis provenance, not source fingerprints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PIN = "4adaf1372c7101620ca1a9f3130be9ae97fb8f30"
LICENSE_HASH = "14b5a1c21a9e0966e295b9e3d66c0cee9475ffe5931e74d063c3713e2ae9a496"
SOURCES = {
    "def2-SVP.metadata.json": "2e979d9f915f397e4e3cc8e2b73c859836c420bce1e4e059dc0defc4571b12ff",
    "def2-SVP.1.table.json": "f5785281f477069def945358872208c8f8c1c6fe589b5ec42eeee6568726588a",
    "ahlrichs/def2-SVP.1.element.json": "809185107d1ad063e55a13117dba23b4e8a3b53406c0259f88d9cbc81c32587f",
    "ahlrichs/SV/def2-SV-base.1.json": "9a83096a009c034a18c4e5fffa93e9022e8a2ff2e2f1ea17621dad86ec01a5a8",
    "ahlrichs/SV/def2-SVP-add.1.json": "2206794676d13abcb4df8a778b9a4a1f3c7cc7c2345622aca6937b01dfde5722",
}


def _serialize(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def main() -> int:
    """Verify source provenance before generating or checking the local payload."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bse-package",
        type=Path,
        help="Parent directory containing the pinned basis_set_exchange package",
    )
    parser.add_argument(
        "--output-directory", type=Path, default=root / "tests/data/global_hybrid"
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.bse_package is not None:
        sys.path.insert(0, str(args.bse_package.resolve()))
    import basis_set_exchange as bse

    package = Path(bse.__file__).resolve().parent
    if (
        args.bse_package is not None
        and package != args.bse_package.resolve() / "basis_set_exchange"
    ):
        raise RuntimeError("BSE import does not match the requested package")
    for relative, expected in SOURCES.items():
        if (
            hashlib.sha256((package / "data" / relative).read_bytes()).hexdigest()
            != expected
        ):
            raise RuntimeError(
                f"BSE source differs from pinned commit {PIN}: {relative}"
            )
    license_path = root / "tests/data/external_basis/BSE-LICENSE"
    if hashlib.sha256(license_path.read_bytes()).hexdigest() != LICENSE_HASH:
        raise RuntimeError("retained BSE license differs from the source pin")
    name = "def2-svp-ni-h.json"
    payload = _serialize(bse.get_basis("def2-svp", elements=[1, 28], version=1))
    manifest = _serialize(
        {
            "schema": "generativeqc.global-hybrid-basis-reference/v1",
            "source_repository": "https://github.com/MolSSI-BSE/basis_set_exchange.git",
            "source_commit": PIN,
            "basis_name": "def2-SVP",
            "basis_version": 1,
            "elements": [1, 28],
            "representation": "real-spherical",
            "source_sha256": SOURCES,
            "license": "BSD-3-Clause",
            "license_file": "../external_basis/BSE-LICENSE",
            "license_sha256": LICENSE_HASH,
            "file": name,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    )
    for filename, contents in ((name, payload), ("manifest.json", manifest)):
        destination = args.output_directory / filename
        if args.check:
            if destination.read_bytes() != contents:
                raise RuntimeError(f"generated basis input is stale: {destination}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(contents)
        print(f"verified {filename}: {hashlib.sha256(contents).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
