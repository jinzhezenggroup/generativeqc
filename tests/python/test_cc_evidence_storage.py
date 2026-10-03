"""Pin every original byte through the approved existing-main CC compaction."""

import gzip
import hashlib
import json
from pathlib import Path

from tools.generativeqc_validation.record import load_json

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "9e9b938239d31564d6f92e9b99b922aee0dbb1cc"
EXPECTED = {
    "benchmarks/results/cc-large-force-20261003/cpu-records.json": (
        "ffdaa95c31a7b88e7eda919813961c365a01b9ffb0637959296254bc734511ab",
        "3ebdc95243a3c27f885ed31d411a28ca892b0a22ea6dfadf5c70f185a1dd4407",
        "47ccf39702925832b6e3cc9f7f994758f1641b83",
        57100,
        7837,
    ),
    "benchmarks/results/cc-large-force-20261003/cuda-records.json": (
        "946c3f538e7f7dae78b4615e2ec6491d1d18a7cb09a2d2c7189ffe40cd8e77e7",
        "7560e8778adc496f67d6d86e3e0289eb3e1d3b96802f34029fe02232b229c619",
        "34235536c9b1753cfc3c4c7e1d4e2401924d036d",
        73348,
        11018,
    ),
    "benchmarks/results/cc-large-force-20261003/experimental-comparator.patch": (
        "5dd3ca84b4a9d8d4a9b0fec75cd7605464fe541ab6eea7ff0ab5426a6ff89f27",
        "60064a66846e1495d8480167d0f84d05a5c018c078595da7a59e93720c7f6116",
        "c4865c4248f5f82ef5783d80c3a0a3a766f7f70b",
        13556,
        4273,
    ),
    "benchmarks/results/cc-large-force-20261003/summary.json": (
        "78b76dcdf3c6deb06bc7edac4c26526ed76e54599a0603d1c829b7c276112ef4",
        "01e4854a6eeac54245defa94b744e455ea46066be5fabb4bd558d313efacadd3",
        "061079a3aae04b130be801a4be7c23cdcc3c0cba",
        17722,
        4858,
    ),
    "benchmarks/results/cc-mo-tiles-20261003/summary.json": (
        "1ad653d52cfb7e2c5c6bc772524da8ec24bea71c41e7bda6e98c7709a859cb05",
        "ed1d3aa32eebe5be7930410ac63d2e7dca3bc6042aa78c7581fb185fefa2a42b",
        "30244a6501e0bfb6bd0a18eb4ea3c415d1bde0b0",
        101547,
        14036,
    ),
    "benchmarks/results/ccsdt-arena-20261002/cpu-summary.json": (
        "81c1545e72be09045b3768e2f3bae7f3136ee618f8d7995443b6163df5267161",
        "8921ae76adde69f13991144d46d353ab5f73315e1ebc8e369d6013664f2ac4db",
        "8ddde93edb5b81738db986af8f8a652b8641d7a3",
        42756,
        5862,
    ),
    "benchmarks/results/ccsdt-arena-20261002/cuda-summary.json": (
        "da9821021fc57d99044d1e39b2b1066233e160e40eb1c50f77952dec66684046",
        "4bc50fa4399b851ef7bba9f87af7e0443e66511f893a7af907ecc7e014ac7909",
        "f99dfca279b6648e656b3293cf7232f721082a0e",
        52485,
        7021,
    ),
    "benchmarks/results/ccsdt-replay-20261002/cpu-summary.json": (
        "63d3cfd81f3f2d8000cbe09b0ae2ad1b287cd50b8cd6f7ab0142b838149f7671",
        "b81c0848b05a531edbf3ffa2de49491e87c0ec51163ac2bdfb08cf37567a3219",
        "9dfa3f7751996f548eb76966c5c376a775131e74",
        116960,
        9490,
    ),
    "benchmarks/results/ccsdt-replay-20261002/cuda-summary.json": (
        "dfd56ed298447e0724e977a87cabd1ebb6cd5c5598f9c37de1f1f1a4f7172b30",
        "73d12e39757ed957e2447e52960588234c8dd08c46912b158b07bf1b388f89ca",
        "b946fb71b505be061afa328b84d0fc0b512718f5",
        52624,
        7038,
    ),
}


def test_approved_cc_storage_preserves_original_bytes_and_readers() -> None:
    seen = set()
    families = {Path(name).parent for name in EXPECTED}
    for family in sorted(families):
        storage = json.loads((ROOT / family / "storage.json").read_text())
        assert storage["schema"] == "generativeqc.cc-evidence-storage.v1"
        assert storage["storage_source_revision"] == SOURCE
        for entry in storage["records"]:
            path = (family / entry["original_path"]).as_posix()
            assert path in EXPECTED and path not in seen
            seen.add(path)
            original_sha, stored_sha, blob_sha, original_bytes, stored_bytes = EXPECTED[
                path
            ]
            assert entry["stored_path"] == entry["original_path"] + ".gz"
            assert entry["encoding"] == "gzip"
            packed = (ROOT / family / entry["stored_path"]).read_bytes()
            original = gzip.decompress(packed)
            assert not (ROOT / path).exists()
            assert len(packed) == entry["stored_bytes"] == stored_bytes
            assert len(original) == entry["original_bytes"] == original_bytes
            assert (
                hashlib.sha256(packed).hexdigest()
                == entry["stored_sha256"]
                == stored_sha
            )
            assert (
                hashlib.sha256(original).hexdigest()
                == entry["original_sha256"]
                == original_sha
            )
            blob = hashlib.sha1(
                f"blob {len(original)}\0".encode() + original
            ).hexdigest()
            assert blob == entry["original_git_blob"] == blob_sha
            if entry["original_path"].endswith(".json"):
                assert load_json(ROOT / family / entry["stored_path"]) == json.loads(
                    original
                )
            else:
                assert original.startswith(b"diff --git ")
    assert seen == set(EXPECTED)
