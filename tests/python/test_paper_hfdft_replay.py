"""Original-byte, process independence and plotted-value failure boundaries."""

import json
import lzma
import shutil
import subprocess
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks import paper_hfdft_replay as replay


def cohort() -> tuple[list[dict], list[dict], set[tuple]]:
    records, commands, expected = [], [], set()
    for repeat in range(3):
        name = f"hf-direct-3-standard-r{repeat}"
        path = "/measurement/" + name + ".json"
        record = {
            "arm": "standard",
            "method": "hf",
            "mode": "direct",
            "atoms": 3,
            "process_repeat": repeat,
            "raw_file": path,
            "status": "complete",
            "source_revision": "a" * 40,
            "samples": [
                {
                    "phase": phase,
                    "repeat": call,
                    "seconds": 1.2345678901234567,
                    "energy": -75.12345678901235,
                    "forces": [[0.0, 0.1, -0.1]] * 3,
                    "converged": True,
                    "iterations": 1,
                    "fock_builds": 2,
                    "semantic_work": {"submitted_items": 9},
                }
                for phase, call in (("cold", 0), ("warm", 0), ("warm", 1), ("warm", 2))
            ],
        }
        record["raw_sha256"] = replay.digest(replay.raw_receipt(record))
        commands.append(
            {
                "name": name,
                "stage": "formal",
                "status": "complete",
                "returncode": 0,
                "process_repeat": repeat,
                "raw_sha256": record["raw_sha256"],
                "argv": [
                    "python",
                    "--method",
                    "hf",
                    "--mode",
                    "direct",
                    "--atoms",
                    "3",
                    "--warm-repeats",
                    "3",
                    "--source-revision",
                    "a" * 40,
                    "--output",
                    path,
                ],
            }
        )
        records.append(record)
        expected.add(("standard", "hf", "direct", 3, repeat))
    return records, commands, expected


def test_distinct_commands_may_have_identical_observations() -> None:
    records, commands, expected = cohort()
    assert len({r["raw_sha256"] for r in records}) == 1
    replay.authenticate(records, commands, expected)
    original = json.loads(replay.raw_receipt(records[0]))
    assert original["samples"] == records[0]["samples"]
    assert "process_repeat" not in original


@pytest.mark.parametrize(
    "defect", ["path", "repeat", "command", "hash", "missing", "extra"]
)
def test_invalid_process_ownership_fails_closed(defect: str) -> None:
    records, commands, expected = cohort()
    if defect == "path":
        records[1]["raw_file"] = records[0]["raw_file"]
    elif defect == "repeat":
        records[1]["process_repeat"] = 0
    elif defect == "command":
        commands[1]["argv"][commands[1]["argv"].index("--method") + 1] = "pbe"
    elif defect == "hash":
        records[1]["samples"][0]["energy"] += 0.1
    elif defect == "missing":
        records.pop()
    else:
        commands.append(deepcopy(commands[0]))
    with pytest.raises(ValueError):
        replay.authenticate(records, commands, expected)


@pytest.mark.parametrize("defect", ["phase", "force", "converged"])
def test_scientific_endpoint_is_checked_after_raw_authentication(defect: str) -> None:
    records, commands, expected = cohort()
    if defect == "phase":
        records[0]["samples"][1]["repeat"] = 1
    elif defect == "force":
        records[0]["samples"][1]["forces"].pop()
    else:
        records[0]["samples"][1]["converged"] = False
    records[0]["raw_sha256"] = replay.digest(replay.raw_receipt(records[0]))
    commands[0]["raw_sha256"] = records[0]["raw_sha256"]
    with pytest.raises(ValueError):
        replay.authenticate(records, commands, expected)


def test_json_and_compression_ambiguities_fail_closed() -> None:
    good = lzma.compress(b'{"value":1}')
    assert replay.read_xz(good) == {"value": 1}
    for bad in (good[:-1], good + good):
        with pytest.raises((ValueError, lzma.LZMAError)):
            replay.read_xz(bad)
    for bad in (b'{"x":1,"x":2}', b'{"x":NaN}'):
        with pytest.raises(ValueError):
            replay.read_json(bad)


def test_source_path_cannot_escape_new_destination(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        replay.reconstruct(
            {
                "base_revision": "a" * 40,
                "keep_git_bytes": ["../escape"],
                "patch_paths": [],
                "crlf_extra_paths": [],
            },
            tmp_path / "source",
        )
    assert not (tmp_path / "source").exists()


@pytest.mark.parametrize("field", ["keep_git_bytes", "patch_paths", "crlf_extra_paths"])
def test_git_archive_source_refuses_patch_rules(tmp_path: Path, field: str) -> None:
    source = {
        "base_revision": "a" * 40,
        "native_export": "git-archive",
        "keep_git_bytes": [],
        "patch_paths": [],
        "crlf_extra_paths": [],
    }
    source[field] = ["src/retained.cpp"]
    with pytest.raises(ValueError, match="git-archive"):
        replay.reconstruct(source, tmp_path / "source")
    assert not (tmp_path / "source").exists()


def test_cohort_without_device_publication_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(replay.ROOT / "python"))
    with pytest.raises(ValueError, match="no device publication"):
        replay.verify(tmp_path)


def test_publication_path_rejected_before_file_io(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        replay.publication_files(tmp_path, {"files": [{"path": "../not-present.json"}]})


def svg_fixture() -> tuple[list[dict], bytes]:
    ns = "{http://www.w3.org/2000/svg}"
    root = ET.Element(ns + "svg")
    rows = []
    for number in range(16):
        mode = ("direct", "df")[number // 8]
        method = list(replay.METHODS)[number % 8]
        axis = ET.SubElement(root, ns + "g", id=f"axes_{number + 1}")
        for i, (y, value) in enumerate(((100, 0), (0, 10))):
            tick = ET.SubElement(axis, ns + "g", id=f"ytick_{i}")
            ET.SubElement(tick, ns + "use", y=str(y))
            ET.SubElement(tick, ns + "text").text = str(value)
        for series, (arm, color, phase) in enumerate(
            (a, c, p) for a, c in replay.COLORS.items() for p in ("cold", "warm")
        ):
            values = [1 + series + size / 100 for size in replay.SIZES]
            style = f"stroke: {color};" + (
                "stroke-dasharray: 3,2" if phase == "cold" else ""
            )
            line = ET.SubElement(axis, ns + "g", id=f"line2d_{series}")
            ET.SubElement(
                line,
                ns + "path",
                d=" ".join(
                    f"{'M' if i == 0 else 'L'} {x} {100 - y * 10}"
                    for i, (x, y) in enumerate(zip(replay.SIZES, values, strict=True))
                ),
                style=style,
            )
            bounds = ET.SubElement(axis, ns + "g", id=f"LineCollection_{series}")
            for size, value in zip(replay.SIZES, values, strict=True):
                ET.SubElement(
                    bounds,
                    ns + "path",
                    d=f"M {size} {100 - (value - 0.01) * 10} L {size} {100 - (value + 0.01) * 10}",
                    style=style,
                )
                key = [arm, method, mode, size]
                row = next((r for r in rows if r["key"] == key), None)
                if row is None:
                    row = {"key": key}
                    rows.append(row)
                row[phase] = {"median": value, "min": value - 0.01, "max": value + 0.01}
    return rows, ET.tostring(root)


def test_svg_values_are_independently_checked() -> None:
    rows, raw = svg_fixture()
    assert replay.compare_svg(rows, raw)["medians_checked"] == 256
    tampered = raw.replace(b"M 3 89.7", b"M 3 79.7", 1)
    assert tampered != raw
    with pytest.raises(ValueError, match="SVG values"):
        replay.compare_svg(rows, tampered)


def test_only_explicit_svg_byte_representations_are_accepted() -> None:
    published = b"<svg>\n  <g/>\n</svg>\n"
    checkout = published.replace(b"\n", b"\r\n")
    binding = {
        "schema": "generativeqc.published-svg-binding.v1",
        "sha256": replay.digest(published),
        "recorded_publication_checkout": {
            "sha256": replay.digest(checkout),
            "line_endings": "CRLF",
        },
    }
    assert replay.bound_figure_bytes(published, binding) == published
    assert replay.bound_figure_bytes(checkout, binding) == published
    for unexpected in (
        published.replace(b"\n", b"\r\n", 1),
        checkout.replace(b"<g/>", b"<path/>"),
    ):
        with pytest.raises(ValueError, match="figure bytes"):
            replay.bound_figure_bytes(unexpected, binding)


@pytest.mark.parametrize("line_endings", ["LF", "CRLF"])
def test_actual_cohorts_replay_with_published_svg_serialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, line_endings: str
) -> None:
    """Exercise the documented real-data path, not just synthetic loader records."""
    root, dataset = replay.ROOT, replay.DEFAULT
    monkeypatch.syspath_prepend(str(root / "python"))
    reference = Path("benchmarks/results/readme-refresh-20261026/oracles.json.xz")
    (tmp_path / reference).parent.mkdir(parents=True)
    (tmp_path / reference).write_bytes((root / reference).read_bytes())
    for device in ("h100", "rtx5090"):
        provenance = replay.read_xz(
            (dataset / device / "provenance.json.xz").read_bytes()
        )
        path = Path(provenance["figure"]["path"])
        # Construct a canonical committed-file fixture independently of the loader.
        published = (root / path).read_bytes().replace(b"\r\n", b"\n")
        assert replay.digest(published) == provenance["figure"]["sha256"]
        contents = (
            published if line_endings == "LF" else published.replace(b"\n", b"\r\n")
        )
        (tmp_path / path).parent.mkdir(exist_ok=True)
        (tmp_path / path).write_bytes(contents)
    monkeypatch.setattr(replay, "ROOT", tmp_path)
    result = replay.verify(dataset)
    assert result["h100"]["wins"] == {"cold": 63, "warm": 62}
    assert result["rtx5090"]["wins"] == {"cold": 64, "warm": 64}
    for cohort in result.values():
        assert len(cohort["records"]) == 384
        assert cohort["numeric_counts"] == {"qualified": 348, "failed": 36}
        assert cohort["plot_replay"]["range_endpoints_checked"] == 512


def test_fresh_autocrlf_checkout_preserves_real_publications(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise Git's actual checkout filters for the complete authenticated data."""
    root = replay.ROOT
    source, checked = tmp_path / "repository", tmp_path / "checkout"
    source.mkdir()
    checked.mkdir()
    dataset = Path("benchmarks/results/paper-hfdft-cold3-20261115")
    shutil.copytree(root / dataset, source / dataset)
    paths = [
        Path(".gitattributes"),
        Path("benchmarks/results/readme-refresh-20261026/oracles.json.xz"),
    ]
    paths.extend(
        Path(f"assets/paper-hfdft-cold3-{device}.svg") for device in ("h100", "rtx5090")
    )
    for path in paths:
        target = source / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / path).read_bytes())
    for argv in (
        ["git", "init", "--quiet"],
        ["git", "config", "core.autocrlf", "true"],
        ["git", "add", "--all"],
        [
            "git",
            "checkout-index",
            "--all",
            "--prefix=" + checked.resolve().as_posix() + "/",
        ],
    ):
        subprocess.run(argv, cwd=source, check=True, capture_output=True)
    monkeypatch.syspath_prepend(str(root / "python"))
    monkeypatch.setattr(replay, "ROOT", checked)
    result = replay.verify(checked / dataset)
    assert result["rtx5090"]["wins"] == {"cold": 64, "warm": 64}
    assert len(result["h100"]["records"]) == 384


def test_master_h100_cohort_replays_single_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The H100-only master cohort replays without an RTX publication."""
    root = replay.ROOT
    dataset = root / "benchmarks/results/paper-hfdft-cold3-20261118"
    monkeypatch.syspath_prepend(str(root / "python"))
    reference = Path("benchmarks/results/readme-refresh-20261026/oracles.json.xz")
    (tmp_path / reference).parent.mkdir(parents=True)
    (tmp_path / reference).write_bytes((root / reference).read_bytes())
    provenance = replay.read_xz((dataset / "h100" / "provenance.json.xz").read_bytes())
    path = Path(provenance["figure"]["path"])
    (tmp_path / path).parent.mkdir(exist_ok=True)
    (tmp_path / path).write_bytes((root / path).read_bytes().replace(b"\r\n", b"\n"))
    monkeypatch.setattr(replay, "ROOT", tmp_path)
    result = replay.verify(dataset)
    assert set(result) == {"h100"}
    cohort = result["h100"]
    assert cohort["wins"] == {"cold": 64, "warm": 62}
    assert len(cohort["records"]) == 384
    assert cohort["numeric_counts"] == {"qualified": 348, "failed": 36}
    assert cohort["plot_replay"]["range_endpoints_checked"] == 512
