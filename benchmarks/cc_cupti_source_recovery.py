"""Recover historical tooling from the publication's real source tree, without execution."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import zlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from benchmarks._retention import raw_output_path
from benchmarks.cc_tile_environment import canonical_hash
from benchmarks.cc_trace_reader import require

PUBLICATION = "benchmarks/results/cc-cupti-attribution-230ao-20261019"
RECOVERY_SCHEMA = "generativeqc.cc-source-recovery.publication-tree.v1"
COMPANIONS = tuple(
    "benchmarks/" + name
    for name in (
        "_retention.py",
        "cc_cupti_campaign.py",
        "cc_cupti_clock.py",
        "cc_cupti_device_probe.cu",
        "cc_cupti_report.py",
        "cc_cupti_timeline.cpp",
        "cc_source_census.py",
        "cc_source_journal.cpp",
        "cc_source_session.hpp",
        "cc_tile_capture_diagnostics.py",
        "cc_tile_child.py",
        "cc_tile_environment.py",
        "cc_tile_pipeline_campaign.py",
        "cc_trace_io.hpp",
        "cc_trace_reader.py",
        "df_ccsdt_force_endpoint.cpp",
    )
)
PATCH_TARGETS = frozenset(
    (
        "benchmarks/cc_cupti_campaign.py",
        "benchmarks/cc_source_census.py",
        "frozen-campaign/worker.py",
    )
)


class SourceUnavailable(ValueError):
    """A local Git object required for closed recovery is unavailable."""


def bounded_gzip(raw: bytes) -> bytes:
    """Decode one finite member; concatenated members and expansion bombs refuse."""
    decoder = zlib.decompressobj(31)
    value = decoder.decompress(raw, 1024 * 1024 + 1)
    require(
        len(value) <= 1024 * 1024
        and decoder.eof
        and not decoder.unconsumed_tail
        and not decoder.unused_data,
        "unsupported compressed evidence size/tail",
    )
    return value


def _git(repository: Path, arguments: list[str], *, data: bytes | None = None) -> bytes:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_SYSTEM=os.devnull,
        GIT_NO_LAZY_FETCH="1",
        GIT_ALLOW_PROTOCOL="",
        GIT_TERMINAL_PROMPT="0",
    )
    try:
        result = subprocess.run(
            [
                "git",
                "--no-replace-objects",
                "-C",
                str(repository),
                "-c",
                "core.autocrlf=false",
                *arguments,
            ],
            input=data,
            capture_output=True,
            timeout=15,
            check=False,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SourceUnavailable(
            "local Git unavailable or bounded recovery operation expired"
        ) from error
    if result.returncode:
        raise SourceUnavailable(
            "required local Git object or recovery operation unavailable; automatic transports are disabled"
        )
    return result.stdout


def _blob(repository: Path, tree: str, path: str) -> tuple[bytes, str]:
    entry = _git(repository, ["ls-tree", "-z", tree, "--", path])
    if not entry:
        raise SourceUnavailable(
            "required publication source entry unavailable; automatic transports are disabled"
        )
    require(
        entry.endswith(b"\0") and entry.count(b"\0") == 1,
        "missing/ambiguous publication source entry",
    )
    header, name = entry[:-1].split(b"\t", 1)
    mode, kind, oid = header.split()
    require(
        mode in (b"100644", b"100755") and kind == b"blob" and name.decode() == path,
        "publication source is not a regular Git blob",
    )
    size = int(_git(repository, ["cat-file", "-s", oid.decode()]))
    require(0 < size <= 512 * 1024, "publication source exceeds admission bound")
    raw = _git(repository, ["cat-file", "blob", oid.decode()])
    require(len(raw) == size, "publication Git blob size differs")
    return raw, oid.decode()


def _identity(raw: bytes, identity: dict) -> bool:
    return (
        type(identity.get("bytes")) is int
        and len(raw) == identity["bytes"]
        and hashlib.sha256(raw).hexdigest() == identity.get("sha256")
    )


def recover_tooling_sources(
    repository: Path, revision: str, output: Path, expected_capsule: dict
) -> dict[str, Any]:
    """Recover all17 measured inputs from a publication-containing tree.

    Git blobs are read without checkout or replacement objects. All transports
    and prompts are disabled. Older Git may enter a blocked lazy-fetch path,
    but missing data cannot be retrieved or produce successful reconstruction.
    Only the declared reverse patch is applied in a new, exclusive directory;
    none of the historical source is imported or executed. Historical identity
    is verified after patching, independently of canonical-source identity.
    """
    require(
        bool(re.fullmatch(r"HEAD|[0-9a-f]{40}|[0-9a-f]{64}", revision)),
        "an exact local source revision or HEAD is required",
    )
    output = raw_output_path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("source recovery requires a new output directory")
    tree = (
        _git(repository, ["rev-parse", "--verify", revision + "^{tree}"])
        .decode()
        .strip()
    )
    require(
        bool(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", tree)), "invalid source tree"
    )
    manifest_raw, _ = _blob(repository, tree, PUBLICATION + "/publication.json")
    manifest = json.loads(manifest_raw)
    entries = manifest["files"]
    names = {entry["path"] for entry in entries}
    require(
        len(entries) == 5
        and names
        == {
            "README.md",
            "campaign.json.gz",
            "oracle.json.gz",
            "measured-tooling.patch.gz",
            "validation.json.gz",
        },
        "publication member inventory differs",
    )
    members = {}
    for entry in entries:
        raw, _ = _blob(repository, tree, PUBLICATION + "/" + entry["path"])
        require(_identity(raw, entry), "publication member identity differs")
        members[entry["path"]] = raw
    retained_capsule = json.loads(bounded_gzip(members["campaign.json.gz"]))
    require(
        canonical_hash(retained_capsule) == canonical_hash(expected_capsule),
        "source tree does not contain this capsule publication",
    )
    record = json.loads(bounded_gzip(members["validation.json.gz"]))
    recovery = record["historical_tooling_reconstruction"]
    require(recovery["schema"] == RECOVERY_SCHEMA, "unsupported source recovery recipe")
    originals = record["historical_tooling_input_identities"]
    canonical = recovery["canonical_input_identities"]
    require(
        set(originals) == {*COMPANIONS, "worker.py"}
        and set(canonical) == set(COMPANIONS),
        "measured/canonical input inventory differs",
    )
    require(
        originals["worker.py"]["sha256"]
        == retained_capsule["original_documents"]["worker.py"]["sha256"],
        "original worker identity differs from frozen packet",
    )
    source_bytes = {}
    for name, identity in canonical.items():
        raw, oid = _blob(repository, tree, name)
        require(
            _identity(raw, identity) and oid == identity["git_blob_oid"],
            "canonical source blob identity differs",
        )
        source_bytes[name] = raw
    patch = bounded_gzip(members["measured-tooling.patch.gz"])
    patch_entry = next(
        entry for entry in entries if entry["path"] == "measured-tooling.patch.gz"
    )
    require(
        len(patch) == patch_entry["decoded_bytes"]
        and hashlib.sha256(patch).hexdigest() == patch_entry["decoded_sha256"],
        "reverse source patch identity differs",
    )
    headers = []
    for line in patch.splitlines():
        if line.startswith(b"diff --git "):
            match = re.fullmatch(rb"diff --git a/(\S+) b/(\S+)", line)
            require(
                match is not None and match[1] == match[2], "unsafe source patch header"
            )
            headers.append(match[1].decode())
        elif line.startswith(
            (
                b"old mode ",
                b"new mode ",
                b"deleted file mode ",
                b"rename ",
                b"copy ",
                b"GIT binary patch",
                b"Binary files",
            )
        ):
            raise ValueError("source patch mode/rename/binary operation refused")
        elif line.startswith(b"new file mode "):
            require(
                line == b"new file mode 100644"
                and headers[-1] == "frozen-campaign/worker.py",
                "source patch file mode refused",
            )
        elif line.startswith((b"--- ", b"+++ ")):
            permitted = {b"--- /dev/null"} | {
                prefix + name.encode()
                for name in PATCH_TARGETS
                for prefix in (b"--- a/", b"+++ b/")
            }
            require(line in permitted, "source patch path refused")
    require(
        len(headers) == len(PATCH_TARGETS) and set(headers) == PATCH_TARGETS,
        "reverse source patch target inventory differs",
    )
    for name, identity in originals.items():
        expected = "frozen-campaign/worker.py" if name == "worker.py" else name
        require(
            identity["reconstructed_path"] == expected,
            "measured reconstruction path differs",
        )
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    root = output.resolve()
    for name, raw in source_bytes.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        require(
            path.parent.resolve().is_relative_to(root), "recovery directory escaped"
        )
        with path.open("xb") as stream:
            stream.write(raw)
    _git(output, ["init", "--quiet", "--template=", "."])
    _git(output, ["apply", "--check", "-"], data=patch)
    _git(output, ["apply", "-"], data=patch)
    for identity in originals.values():
        path = output / identity["reconstructed_path"]
        require(
            not path.is_symlink()
            and path.resolve().is_relative_to(root)
            and _identity(path.read_bytes(), identity),
            "recovered measured source identity differs",
        )
    return {
        "status": "PASS_RECONSTRUCTED_17_MEASURED_TOOLING_INPUTS",
        "publication_source_revision": revision,
        "publication_source_tree": tree,
        "recovered_input_count": len(originals),
        "historical_source_executed": False,
        "native_scientific_requalification": "NOT_CLAIMED",
    }
