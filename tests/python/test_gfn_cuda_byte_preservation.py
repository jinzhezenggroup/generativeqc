"""Local, source-only preservation gate for embedded GFN CUDA refactors.

Run the complete gate (no CUDA compiler or GPU is needed):

    python tests/python/test_gfn_cuda_byte_preservation.py --base 6a02e4b1

The default pytest run tests the gate itself. Set GFN_CUDA_PRESERVATION_BASE to
the desired local commit to opt into the historical comparison under pytest.
This keeps ordinary/shallow-clone tests independent of retained Git history.

Bodies include both braces and every original byte between them. Comments,
strings, whitespace, inactive preprocessor branches, and repeated identical
bodies are preserved. Function moves/renames are allowed; changing the body of
an otherwise unchanged declaration is not. This is a source preservation check,
not a numerical or ABI equivalence proof.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = "6a02e4b1"
SOURCE_ROOT = "src/xtb/native/src"
SOURCE_SUFFIXES = {".cu", ".cuh", ".h", ".hpp", ".cc", ".cpp", ".cxx"}
# These are exactly the ten generated headers included by the embedded CUDA
# backend. The density include is an additional gate: its arithmetic is included
# inside otherwise byte-identical kernel bodies.
GENERATORS = {
    "generated_gfn2_aes2_native.cuh": ("aes2_native", "--cuda-output"),
    "generated_gfn2_electronic_native.cuh": ("electronic_cuda", "--output"),
    "generated_gfn2_es2_native.hpp": ("es2_native", "--output"),
    "generated_gfn2_es3_native.cuh": ("es3_native", "--output"),
    "generated_gfn2_external_point_charge_force.hpp": (
        "external_point_charge_force",
        "--output",
    ),
    "generated_gfn2_h0_native.hpp": ("h0_native", "--output"),
    "generated_gfn2_pair_native.hpp": ("pair_native", "--output"),
    "generated_gfn2_scc_free_energy_native.hpp": ("scc_free_energy_native", "--output"),
    "generated_gfn2_sdq_cuda.cuh": ("sdq_native", "--cuda-output"),
    "generated_gfn2_spin_native.hpp": ("spin_native", "--output"),
    "generated_gfn2_density_contract.inc": ("density_cuda", "--output"),
}
_DIRECTIVE = re.compile(rb"(?m)^[ \t]*#[^\n]*(?:\n|$)")
_TOKEN = re.compile(rb"[A-Za-z_]\w*|[^\s]")
_RAW_STRING = re.compile(rb'(?:u8|u|U|L)?R"([^ ()\\\t\r\n]{0,16})\(')


class PreservationError(ValueError):
    """Incomplete inventory or a preservation invariant failed."""


def lexical_mask(source: bytes) -> bytes:
    """Blank C++ comments/literals, retaining byte offsets and newlines."""
    masked = bytearray(source)
    position = 0
    while position < len(source):
        end = position
        raw = _RAW_STRING.match(source, position)
        if source.startswith(b"//", position):
            end = source.find(b"\n", position)
            while end >= 0 and source[position:end].rstrip(b"\r").endswith(b"\\"):
                end = source.find(b"\n", end + 1)
            if end < 0:
                end = len(source)
        elif source.startswith(b"/*", position):
            end = source.find(b"*/", position + 2)
            if end < 0:
                raise PreservationError("unterminated block comment")
            end += 2
        elif raw:
            terminator = b")" + raw[1] + b'"'
            end = source.find(terminator, raw.end())
            if end < 0:
                raise PreservationError("unterminated raw string")
            end += len(terminator)
        elif source[position] in (34, 39):
            # C++ digit separators are not character literals.
            if (
                source[position] == 39
                and position > 0
                and position + 1 < len(source)
                and chr(source[position + 1]).isalnum()
                and re.search(rb"(?<![\w'])\d[\w.']*$", source[:position])
            ):
                position += 1
                continue
            quote = source[position]
            end = position + 1
            while end < len(source):
                if source[end] == 92:
                    end += 2
                elif source[end] == quote:
                    end += 1
                    break
                else:
                    end += 1
            else:
                raise PreservationError("unterminated quoted literal")
        if end > position:
            masked[position:end] = bytes(
                byte if byte in (10, 13) else 32 for byte in source[position:end]
            )
            position = end
        else:
            position += 1
    return bytes(masked)


def _directives(masked: bytes) -> list[tuple[int, int, bytes]]:
    result = []
    consumed = 0
    for match in _DIRECTIVE.finditer(masked):
        if match.start() < consumed:
            continue
        end = match.end()
        while masked[match.start() : end].rstrip(b"\r\n").endswith(b"\\"):
            following = masked.find(b"\n", end)
            end = len(masked) if following < 0 else following + 1
            if end == len(masked):
                break
        result.append((match.start(), end, masked[match.start() : end]))
        consumed = end
    return result


def qualifier_aliases(sources: Iterable[bytes]) -> dict[bytes, str]:
    """Recognize object-like CUDA annotation aliases, including shared headers."""
    aliases = {b"__global__": "kernel", b"__device__": "device"}
    definitions = []
    for source in sources:
        for _, _, directive in _directives(lexical_mask(source)):
            match = re.match(rb"\s*#\s*define\s+(\w+)[ \t]+(.*)", directive, re.DOTALL)
            if match:
                definitions.append((match[1], match[2]))
    changed = True
    while changed:
        changed = False
        for name, value in definitions:
            kinds = {
                aliases[token] for token in _TOKEN.findall(value) if token in aliases
            }
            if name not in aliases and kinds:
                aliases[name] = "kernel" if "kernel" in kinds else "device"
                changed = True
    return aliases


@dataclass(frozen=True)
class DeviceBody:
    path: str
    line: int
    name: str
    kind: str
    declaration: bytes
    body: bytes

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.body).hexdigest()

    @property
    def label(self) -> str:
        return f"{self.path}:{self.line} {self.name} sha256={self.digest}"


def device_bodies(
    source: bytes, path: str = "fixture.cu", aliases: Mapping[bytes, str] | None = None
) -> list[DeviceBody]:
    aliases = aliases or qualifier_aliases((source,))
    masked = bytearray(lexical_mask(source))
    # Definitions/declarations in macros are not adjacent ordinary functions.
    # Their expanded annotation aliases are handled explicitly above.
    for start, end, _ in _directives(bytes(masked)):
        masked[start:end] = b" " * (end - start)
    tokens = list(_TOKEN.finditer(masked))
    bodies = []
    index = 0
    while index < len(tokens):
        qualifier = tokens[index]
        if qualifier[0] not in aliases:
            index += 1
            continue
        kind = aliases[qualifier[0]]
        depth = 0
        name = None
        opening = None
        cursor = index + 1
        while cursor < len(tokens):
            token = tokens[cursor][0]
            if token in aliases and aliases[token] == "kernel":
                kind = "kernel"
            if token == b"(" and depth == 0 and name is None:
                previous = tokens[cursor - 1][0]
                if previous not in (
                    b"__launch_bounds__",
                    b"__attribute__",
                    b"decltype",
                    b"alignas",
                ):
                    name = previous.decode("ascii")
            if token in (b"(", b"["):
                depth += 1
            elif token in (b")", b"]"):
                depth -= 1
                if depth < 0:
                    raise PreservationError(f"unbalanced CUDA declaration in {path}")
            elif depth == 0 and token in (b";", b"="):
                break  # Prototype, device variable, or deleted/defaulted function.
            elif depth == 0 and token == b"{":
                if name is not None:
                    opening = cursor
                break
            cursor += 1
        if opening is None:
            if cursor == len(tokens):
                raise PreservationError(f"unfinished CUDA declaration in {path}")
            index = cursor + 1
            continue
        depth = 1
        cursor = opening + 1
        while cursor < len(tokens) and depth:
            token = tokens[cursor][0]
            depth += (token == b"{") - (token == b"}")
            cursor += 1
        if depth:
            raise PreservationError(f"unbalanced CUDA body in {path}: {name}")
        start, end = tokens[opening].start(), tokens[cursor - 1].end()
        declaration = b" ".join(_TOKEN.findall(masked[qualifier.start() : start]))
        bodies.append(
            DeviceBody(
                path,
                source.count(b"\n", 0, start) + 1,
                str(name),
                kind,
                declaration,
                source[start:end],
            )
        )
        index = cursor
    return bodies


def compare_bodies(before: Sequence[DeviceBody], after: Sequence[DeviceBody]) -> None:
    """Keep multiplicity and kind; don't let moved/renamed bodies hide edits."""
    groups = []
    for bodies in (before, after):
        grouped = defaultdict(list)
        for body in bodies:
            grouped[(body.path, body.declaration)].append(body)
        groups.append(grouped)
    old_groups, new_groups = groups
    errors = []
    for identity in old_groups.keys() & new_groups.keys():
        old, new = old_groups[identity], new_groups[identity]
        if Counter((body.kind, body.body) for body in old) != Counter(
            (body.kind, body.body) for body in new
        ):
            errors.append(
                "changed declaration body: " + old[0].label + " -> " + new[0].label
            )
    old_counts = Counter((body.kind, body.body) for body in before)
    new_counts = Counter((body.kind, body.body) for body in after)
    for label, difference, bodies in (
        ("removed/changed", old_counts - new_counts, before),
        ("added/changed", new_counts - old_counts, after),
    ):
        for key, count in difference.items():
            representative = next(
                body for body in bodies if (body.kind, body.body) == key
            )
            errors.append(f"{label} ({count} occurrence(s)): {representative.label}")
    if errors:
        raise PreservationError("\n".join(errors))


def _git(root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, check=False, timeout=60
    )
    if result.returncode:
        raise PreservationError(result.stderr.decode(errors="replace").strip())
    return result.stdout


def _base_sources(root: Path, base: str, scopes: Sequence[str]) -> dict[str, bytes]:
    paths = _git(root, "ls-tree", "-rz", "--name-only", base, "--", *scopes).split(
        b"\0"
    )
    selected = [
        os.fsdecode(path)
        for path in paths
        if Path(os.fsdecode(path)).suffix in SOURCE_SUFFIXES
    ]
    # Batch the local Git reads; never check out, reset, fetch, or modify an index.
    result = subprocess.run(
        ["git", "-C", str(root), "cat-file", "--batch"],
        input=b"".join(f"{base}:{path}\n".encode() for path in selected),
        capture_output=True,
        check=True,
        timeout=60,
    )
    stream = io.BytesIO(result.stdout)
    sources = {}
    for path in selected:
        metadata = stream.readline().split()
        if len(metadata) != 3 or metadata[1] != b"blob":
            raise PreservationError(f"cannot read base source: {path}")
        sources[path] = stream.read(int(metadata[2]))
        if stream.read(1) != b"\n":
            raise PreservationError(f"invalid Git blob framing: {path}")
    return sources


def _current_sources(root: Path, scopes: Sequence[str]) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for scope in scopes
        for path in sorted((root / scope).rglob("*"))
        if path.is_file() and path.suffix in SOURCE_SUFFIXES
    }


def _inventory(sources: Mapping[str, bytes]) -> list[DeviceBody]:
    aliases = qualifier_aliases(sources.values())
    return [
        body
        for path, source in sorted(sources.items())
        for body in device_bodies(source, path, aliases)
    ]


def _export_generators(root: Path, base: str, destination: Path) -> None:
    archive = _git(root, "archive", "--format=tar", base, "python", "tools")
    with tarfile.open(fileobj=io.BytesIO(archive)) as contents:
        for member in contents:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise PreservationError(f"unsafe archive path: {member.name}")
            if member.isfile():
                reader = contents.extractfile(member)
                if reader is None:
                    raise PreservationError(f"unreadable archive member: {member.name}")
                target = destination / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(reader.read())


def _generated_products(root: Path, destination: Path) -> dict[str, bytes]:
    destination.mkdir(parents=True, exist_ok=True)
    products = {}
    for name, (generator, flag) in sorted(GENERATORS.items()):
        output = destination / name
        result = subprocess.run(
            [
                sys.executable,
                str(root / "tools" / f"generate_gfn2_{generator}.py"),
                flag,
                str(output),
            ],
            cwd=root,
            env={
                **os.environ,
                "PYTHONPATH": str(root / "python") + os.pathsep + str(root),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": "0",
            },
            capture_output=True,
            check=False,
            timeout=180,
        )
        if result.returncode:
            raise PreservationError(
                f"generation failed for {name}:\n"
                + result.stderr.decode(errors="replace")
            )
        products[name] = output.read_bytes()
    return products


def _body_digest(bodies: Sequence[DeviceBody]) -> str:
    digest = hashlib.sha256()
    for kind, body in sorted((body.kind, body.body) for body in bodies):
        digest.update(kind.encode() + b"\0" + str(len(body)).encode() + b"\0" + body)
    return digest.hexdigest()


def preservation_report(
    root: Path = ROOT,
    base: str = DEFAULT_BASE,
    scopes: Sequence[str] = (SOURCE_ROOT,),
    *,
    generated: bool = True,
) -> dict[str, object]:
    commit = _git(root, "rev-parse", "--verify", base + "^{commit}").decode().strip()
    old_sources = _base_sources(root, commit, scopes)
    new_sources = _current_sources(root, scopes)
    before, after = _inventory(old_sources), _inventory(new_sources)
    if not before:
        raise PreservationError("base inventory contains no CUDA function bodies")
    compare_bodies(before, after)
    report: dict[str, object] = {
        "base": commit,
        "source_roots": list(scopes),
        "source_files": {"base": len(old_sources), "working": len(new_sources)},
        "function_bodies": {
            "base": len(before),
            "working": len(after),
            "kernels": sum(body.kind == "kernel" for body in after),
            "device_helpers": sum(body.kind == "device" for body in after),
            "base_sha256": _body_digest(before),
            "working_sha256": _body_digest(after),
        },
    }
    if generated:
        # Fail closed when the runtime gains/removes a generated GFN include.
        include_pattern = rb'#\s*include\s*"(generated_gfn2_[^"\n]+)"'
        for label, sources in (("base", old_sources), ("working", new_sources)):
            includes = {
                name.decode()
                for path, source in sources.items()
                if path.startswith(SOURCE_ROOT + "/backends/cuda/")
                for name in re.findall(include_pattern, source)
            }
            if includes != set(GENERATORS):
                raise PreservationError(
                    f"{label} generated include inventory changed: {sorted(includes ^ set(GENERATORS))}"
                )
        with tempfile.TemporaryDirectory(prefix="gfn-byte-preservation-") as temporary:
            scratch = Path(temporary)
            export = scratch / "base"
            _export_generators(root, commit, export)
            old = _generated_products(export, scratch / "old-products")
            new = _generated_products(root, scratch / "new-products")
        changed = [name for name in old if old[name] != new[name]]
        if changed:
            raise PreservationError("generated bytes changed: " + ", ".join(changed))
        report["generated_headers"] = sum(Path(name).suffix != ".inc" for name in old)
        report["generated_include_fragments"] = sum(
            Path(name).suffix == ".inc" for name in old
        )
        report["generated_products"] = {
            name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            for name, content in sorted(old.items())
        }
    report["status"] = "byte-identical"
    return report


def test_lexical_body_boundaries_preserve_original_bytes() -> None:
    body = rb"""{
        // } __global__ void decoy() { \
        } still a comment
        /* { } */ const char* value = "}\\\"{";
        const char brace = '}';
        const auto wide = L'}'; const auto utf8 = u8'{';
        const char* raw = u8R"tag({ \" } //)tag";
        int count = 1'000; if (count) { count++; }
    }"""
    source = b"// __device__ void absent() {}\n__global__ void kernel() " + body
    found = device_bodies(source)
    assert len(found) == 1
    assert found[0].name == "kernel"
    assert found[0].body == body


def test_prototypes_variables_and_annotation_macros() -> None:
    source = b"""
#define CUDA_HD __host__ __device__
#define CUDA_ALIAS CUDA_HD
extern __device__ int count;
extern __device__ Thing value{};
__device__ void prototype(int value);
CUDA_ALIAS inline int helper(int value) { return value; }
__global__ __launch_bounds__(128) void kernel(int (*fn)(int)) { fn(0); }
"""
    found = device_bodies(source)
    assert [(body.name, body.kind) for body in found] == [
        ("helper", "device"),
        ("kernel", "kernel"),
    ]


def test_move_and_rename_preserve_duplicate_bodies() -> None:
    before = device_bodies(
        b"__device__ int first() { return 1; }\n__device__ int second() { return 1; }",
        "old.cu",
    )
    after = device_bodies(
        b"__device__ int renamed() { return 1; }\n__device__ int other() { return 1; }",
        "new.cuh",
    )
    compare_bodies(before, after)


def test_duplicate_deletion_is_detected() -> None:
    import pytest

    before = device_bodies(b"__device__ int a() {} __device__ int b() {}")
    with pytest.raises(PreservationError, match="removed/changed"):
        compare_bodies(before, before[:1])


def test_whitespace_and_comment_edits_inside_body_are_detected() -> None:
    import pytest

    before = device_bodies(b"__global__ void work() { /* original */ }\r\n")
    for source in (
        b"__global__ void work() {/* original */ }",
        b"__global__ void work() { /* edited */ }",
    ):
        with pytest.raises(PreservationError, match="changed declaration body"):
            compare_bodies(before, device_bodies(source))


def test_swapped_bodies_do_not_pass_as_renames() -> None:
    import pytest

    before = device_bodies(
        b"__device__ int a() { return 1; } __device__ int b() { return 2; }"
    )
    after = device_bodies(
        b"__device__ int a() { return 2; } __device__ int b() { return 1; }"
    )
    with pytest.raises(PreservationError, match="changed declaration body"):
        compare_bodies(before, after)


def test_unterminated_constructs_fail_closed() -> None:
    import pytest

    for source in (
        b"/*",
        b'__device__ void f() { "',
        b'R"tag(unclosed',
        b"__global__ void f() {",
    ):
        with pytest.raises(PreservationError):
            device_bodies(source)


def test_gfn_cuda_bytes_against_requested_local_base() -> None:
    import pytest

    base = os.environ.get("GFN_CUDA_PRESERVATION_BASE")
    if not base:
        pytest.skip("set GFN_CUDA_PRESERVATION_BASE to run the local historical gate")
    report = preservation_report(base=base)
    assert report["status"] == "byte-identical"
    assert report["generated_headers"] == 10


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--source-root", action="append", help="repeat for broader source scopes"
    )
    parser.add_argument(
        "--bodies-only", action="store_true", help="omit generated-product regeneration"
    )
    args = parser.parse_args(argv)
    try:
        report = preservation_report(
            args.root.resolve(),
            args.base,
            args.source_root or (SOURCE_ROOT,),
            generated=not args.bodies_only,
        )
    except (PreservationError, OSError, subprocess.SubprocessError) as error:
        print(f"CUDA byte preservation FAILED: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
