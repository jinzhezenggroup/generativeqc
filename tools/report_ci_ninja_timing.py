"""Report bounded completed-command timings from one fresh CI Ninja log.

The owning workflow checks freshness before its existing configure/build step.
Ninja logs have no invocation IDs: historical or compacted logs cannot establish
an invocation boundary, and are deliberately unsupported. No compiler, source,
command line, environment dump, or raw log is executed or published here.
"""

from __future__ import annotations

import argparse
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path

MAX_BYTES = 16 * 1024 * 1024
MAX_RECORDS = 50_000
MAX_LINE = 4096
MAX_TIME_MS = 24 * 60 * 60 * 1000
MAX_SUMMARY_BYTES = 24 * 1024
TOP_COUNT = 20
SAFE_OUTPUT = re.compile(r"[A-Za-z0-9_-]+(?:[./][A-Za-z0-9_-]+)*")


@dataclass
class Edge:
    start: int
    end: int
    outputs: list[str] = field(default_factory=list)
    hidden_outputs: int = 0

    @property
    def duration(self) -> int:
        return self.end - self.start


def display_output(output: str) -> str | None:
    """Allow only bounded build-relative native outputs; never normalize paths."""
    if len(output) > 180 or not SAFE_OUTPUT.fullmatch(output):
        return None
    if output == "generativeqc":
        return output
    if output.startswith("generated/"):
        return output
    if output.startswith("CMakeFiles/") and (
        output.endswith((".o", ".obj")) or output == "CMakeFiles/cmake.verify_globs"
    ):
        return output
    return None


def parse_log(raw: bytes) -> list[Edge]:
    """Validate the entire fresh log before emitting any aggregate or filename."""
    if len(raw) > MAX_BYTES:
        raise ValueError("input size limit exceeded")
    if raw.count(b"\n") > MAX_RECORDS + 1:
        raise ValueError("record limit exceeded")
    lines = raw.split(b"\n")
    if lines[0] not in (b"# ninja log v5", b"# ninja log v6", b"# ninja log v7"):
        raise ValueError("unsupported log version")
    if lines[-1] != b"":
        raise ValueError("incomplete final record")
    records = lines[1:-1]
    if len(records) > MAX_RECORDS:
        raise ValueError("record limit exceeded")
    edges: list[Edge] = []
    seen_outputs: set[bytes] = set()
    seen_keys: set[tuple[int, int, bytes]] = set()
    previous_key = None
    previous_end = -1
    for line in records:
        if len(line) > MAX_LINE:
            raise ValueError("record size limit exceeded")
        fields = line.split(b"\t")
        if len(fields) != 5:
            raise ValueError("malformed record")
        start_raw, end_raw, mtime, output, command_hash = fields
        if (
            not re.fullmatch(rb"[0-9]{1,9}", start_raw)
            or not re.fullmatch(rb"[0-9]{1,9}", end_raw)
            or not re.fullmatch(rb"-?[0-9]{1,20}", mtime)
            or not re.fullmatch(rb"[0-9a-fA-F]{1,16}", command_hash)
            or not output
            or any(character < 32 or character == 127 for character in output)
        ):
            raise ValueError("malformed record")
        start, end = int(start_raw), int(end_raw)
        if not 0 <= start <= end <= MAX_TIME_MS:
            raise ValueError("invalid or excessive timestamp")
        if end < previous_end or output in seen_outputs:
            raise ValueError("repeated invocation or reordered log is ambiguous")
        seen_outputs.add(output)
        previous_end = end
        key = (start, end, command_hash.lower())
        if key != previous_key:
            if key in seen_keys:
                raise ValueError("noncontiguous command timing group is ambiguous")
            seen_keys.add(key)
            edges.append(Edge(start, end))
            previous_key = key
        # Ninja writes every output of a command with identical timing/hash.
        # Absolute aliases are counted but never printed, including error paths.
        name = display_output(output.decode("utf-8", errors="replace"))
        if name is None:
            edges[-1].hidden_outputs += 1
        else:
            edges[-1].outputs.append(name)
    return edges


def category(edge: Edge) -> str:
    if any(name.endswith(".cu.o") for name in edge.outputs):
        return "CUDA object"
    if any(name.endswith((".o", ".obj")) for name in edge.outputs):
        return "Other object"
    if "generativeqc" in edge.outputs:
        return "CLI executable"
    if any(name.startswith("generated/") for name in edge.outputs):
        return "Generated output"
    return "Other / redacted output"


def render(edges: list[Edge]) -> str:
    """Keep cumulative parallel durations distinct from the timestamp envelope."""
    rows = [
        "### Native CLI Ninja completed-command timings",
        "",
        "Fresh-log boundary checked before the existing CLI configure/build step.",
        "Only completed, logged commands are represented; failed/interrupted commands are absent.",
        "Contiguous records with equal command hash/start/end form one timing group,",
        "deduplicating multi-output commands (indistinguishable equal-time/hash commands coalesce).",
        "",
        f"Recorded command timing groups: {len(edges)}",
        f"Output records: {sum(len(edge.outputs) + edge.hidden_outputs for edge in edges)}",
        f"Redacted output names: {sum(edge.hidden_outputs for edge in edges)}",
    ]
    if edges:
        total = sum(edge.duration for edge in edges) / 1000
        span = (
            max(edge.end for edge in edges) - min(edge.start for edge in edges)
        ) / 1000
        rows += [
            f"Cumulative command duration (parallel work): {total:.3f} s",
            f"Recorded timestamp span (first start to last completion): {span:.3f} s",
        ]
    else:
        rows.append("No completed command records; timestamp span unavailable.")
    rows += [
        "",
        "Cumulative duration is not wall time. The timestamp span excludes configure/setup",
        "and unlogged work; neither metric establishes the dependency critical path.",
        "Use the GitHub build-step duration for elapsed configure/build wall time.",
        "Ninja has no invocation IDs; freshness comes from the workflow, not a timestamp heuristic.",
        "",
        "| Output category | Groups | Cumulative seconds | Longest seconds |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label in sorted({category(edge) for edge in edges}):
        durations = [edge.duration for edge in edges if category(edge) == label]
        rows.append(
            f"| {label} | {len(durations)} | {sum(durations) / 1000:.3f} | {max(durations) / 1000:.3f} |"
        )
    rows += [
        "",
        f"Longest {TOP_COUNT} recorded command timing groups (one allowlisted relative output per group):",
        "",
        "| Seconds | Start s | End s | Representative output | Additional outputs |",
        "| ---: | ---: | ---: | --- | ---: |",
    ]
    for edge in sorted(edges, key=lambda item: item.duration, reverse=True)[:TOP_COUNT]:
        name = edge.outputs[0] if edge.outputs else "[redacted]"
        extra = len(edge.outputs) + edge.hidden_outputs - 1
        rows.append(
            f"| {edge.duration / 1000:.3f} | {edge.start / 1000:.3f} | {edge.end / 1000:.3f} | {name} | {extra} |"
        )
    summary = "\n".join(rows) + "\n"
    if len(summary.encode()) > MAX_SUMMARY_BYTES:
        raise ValueError("summary size limit exceeded")
    return summary


def log_path(build_dir: Path) -> Path:
    path = build_dir / ".ninja_log"
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("symlinked log or build directory")
    return path


def read_log(path: Path) -> bytes:
    # O_NONBLOCK prevents a substituted FIFO from blocking before fstat.
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_BYTES:
            raise ValueError("invalid log file type or size")
        raw = stream.read(MAX_BYTES + 1)
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("log changed while reading")
        return raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--check-fresh", action="store_true")
    parser.add_argument("--fresh-log", default="false")
    args = parser.parse_args()
    if args.check_fresh:
        fresh = False
        try:
            path = log_path(args.build_dir)
            fresh = not path.exists()
        except (OSError, ValueError):
            pass
        destination = os.environ.get("GITHUB_OUTPUT")
        if destination:
            try:
                with Path(destination).open("a", encoding="utf-8") as stream:
                    stream.write(f"fresh={str(fresh).lower()}\n")
            except OSError:
                print("Ninja freshness boundary unavailable.")
        return

    try:
        if args.fresh_log != "true":
            raise ValueError(
                "no verified fresh-log boundary; historical invocations are unsupported"
            )
        summary = render(parse_log(read_log(log_path(args.build_dir))))
    except (OSError, ValueError):
        # Never echo exception text: filesystem errors may expose absolute paths.
        summary = (
            "### Native CLI Ninja completed-command timings\n\n"
            "Timing unavailable: missing/unverified fresh log, invalid/ambiguous records, "
            "or diagnostic resource limit. Build/test results remain authoritative.\n"
        )
    print(summary)
    destination = os.environ.get("GITHUB_STEP_SUMMARY")
    if destination:
        try:
            with Path(destination).open("a", encoding="utf-8") as stream:
                stream.write(summary)
        except OSError:
            print(
                "Ninja step summary unavailable; timing report remains in the job log."
            )


if __name__ == "__main__":
    main()
