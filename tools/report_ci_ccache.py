"""Summarize observed ccache compilation results without treating restore as a hit.

This is CI reporting only: neither a low hit rate nor missing ccache hides an
actual compiler/test failure. Counters are reset by the owning workflow before
its build (a restored cache can still have zero actual compiler hits).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CacheStats:
    cacheable: int
    calls: int
    hits: int
    misses: int
    uncacheable: int


def _count(raw: str, name: str) -> tuple[int, int | None] | None:
    match = re.search(
        rf"^\s*{re.escape(name)}:\s*(\d+)(?:\s*/\s*(\d+))?",
        raw,
        flags=re.MULTILINE,
    )
    if match is None:
        return None
    return int(match[1]), int(match[2]) if match[2] is not None else None


def parse_stats(raw: str) -> CacheStats:
    """Read ccache 4.x human-readable counters, including zero-call builds."""
    cacheable = _count(raw, "Cacheable calls")
    hits = _count(raw, "Hits")
    misses = _count(raw, "Misses")
    uncacheable = _count(raw, "Uncacheable calls")
    if cacheable is None or hits is None or misses is None:
        raise ValueError("ccache did not report cacheable calls, hits and misses")

    count, total = cacheable
    hit_count, _ = hits
    miss_count, _ = misses
    uncacheable_count = uncacheable[0] if uncacheable else 0

    if hit_count + miss_count != count:
        raise ValueError("ccache hit and miss totals disagree")
    if total is None:
        total = count + uncacheable_count
    if count + uncacheable_count != total:
        raise ValueError("ccache cacheable and uncacheable totals disagree")

    return CacheStats(count, total, hit_count, miss_count, uncacheable_count)


def render_summary(label: str, stats: CacheStats | None, detail: str = "") -> str:
    """Render GitHub Actions step-summary Markdown, not a pass/fail gate."""
    label = label.replace("|", r"\|").replace("\n", " ")
    header = f"### Compiler cache: {label}\n\n"
    if stats is None:
        return header + f"ccache statistics unavailable: {detail}\n\n"
    hit_rate = (
        f"{100 * stats.hits / stats.cacheable:.1f}%"
        if stats.cacheable
        else "n/a (no cacheable calls)"
    )
    cacheable_fraction = (
        f"{100 * stats.cacheable / stats.calls:.1f}%"
        if stats.calls
        else "n/a (no compiler calls)"
    )
    return (
        header
        + "| Metric | Observed value |\n| --- | ---: |\n"
        + f"| Compiler calls | {stats.calls} |\n"
        + f"| Cacheable calls | {stats.cacheable} ({cacheable_fraction}) |\n"
        + f"| Cache hits | {stats.hits} ({hit_rate}) |\n"
        + f"| Cache misses | {stats.misses} |\n"
        + f"| Uncacheable calls | {stats.uncacheable} |\n\n"
        + "A restored Actions cache does not guarantee compiler-cache hits; "
        + "changed inputs still require compilation. This is not an "
        + "end-to-end CI time saving measurement.\n\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    binary = shutil.which("ccache")
    stats: CacheStats | None = None
    detail = "executable not installed"
    if binary:
        try:
            result = subprocess.run(
                [binary, "--show-stats"],
                check=True,
                text=True,
                capture_output=True,
                timeout=15,
            )
            stats = parse_stats(result.stdout)
            detail = ""
        except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            detail = str(exc)

    summary = render_summary(args.label, stats, detail=detail)
    destination = os.environ.get("GITHUB_STEP_SUMMARY")
    if destination:
        with Path(destination).open("a", encoding="utf-8") as stream:
            stream.write(summary)
    print(summary)


if __name__ == "__main__":
    main()
