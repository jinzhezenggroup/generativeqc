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


def render_build_context(build_dir: Path) -> str:
    """Read selected configured identities without executing CMake or a compiler."""
    lines = [f"Build context: {build_dir}"]
    try:
        cache = (build_dir / "CMakeCache.txt").read_text(encoding="utf-8")
        for key in (
            "CMAKE_BUILD_TYPE",
            "GENERATIVEQC_CPU_LINALG_PROVIDER",
            "GENERATIVEQC_CUDA_FAST_COMPILE",
            "GENERATIVEQC_MONOLITHIC_CLI",
            "GENERATIVEQC_RESOLVED_COMPILER_CACHE",
        ):
            match = re.search(rf"^{key}:[^=]+=(.*)$", cache, re.MULTILINE)
            lines.append(f"{key}={match[1] if match else 'unavailable'}")
    except (OSError, UnicodeError) as exc:
        lines.append(f"CMake cache unavailable: {exc}")

    # 'auto' is a request, not the selected provider. Inspect only the emitted
    # provider definition; never dump the complete cache or build environment.
    try:
        ninja = (build_dir / "build.ninja").read_text(encoding="utf-8")
        providers = set(re.findall(r"-DGENERATIVEQC_HAS_OPENBLAS=([01])\b", ninja))
        provider = {"0": "scalar", "1": "OpenBLAS"}
        selected = (
            provider[next(iter(providers))]
            if len(providers) == 1
            else "unavailable or mixed"
        )
        lines.append(f"Emitted CPU linear algebra provider: {selected}")
    except (OSError, UnicodeError) as exc:
        lines.append(f"Emitted CPU linear algebra provider unavailable: {exc}")

    for language in ("CXX", "CUDA"):
        records = sorted(
            (build_dir / "CMakeFiles").glob(f"*/CMake{language}Compiler.cmake")
        )
        if not records:
            lines.append(f"Configured {language} compiler identity: unavailable")
        for record in records:
            try:
                contents = record.read_text(encoding="utf-8")
                lines.append(
                    f"Configured {language} compiler identity ({record.parent.name}):"
                )
                for suffix in (
                    "COMPILER",
                    "COMPILER_ID",
                    "COMPILER_VERSION",
                    "HOST_COMPILER",
                ):
                    key = f"CMAKE_{language}_{suffix}"
                    match = re.search(
                        rf'^set\({key} "([^"\n]*)"\)$', contents, re.MULTILINE
                    )
                    if match:
                        lines.append(f"  {key}={match[1]}")
            except (OSError, UnicodeError) as exc:
                lines.append(
                    f"Configured {language} compiler identity unavailable: {exc}"
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--build-dir", type=Path)
    args = parser.parse_args()

    binary = shutil.which("ccache")
    stats: CacheStats | None = None
    detail = "executable not installed"
    if binary:
        try:
            result = subprocess.run(
                [binary, "--show-stats", *(["--verbose"] if args.verbose else [])],
                check=True,
                text=True,
                capture_output=True,
                timeout=15,
            )
            if args.verbose:
                print(
                    f"Compiler cache detail: {args.label}\n{result.stdout}", flush=True
                )
            stats = parse_stats(result.stdout)
            detail = ""
        except (
            OSError,
            ValueError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as exc:
            detail = str(exc)

    summary = render_summary(args.label, stats, detail=detail)
    if args.build_dir is not None:
        context = render_build_context(args.build_dir)
        print(f"Compiler build context: {args.label}\n{context}", flush=True)
        summary += "Configured build context:\n\n```text\n" + context + "```\n\n"
    destination = os.environ.get("GITHUB_STEP_SUMMARY")
    if destination:
        try:
            with Path(destination).open("a", encoding="utf-8") as stream:
                stream.write(summary)
        except OSError as exc:
            print(f"::warning::Unable to append compiler-cache summary: {exc}")
    print(summary)


if __name__ == "__main__":
    main()
