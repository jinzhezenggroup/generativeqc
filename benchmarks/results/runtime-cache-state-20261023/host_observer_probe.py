"""Actual CPU compile/link, artifact replay and scoped observer qualification."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

from benchmarks.runtime_cache_state import Observer, cache_environment, new_cache, save


def main() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("requires real owned CPU Slurm allocation")
    root = Path("/data/srxu/projects/gqc-runtime-cache-state-20261023")
    probe = root / "host-observer-probe"
    probe.mkdir(exist_ok=False)
    cache = probe / "cache"
    new_cache(cache)
    environment = cache_environment(
        cache,
        {
            "library": str(probe / "unused-no-native-sdk.so"),
            "python_paths": [str(root / "source/python")],
        },
        dict(os.environ),
    )
    environment["PATH"] = str(root / "runtime-env/bin") + ":/usr/bin:/bin"
    os.environ.clear()
    os.environ.update(environment)
    from generativeqc_compiler.common import native_runtime
    from generativeqc_compiler.common.compiler_cache import resolve_compiler_cache
    from generativeqc_compiler.common.cpp_adapter import CppCompilerAdapter

    launcher = resolve_compiler_cache()
    source = probe / "probe.cpp"
    source.write_text('extern "C" int evaluate(int x) { return x*x - 3*x + 7; }\n')
    compiler = CppCompilerAdapter(Path("/usr/bin/g++"), compile_timeout=60)
    observations = []
    for phase in ("artifact-miss", "artifact-replay", "compiler-cache-reuse"):
        observer = Observer()
        with observer:
            if phase == "compiler-cache-reuse":
                library = probe / "independent.so"
                outcome = compiler.compile_shared(
                    source, library, options=("-ffp-contract=off",)
                )
                if outcome.returncode:
                    raise RuntimeError(outcome.stderr)
            else:
                artifact = native_runtime.compile_runtime(
                    compiler, cache / "artifact", source
                )
                library = artifact.library
        native = ctypes.CDLL(str(library))
        native.evaluate.argtypes = [ctypes.c_int]
        native.evaluate.restype = ctypes.c_int
        values = [native.evaluate(x) for x in (-3, 0, 4)]
        if values != [25, 7, 11]:
            raise ValueError("compiled public ABI result differs")
        receipt = observer.receipt()
        if phase == "artifact-replay" and receipt["cached_driver_invocations"] != 0:
            raise ValueError("artifact replay unexpectedly invoked compiler")
        if phase != "artifact-replay" and receipt["cached_driver_invocations"] < 1:
            raise ValueError("compiler observer missed actual invocation")
        for event in receipt["events"]:
            if (
                event["kind"] == "compiler-process"
                and event["phase"] != "probe"
                and any(
                    (event.get("stats_" + when) or {}).get("returncode") != 0
                    for when in ("before", "after")
                )
            ):
                raise ValueError("actual private server scoped stats missing")
        observations.append({"phase": phase, "values": values, "observation": receipt})
        save(
            probe / "observations.json",
            {
                "scope": "CPU GCC C++ compile/link, artifact replay and compiler cache; no CUDA/E+F qualification",
                "launcher": {"path": str(launcher.path), "version": launcher.version},
                "phases": observations,
            },
        )
    save(
        probe / "outcome.json",
        {"host_observer": "PASS", "gpu": "NOT_RUN", "endpoint": "NOT_RUN"},
    )


if __name__ == "__main__":
    main()
