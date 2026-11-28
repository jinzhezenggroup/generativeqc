"""Rebuild only the trace-enabled CLI translation unit against the pinned core."""

import argparse
import hashlib
import shlex
import shutil
import subprocess
from pathlib import Path


def run(args: list[str], cwd: Path) -> None:
    print(shlex.join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("build", type=Path)
    args = parser.parse_args()
    pin, work = args.baseline.resolve(), args.build.resolve()
    cli_dir = pin / "c/build"
    work.mkdir()
    subprocess.run(
        ["sha256sum", "-c", "build-receipts/binaries.sha256"],
        cwd=pin / "c",
        check=True,
    )
    files = (
        "src/cli/geometry_command.cpp",
        "include/generativeqc/geometry_optimizer.hpp",
    )
    for name in files:
        dest = work / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((pin / "source" / name).read_bytes())
    subprocess.run(
        [
            "patch",
            "--batch",
            "-p1",
            "-i",
            str(Path(__file__).with_name("optimizer-trace.patch").resolve()),
        ],
        cwd=work,
        check=True,
    )
    commands = (pin / "c/build-receipts/compiler-commands.txt").read_text().splitlines()
    target = "CMakeFiles/generativeqc_cli.dir/src/cli/geometry_command.cpp.o"
    (compile_line,) = [c for c in commands if f"-o {target} " in c]
    (link_line,) = [c for c in commands if " -o generativeqc " in c and target in c]
    compile_args = shlex.split(compile_line)
    recorded_launcher = None
    if Path(compile_args[0]).name in {"sccache", "ccache"}:
        recorded_launcher = shutil.which(compile_args.pop(0))
    launcher = recorded_launcher or shutil.which("sccache") or shutil.which("ccache")
    if launcher is None:
        raise RuntimeError("CLI recompilation requires sccache or ccache")
    run([launcher, "--version"], cli_dir)
    compile_args[1:1] = ["-I", str(work / "include")]
    compile_args.insert(0, launcher)
    obj = work / "geometry_command.cpp.o"
    for option, value in (
        ("-o", obj),
        ("-MF", obj.with_suffix(".d")),
        ("-c", work / files[0]),
    ):
        compile_args[compile_args.index(option) + 1] = str(value)
    link_args = shlex.split(link_line.split(" && ")[1])
    link_args[link_args.index(target)] = str(obj)
    link_args[link_args.index("-o") + 1] = str(work / "generativeqc")
    run(compile_args, cli_dir)
    run(link_args, cli_dir)
    for path in (work / "generativeqc", cli_dir / "libgenerativeqc.so"):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        print(digest.hexdigest(), path.name)


if __name__ == "__main__":
    main()
