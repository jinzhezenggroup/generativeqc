"""Rebuild only the trace-enabled CLI translation unit against the pinned core."""

import argparse
import hashlib
import shlex
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
    if Path(compile_args[0]).name == "sccache":
        # This single translation unit runs in a unique directory and has no warm object.
        compile_args.pop(0)
    compile_args[1:1] = ["-I", str(work / "include")]
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
        with path.open("rb") as stream:
            print(hashlib.file_digest(stream, "sha256").hexdigest(), path.name)


if __name__ == "__main__":
    main()
