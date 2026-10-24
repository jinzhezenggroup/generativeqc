"""Execute the capacity wrapper with offline scheduler and artifact-provider doubles.

Only the positive controls substitute artifact hashes; real sha256sum must refuse
unrelated binaries before any scheduler call. No GPU or Slurm work is launched.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PUBLICATION = ROOT / "benchmarks/results/wb97mv-hessian-capacity-20261017"
DRIVER_REVISION = "9abfe3b0b3560576da8c3eef23fb24772b4e76e6"


def _recorded_identity() -> dict[str, str]:
    samples = json.loads(
        lzma.decompress((PUBLICATION / "samples.json.xz").read_bytes())
    )
    records = [
        json.loads(value["text"])["provenance"]
        for path, value in samples["raw_files"].items()
        if path.endswith(("raw-9977/capacity.json", "raw-9979/capacity.json"))
    ]
    assert len(records) == 2
    identity = {
        key: records[0][key]
        for key in ("source_commit", "cli_launcher_sha256", "native_library_sha256")
    }
    assert all(
        all(record[key] == value for key, value in identity.items())
        for record in records
    )
    return identity


def _executable(path: Path, text: str) -> None:
    path.write_text(f"#!{sys.executable}\n" + text)
    path.chmod(0o755)


@pytest.fixture
def replay(tmp_path: Path) -> tuple[Path, dict[str, str], dict[str, str]]:
    """Isolate archive/authentication boundaries without editing the wrapper."""
    root = tmp_path / "checkout with spaces"
    publication = root / PUBLICATION.relative_to(ROOT)
    publication.mkdir(parents=True)
    shutil.copyfile(PUBLICATION / "reproduce.sh", publication / "reproduce.sh")
    inputs = tmp_path / "native inputs"
    inputs.mkdir()
    reference_python = inputs / "reference-python"
    reference_python.symlink_to(sys.executable)
    cli = inputs / "cli"
    library = inputs / "library"
    cli.write_bytes(b"recorded test CLI\n")
    cli.chmod(0o755)
    library.write_bytes(b"recorded test library\n")
    binary_digests = {
        "generativeqc": hashlib.sha256(cli.read_bytes()).hexdigest(),
        "libgenerativeqc.so.0.1.0": hashlib.sha256(library.read_bytes()).hexdigest(),
    }
    identity = _recorded_identity()
    expected_hashes = {
        "generativeqc": identity["cli_launcher_sha256"],
        "libgenerativeqc.so.0.1.0": identity["native_library_sha256"],
    }
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _executable(bin_dir / "cache", "print('verified test cache')\n")
    _executable(
        bin_dir / "git",
        f"""import io, json, os, sys, tarfile
from pathlib import Path
args = sys.argv[1:]
assert args == ['-C', {str(root)!r}, 'archive', {DRIVER_REVISION!r}], args
with open(os.environ['ARCHIVE_LOG'], 'w') as out:
    json.dump(args, out)
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|') as archive:
    for name in ['benchmarks/compare_wb97mv_hessian_gpu4pyscf.py',
                 'manifests/public_methods.json', 'python/fixture', 'tests/python/fixture']:
        data = b'fixture-bearing archived driver\\n'
        member = tarfile.TarInfo(name)
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
""",
    )
    _executable(
        bin_dir / "sha256sum",
        f"""import hashlib, sys
from pathlib import Path
expected = {expected_hashes!r}
fixture = {binary_digests!r}
assert sys.argv[1:] == ['--check']
seen = set()
for line in sys.stdin:
    digest, name = line.rstrip('\\n').split('  ', 1)
    path = Path(name)
    if digest != expected.get(path.name) or hashlib.sha256(path.read_bytes()).hexdigest() != fixture.get(path.name):
        sys.exit('SHA256 mismatch: ' + name)
    seen.add(path.name)
assert seen == set(expected), seen
""",
    )
    _executable(
        bin_dir / "srun",
        """import hashlib, json, os, sys
from pathlib import Path
args = sys.argv[1:]
row = {'args': args, 'cwd': os.getcwd(), 'env': {key: os.environ.get(key) for key in
       ['GENERATIVEQC_LIBRARY', 'LD_LIBRARY_PATH', 'PYTHONPATH', 'CUDA_VISIBLE_DEVICES',
        'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'CUDA_DISABLE_PTX_JIT']}}
with open(os.environ['SCHEDULER_LOG'], 'a') as log:
    log.write(json.dumps(row) + '\\n')
assert Path('benchmarks/compare_wb97mv_hessian_gpu4pyscf.py').read_text() == 'fixture-bearing archived driver\\n'
cli = Path(args[args.index('--cli') + 1])
library = Path(os.environ['GENERATIVEQC_LIBRARY'])
assert cli.read_bytes() == b'recorded test CLI\\n'
assert library.read_bytes() == b'recorded test library\\n'
assert Path(os.environ['LD_LIBRARY_PATH'].split(':')[0], 'libgenerativeqc.so.0').resolve() == library
case = args[args.index('--cases') + 1]
failed = case == os.environ.get('FAIL_CASE')
output = Path(args[args.index('--output') + 1])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({'status': 'failed' if failed else 'completed'}))
sys.exit(17 if failed else 0)
""",
    )
    env = {
        **os.environ,
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "GENERATIVEQC_REFERENCE_PYTHON": str(reference_python),
        "GENERATIVEQC_CLI": str(cli),
        "GENERATIVEQC_LIBRARY": str(library),
        "GENERATIVEQC_COMPILER_CACHE": "cache",
        "CUDA_VISIBLE_DEVICES": "assigned-by-slurm",
        "ARCHIVE_LOG": str(tmp_path / "archive.json"),
        "SCHEDULER_LOG": str(tmp_path / "scheduler.jsonl"),
    }
    return root, env, identity


def _run(
    root: Path, env: dict[str, str], cwd: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(root / PUBLICATION.relative_to(ROOT) / "reproduce.sh")],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize("invocation", ["repository", "publication", "unrelated"])
def test_capacity_replay_is_cwd_independent_and_binds_native_artifacts(
    replay: tuple[Path, dict[str, str], dict[str, str]], invocation: str
) -> None:
    root, env, identity = replay
    cwd = {
        "repository": root,
        "publication": root / PUBLICATION.relative_to(ROOT),
        "unrelated": root.parent,
    }[invocation]
    # Resolve caller-relative artifact paths before changing into the archive.
    for key in ("GENERATIVEQC_CLI", "GENERATIVEQC_LIBRARY"):
        env[key] = os.path.relpath(env[key], cwd)
    result = _run(root, env, cwd)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [
        json.loads(line) for line in Path(env["SCHEDULER_LOG"]).read_text().splitlines()
    ]
    assert len(rows) == 2
    for row, case, basis, output in zip(
        rows,
        ("water-tetramer", "water-dimer"),
        ("sto-3g", "def2-svp"),
        ("capacity-tetramer.json", "capacity-def2svp.json"),
        strict=True,
    ):
        args = row["args"]
        for key, value in {
            "--cases": case,
            "--reference-python": env["GENERATIVEQC_REFERENCE_PYTHON"],
            "--basis": basis,
            "--source-commit": identity["source_commit"],
            "--repeats": "1",
            "--timeout-seconds": "180",
            "--gpu-memory-limit-mib": "24576",
            "--output": str(root / ".artifacts" / output),
        }.items():
            assert args[args.index(key) + 1] == value
        assert "--time=00:12:00" in args and "--gres=gpu:5090:1" in args
        assert row["env"]["CUDA_VISIBLE_DEVICES"] == "assigned-by-slurm"
        assert row["env"]["PYTHONPATH"].split(":") == [
            row["cwd"] + "/python",
            row["cwd"] + "/tests/python",
            row["cwd"],
        ]
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "CUDA_DISABLE_PTX_JIT",
        ):
            assert row["env"][key] == "1"
        assert Path(row["cwd"]).is_relative_to(root / ".artifacts"), (
            "Slurm staging must share the checkout filesystem, not login-local /tmp"
        )
        assert not Path(row["cwd"]).exists(), "temporary archive was not cleaned up"
    assert (
        identity["source_commit"] in result.stdout and DRIVER_REVISION in result.stdout
    )


@pytest.mark.parametrize(
    "bad_input", ["cli", "library", "missing_library", "unrelated_binaries"]
)
def test_capacity_replay_refuses_unbound_binaries_before_slurm(
    replay: tuple[Path, dict[str, str], dict[str, str]], bad_input: str
) -> None:
    root, env, _ = replay
    if bad_input == "missing_library":
        del env["GENERATIVEQC_LIBRARY"]
    elif bad_input == "unrelated_binaries":
        # Exercise the real system hash verifier, not the positive-control double.
        (Path(env["PATH"].split(os.pathsep)[0]) / "sha256sum").unlink()
    else:
        key = "GENERATIVEQC_CLI" if bad_input == "cli" else "GENERATIVEQC_LIBRARY"
        Path(env[key]).write_bytes(b"unrelated build\n")
    result = _run(root, env, root)
    assert result.returncode != 0
    assert not Path(env["SCHEDULER_LOG"]).exists()
    assert not Path(env["ARCHIVE_LOG"]).exists()


def test_capacity_replay_preserves_failed_comparator_status(
    replay: tuple[Path, dict[str, str], dict[str, str]],
) -> None:
    root, env, _ = replay
    env["FAIL_CASE"] = "water-dimer"
    result = _run(root, env, root)
    assert result.returncode == 17
    assert json.loads((root / ".artifacts/capacity-def2svp.json").read_text()) == {
        "status": "failed"
    }
