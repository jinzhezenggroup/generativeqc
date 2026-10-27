#!/usr/bin/env python3
"""汇总 GenerativeQC 在 USTC SCC 上跑出的 water-96 standard 算例结果。

读取 readme-defaults.v2 schema:
  samples = [{phase, repeat, seconds, energy, forces:[[x,y,z],...], converged, iterations, ...}]
  phase ∈ {"cold", "warm"}
能量/力的验收门槛来自 protocol.common_observable_gates:
  energy_abs_per_atom_hartree  1e-5
  force_max_abs_hartree_per_bohr 1e-4
  force_rms_hartree_per_bohr   3e-5

两种用法：

  1) 绝对量汇总（默认，只给一个目录）——列出每个算例的机器、冷/热时间、
     收敛轮数、能量和 max|F|，并检查 4 个算例是否都跑出来了。这是 SCC
     上的主用法：结果自洽即可，不需要外部基线。

       python3 tools/scc/collect.py .artifacts/scc-a100-sm80

  2) 两组结果对比（给两个目录）——以第二个目录为基线，报告加速比
     与能量/力偏差是否在门槛内。这是可选的，只在需要一致性对照时用
     （同架构重跑、换节点、或与别的机器/旧版本比）。

       python3 tools/scc/collect.py <dir_a> <dir_b> [out.csv]

     加速比 = baseline / a；>1 表示 a 更快。

能量/力门槛只在对比模式下检查：绝对模式没有参照量，无从判定偏差。
"""
import csv
import json
import math
import sys
from pathlib import Path

CASES = ["standard-hf-direct", "standard-hf-df", "standard-pbe-direct", "standard-pbe-df"]

DEFAULT_GATES = {
    "energy_abs_per_atom_hartree": 1e-5,
    "force_max_abs_hartree_per_bohr": 1e-4,
    "force_rms_hartree_per_bohr": 3e-5,
}


def load(directory, case):
    path = Path(directory) / f"{case}.json"
    if not path.exists():
        print(f"WARN: 缺 {path}", file=sys.stderr)
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001 - 报告并继续，不中断整批汇总
        print(f"WARN: 读不了 {path}: {exc}", file=sys.stderr)
        return None


def gates_of(doc, fallback):
    """优先用结果文件自带的门槛，避免汇总脚本与源码里的门槛漂移。"""
    protocol = (doc or {}).get("protocol") or {}
    declared = protocol.get("common_observable_gates") or {}
    return {
        key: float(declared.get(key, fallback[key])) for key in fallback
    }


def summarise(doc):
    """从 samples 抽出 cold/warm 统计与能量、力。"""
    if doc is None:
        return None
    samples = doc.get("samples") or []
    cold = [s for s in samples if s.get("phase") == "cold"]
    warm = [s for s in samples if s.get("phase") == "warm"]

    out = {"status": doc.get("status", "?")}
    out["atoms"] = doc.get("atoms") or (len(doc.get("geometry_bohr") or []) or None)
    out["method"] = doc.get("method")
    out["mode"] = doc.get("mode")
    out["source_revision"] = doc.get("source_revision")

    out["cold_s"] = cold[0].get("seconds") if cold else None
    out["cold_iterations"] = cold[0].get("iterations") if cold else None
    warm_s = [s.get("seconds") for s in warm if s.get("seconds") is not None]
    out["warm_s"] = (sum(warm_s) / len(warm_s)) if warm_s else None
    out["warm_spread_pct"] = (
        (max(warm_s) - min(warm_s)) / out["warm_s"] * 100
        if warm_s and out["warm_s"]
        else None
    )

    # 能量取最后一次 warm（收敛态），退化到 cold
    last = (warm or cold or [None])[-1]
    out["energy"] = last.get("energy") if last else None
    out["forces"] = last.get("forces") if last else None
    out["converged"] = last.get("converged") if last else None
    out["iterations"] = cold[0].get("iterations") if cold else None
    if out["forces"]:
        out["force_max_norm"] = max(
            math.sqrt(sum(c * c for c in row)) for row in out["forces"]
        )

    acc = doc.get("accelerator") or {}
    out["device"] = acc.get("name")
    out["cc"] = tuple(acc.get("compute_capability") or ())
    out["vram_gb"] = (
        acc.get("total_global_memory_bytes", 0) / 1024**3
        if acc.get("total_global_memory_bytes")
        else None
    )
    env = doc.get("environment") or {}
    out["node"] = (env.get("host") or {}).get("node")
    out["lib"] = (env.get("runtime") or {}).get("generativeqc_library")
    out["commit"] = (env.get("git") or {}).get("commit")
    out["dirty"] = (env.get("git") or {}).get("dirty")
    out["threads"] = (doc.get("thread_controls") or {}).get("SLURM_CPUS_PER_TASK")
    return out


def force_stats(a, b):
    """a,b 都是 [[x,y,z],...]。返回 (max_abs, rms)。"""
    if not a or not b or len(a) != len(b):
        return None, None
    diffs = []
    for fa, fb in zip(a, b):
        for ca, cb in zip(fa, fb):
            diffs.append(ca - cb)
    if not diffs:
        return None, None
    max_abs = max(abs(d) for d in diffs)
    rms = math.sqrt(sum(d * d for d in diffs) / len(diffs))
    return max_abs, rms


def fmt(value, spec=".3f"):
    return "" if value is None else format(value, spec)


def report_absolute(rows):
    """只有一个目录：报绝对量，不做对比。"""
    header = (
        f"{'case':<22} {'状态':>7} {'冷(s)':>9} {'热(s)':>9} {'抖动%':>6} "
        f"{'迭代':>4} {'能量/Ha':>18} {'max|F|':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['case']:<22} {row.get('status') or '-':>7} "
            f"{fmt(row.get('cold_s'), '.2f'):>9} {fmt(row.get('warm_s'), '.2f'):>9} "
            f"{fmt(row.get('warm_spread_pct'), '.1f'):>6} "
            f"{fmt(row.get('iterations'), '.0f'):>4} "
            f"{fmt(row.get('energy'), '.12f'):>18} "
            f"{fmt(row.get('force_max_norm'), '.2e'):>9}"
        )

    print()
    for row in rows:
        cc = ".".join(map(str, row.get("cc") or ())) or "?"
        print(
            f"  {row['case']}: {row.get('device') or '?'} CC {cc} "
            f"{fmt(row.get('vram_gb'), '.1f') or '?'} GB @ {row.get('node') or '?'}  "
            f"converged={row.get('converged')} lib={row.get('lib') or '?'}"
        )

    revs = {row.get("commit") for row in rows if row.get("commit")}
    if revs:
        dirty = any(row.get("dirty") for row in rows)
        print(f"\n  source commit: {', '.join(sorted(revs))}{' (dirty)' if dirty else ''}")


def report_comparison(rows, base_label, cand_label):
    """两个目录：报加速比与能量/力偏差门槛。"""
    header = (
        f"{'case':<22} {cand_label[:6] + ' 冷':>9} {base_label[:6] + ' 冷':>9} {'倍':>6} | "
        f"{cand_label[:6] + ' 热':>9} {base_label[:6] + ' 热':>9} {'倍':>6} | "
        f"{'E偏差/原子':>11} {'力max':>9} {'门槛':>5}"
    )
    print(header)
    print("-" * len(header))

    def num(row, key):
        try:
            return float(row.get(key, ""))
        except (TypeError, ValueError):
            return None

    for row in rows:
        def s(key, spec=".2f"):
            v = num(row, key)
            return "-" if v is None else format(v, spec)

        print(
            f"{row['case']:<22} {s('cand_cold_s'):>9} {s('base_cold_s'):>9} "
            f"{s('speedup_cold_x'):>6} | "
            f"{s('cand_warm_s'):>9} {s('base_warm_s'):>9} {s('speedup_warm_x'):>6} | "
            f"{s('energy_dev_per_atom', '.2e'):>11} {s('force_max_abs', '.2e'):>9} "
            f"{row.get('gate', ''):>5}"
        )


def main(argv):
    if len(argv) < 2:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        print("\n用法: collect.py <结果目录> [基线目录] [out.csv]", file=sys.stderr)
        return 2

    cand_dir = Path(argv[1])
    base_dir = Path(argv[2]) if len(argv) > 2 else None
    out_csv = Path(argv[3]) if len(argv) > 3 else (
        cand_dir.parent / f"{cand_dir.name}-summary.csv"
    )

    if not cand_dir.is_dir():
        print(f"ERROR: 结果目录不存在: {cand_dir}", file=sys.stderr)
        return 2
    if base_dir is not None and not base_dir.is_dir():
        print(f"ERROR: 基线目录不存在: {base_dir}", file=sys.stderr)
        return 2

    print(f"结果目录: {cand_dir}")
    if base_dir is not None:
        print(f"基线目录: {base_dir}")
    print()

    docs = {case: load(cand_dir, case) for case in CASES}
    base_docs = {case: load(base_dir, case) for case in CASES} if base_dir else {}

    all_gates = [gates_of(d, DEFAULT_GATES) for d in docs.values() if d]
    gates = all_gates[0] if all_gates else dict(DEFAULT_GATES)

    # ---------- 绝对量模式 ----------
    if base_dir is None:
        rows = []
        for case in CASES:
            item = summarise(docs[case])
            row = {"case": case}
            if item is None:
                row["status"] = "missing"
            else:
                row.update(
                    status=item["status"],
                    cold_s=item["cold_s"],
                    warm_s=item["warm_s"],
                    warm_spread_pct=item["warm_spread_pct"],
                    iterations=item["iterations"],
                    energy=item["energy"],
                    force_max_norm=item.get("force_max_norm"),
                    converged=item["converged"],
                    device=item["device"],
                    cc=item["cc"],
                    vram_gb=item["vram_gb"],
                    node=item["node"],
                    lib=item["lib"],
                    commit=item["commit"],
                    dirty=item["dirty"],
                    threads=item["threads"],
                )
            rows.append(row)

        report_absolute(rows)

        fields = [
            "case", "status", "cold_s", "warm_s", "warm_spread_pct", "iterations",
            "energy", "force_max_norm", "converged", "device", "cc", "vram_gb",
            "node", "lib", "commit", "dirty", "threads",
        ]
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                flat = dict(row)
                flat["cc"] = ".".join(map(str, row.get("cc") or ()))
                writer.writerow(flat)
        print(f"\n写出 {out_csv}")

        missing = [r["case"] for r in rows if r.get("status") == "missing"]
        if missing:
            print(f"\n缺失结果: {missing}", file=sys.stderr)
            return 1
        if any(r.get("status") != "ok" for r in rows):
            bad = [r["case"] for r in rows if r.get("status") != "ok"]
            print(f"\n状态非 ok 的算例: {bad}", file=sys.stderr)
            return 1
        return 0

    # ---------- 对比模式 ----------
    rows = []
    for case in CASES:
        a = summarise(docs[case])
        b = summarise(base_docs[case])
        row = {"case": case}
        if a is None:
            row["cand_status"] = "missing"
        else:
            row["cand_status"] = a["status"]
            row["cand_device"] = a["device"] or ""
            row["cand_cc"] = ".".join(map(str, a["cc"])) if a["cc"] else ""
            row["cand_cold_s"] = fmt(a["cold_s"])
            row["cand_warm_s"] = fmt(a["warm_s"])
        if b is None:
            row["base_status"] = "missing"
        else:
            row["base_status"] = b["status"]
            row["base_device"] = b["device"] or ""
            row["base_cold_s"] = fmt(b["cold_s"])
            row["base_warm_s"] = fmt(b["warm_s"])

        if a and b:
            for phase, key in (("cold", "cold_s"), ("warm", "warm_s")):
                if a[key] and b[key]:
                    row[f"speedup_{phase}_x"] = fmt(b[key] / a[key])
                else:
                    row[f"speedup_{phase}_x"] = ""

            atoms = a["atoms"] or b["atoms"] or 1
            if a["energy"] is not None and b["energy"] is not None:
                dev = abs(a["energy"] - b["energy"])
                row["energy_abs_dev"] = fmt(dev, ".3e")
                row["energy_dev_per_atom"] = fmt(dev / atoms, ".3e")
                energy_ok = dev / atoms <= gates["energy_abs_per_atom_hartree"]
            else:
                row["energy_abs_dev"] = ""
                row["energy_dev_per_atom"] = ""
                energy_ok = None

            fmax, frms = force_stats(a["forces"], b["forces"])
            row["force_max_abs"] = fmt(fmax, ".3e")
            row["force_rms"] = fmt(frms, ".3e")
            force_ok = (
                None
                if fmax is None
                else (
                    fmax <= gates["force_max_abs_hartree_per_bohr"]
                    and frms <= gates["force_rms_hartree_per_bohr"]
                )
            )

            if energy_ok is None or force_ok is None:
                row["gate"] = ""
            else:
                row["gate"] = "PASS" if (energy_ok and force_ok) else "FAIL"

        rows.append(row)

    report_comparison(rows, base_dir.name, cand_dir.name)

    fields = [
        "case",
        "cand_device", "cand_cc", "cand_status", "cand_cold_s", "cand_warm_s",
        "base_device", "base_status", "base_cold_s", "base_warm_s",
        "speedup_cold_x", "speedup_warm_x",
        "energy_abs_dev", "energy_dev_per_atom", "force_max_abs", "force_rms", "gate",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n写出 {out_csv}")

    missing = [
        r["case"]
        for r in rows
        if r.get("cand_status") == "missing" or r.get("base_status") == "missing"
    ]
    if missing:
        print(f"\n缺失结果（候选或基线）: {missing}", file=sys.stderr)
        return 1
    failed = [r["case"] for r in rows if r.get("gate") == "FAIL"]
    if failed:
        print(f"\n未通过门槛: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
