"""Use established nested water coordinates and the retained water fixture bases."""

import copy
import json
from pathlib import Path

from test_df_triples_mixed_endpoint import write_input

from benchmarks.readme_hf_scaling import scaling_cases
from tools.generativeqc_posthf.fixtures import load_fixture

root = Path("/home/jzzeng/codes/qc-branch-audit-20260922/evidence-2516")
base, _ = load_fixture("water")
for n in (4, 8):
    atoms = scaling_cases()[f"water-{3 * n}"].atoms
    m = copy.deepcopy(base)
    m["inputs"]["atomic_numbers"] = [{"O": 8, "H": 1}[z] for z, xyz in atoms]
    m["inputs"]["coordinates"] = [list(xyz) for z, xyz in atoms]
    for key in ("shells", "auxiliary_shells"):
        initial = base["inputs"]["shells"] if key == "shells" else base[key]
        expanded = []
        for i in range(n):
            for original in initial:
                s = copy.deepcopy(original)
                s["atom_index"] += 3 * i
                expanded.append(s)
        if key == "shells":
            m["inputs"][key] = expanded
        else:
            m[key] = expanded
    write_input(root / f"water{7 * n}.input", m, budget=8 << 30)
    (root / f"water{7 * n}.metadata.json").write_text(json.dumps(m, indent=2) + "\n")
