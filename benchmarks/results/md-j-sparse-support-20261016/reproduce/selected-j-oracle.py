"""Independent libcint audit of the largest resident raw-J disagreements."""

import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from generativeqc_compiler.dft import NativeAO
from benchmarks._md_j_reference import reference_molecule, schwarz_bounds
from benchmarks.compare_gpu4pyscf_batch import load_comparison_basis
from benchmarks.readme_hf_scaling import scaling_cases


def main():
    root = Path("/data/jzzeng/gqc-direct-md-20261016")
    outputs = []
    for size in (48, 96):
        case = scaling_cases()[f"water-{size}"]
        basis, _ = load_comparison_basis(Path("python/generativeqc/data/r2scan3c/def2-mtzvpp-h-ar.json"), case, role="orbital", compute_forces=True)
        arrays = root / "results/candidate-current/diagnostic/arrays"
        normal = np.load(arrays / f"{size}-atomic-normal.npy", allow_pickle=False)
        candidate = np.load(arrays / f"{size}-atomic-md.npy", allow_pickle=False)
        old_path = root / f"results/baseline-current/diagnostic/arrays/{size}-atomic-md.npy"
        old = np.load(old_path, allow_pickle=False) if old_path.exists() else None
        with NativeAO(case.atoms, basis, representation="spherical") as ao:
            molecule, scales = reference_molecule(ao)
            from pyscf.scf import hf
            density = hf.init_guess_by_minao(molecule)
            started = perf_counter()
            bounds = schwarz_bounds(molecule, scales)
            print(json.dumps({"atoms":size,"schwarz_seconds":perf_counter()-started}),flush=True)
            locations = molecule.ao_loc_nr()
            selected = np.argsort(np.abs(candidate-normal).ravel())[::-1]
            shells = set()
            for location in selected:
                first, second = np.unravel_index(location, normal.shape)
                shell_first = int(np.searchsorted(locations, first, side="right")-1)
                shell_second = int(np.searchsorted(locations, second, side="right")-1)
                key = tuple(sorted((shell_first,shell_second)))
                if key in shells:
                    continue
                shells.add(key)
                started = perf_counter()
                block = molecule.intor("int2e",aosym="s1",shls_slice=(shell_first,shell_first+1,shell_second,shell_second+1,0,molecule.nbas,0,molecule.nbas))
                values = block[first-locations[shell_first],second-locations[shell_second]]
                mask = bounds[first,second] * bounds >= 1e-12
                reference = float(np.sum(values*density*mask,dtype=np.longdouble)*scales[first]*scales[second])
                unscreened = float(np.sum(values*density,dtype=np.longdouble)*scales[first]*scales[second])
                row = {"atoms":size,"ao_pair":[int(first),int(second)],"shell_pair":[shell_first,shell_second],"reference_screened":reference,"reference_unscreened":unscreened,"normal":float(normal[first,second]),"candidate_md":float(candidate[first,second]),"old_md":None if old is None else float(old[first,second]),"normal_error":float(normal[first,second]-reference),"candidate_error":float(candidate[first,second]-reference),"old_error":None if old is None else float(old[first,second]-reference),"oracle_seconds":perf_counter()-started}
                outputs.append(row)
                print(json.dumps(row),flush=True)
                if len(shells)==3:
                    break
        if old is not None:
            print(json.dumps({"atoms":size,"storage_change_maximum_error":float(np.max(np.abs(old-candidate)))}),flush=True)
    destination = root / "results/selected-j-oracle.json"
    destination.write_text(json.dumps(outputs,indent=2)+"\n")


if __name__ == "__main__":
    assert os.environ.get("SLURM_JOB_ID")
    main()
