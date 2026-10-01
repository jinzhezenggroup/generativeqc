"""Count unscreened ERI work from the exact frozen Cartesian shell topology."""
import json
from pathlib import Path

source = Path(__file__).resolve().parent.parent / 'cpu-eri-20261001' / 'cases.json'
rows = {}
for name, case in json.loads(source.read_text()).items():
    shells = [((s[0]+1)*(s[0]+2)//2, len(s)-1) for atom, _ in case['atoms'] for s in case['pyscf_basis'][atom]]
    aos = [p for c,p in shells for _ in range(c)]
    ao_work=0
    ao_quartets=0
    for i in range(len(aos)):
        for j in range(i+1):
            for k in range(i+1):
                for l in range(k+1):
                    if i==k and j<l: continue
                    ao_quartets+=1
                    ao_work+=aos[i]*aos[j]*aos[k]*aos[l]
    offsets=[0]
    for c,_ in shells: offsets.append(offsets[-1]+c)
    geo_work=0
    shell_quartets=0
    grouped_ao_work=0
    grouped_ao_count=0
    maximum_components=0
    for i in range(len(shells)):
        for j in range(i+1):
            for k in range(i+1):
                for l in range(k+1):
                    if i==k and j<l: continue
                    shell_quartets+=1
                    primitives=shells[i][1]*shells[j][1]*shells[k][1]*shells[l][1]
                    geo_work+=primitives
                    count=0
                    for a in range(offsets[i],offsets[i+1]):
                        for b in range(offsets[j],offsets[j+1]):
                            if i==j and b>a: continue
                            for c in range(offsets[k],offsets[k+1]):
                                for d in range(offsets[l],offsets[l+1]):
                                    if k==l and d>c: continue
                                    if i==k and j==l and a*(a+1)//2+b < c*(c+1)//2+d: continue
                                    count+=1
                    grouped_ao_count+=count
                    grouped_ao_work+=count*primitives
                    maximum_components=max(maximum_components,count)
    assert grouped_ao_count==ao_quartets
    assert grouped_ao_work==ao_work
    rows[name]=dict(shells=len(shells),cartesian_aos=len(aos),canonical_ao_quartets=ao_quartets,
       primitive_component_evaluations=ao_work,baseline_geometry_boys_evaluations=ao_work,
       candidate_geometry_boys_evaluations=geo_work,canonical_shell_quartets=shell_quartets,
       maximum_active_components=maximum_components,geometry_reuse_ratio=ao_work/geo_work)
print(json.dumps(rows,indent=2))
