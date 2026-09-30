from pathlib import Path
import re, subprocess, sys
r=Path(sys.argv[1])
pat=re.compile(r'^<<<<<<<[^\n]*\n(.*?)^=======\n(.*?)^>>>>>>>[^\n]*\n',re.M|re.S)
def master(p):return subprocess.check_output(['git','show','c8886146a77a115681eb5b90ef6c11cef022fdea:'+p],cwd=r,text=True)
def funcs(s,name):
 out=[]
 for m in re.finditer(r'^([ ]*)generativeqc_status '+name+r'\(',s,re.M):
  opening=s.index('{',m.start()); depth=1;j=opening+1
  while depth:
   depth+=(s[j]=='{')-(s[j]=='}');j+=1
  out.append(s[m.start():j])
 return out
for path in ['src/api/c_api_ks_snapshot.cpp','src/methods/dft_method.cpp']:
 p=r/path;s=pat.sub(lambda m:m.group(1),p.read_text())
 if '/api/' in path:
  f=funcs(master(path),'generativeqc_ks_snapshot_cuda_seed_nonlocal_force_v1');assert len(f)==1
  idx=s.index('// Compact two-source handoff')
  s=s[:idx]+f[0]+'\n\n'+s[idx:]
 else:
  old=funcs(master(path),'cuda_full_range_integral_derivatives');assert len(old)==2
  positions=list(re.finditer(r'^  generativeqc_status cuda_full_range_shell_gradient\(',s,re.M))
  assert len(positions)==2
  for m,f in reversed(list(zip(positions,old))): s=s[:m.start()]+f+'\n\n'+s[m.start():]
  old=funcs(master(path),'dft_cuda_full_range_integral_derivatives');assert len(old)==1
  idx=s.index('generativeqc_status dft_cuda_full_range_shell_gradient(')
  s=s[:idx]+old[0]+'\n\n'+s[idx:]
 p.write_text(s)
for path in ['src/api/ks_snapshot.hpp','src/dft/stationary_gradient_cuda.cuh']:
 p=r/path;s=pat.sub(lambda m:m.group(1)+'\n'+m.group(2),p.read_text())
 if path.endswith('.cuh'):
  idx=s.index('int stationary_geometry_reset(')
  before,body=s[:idx],s[idx:]
  body=body.replace('    p->failed = false;','    p->failed = false;\n    p->source_seed_ready = false;',1)
  s=before+body
 p.write_text(s)
for path in ['tests/python/test_dft_mp_v1_capacity.py','tools/dft_mp_v1/qualify_capacity.py']:
 p=r/path;p.write_text(pat.sub(lambda m:m.group(1),p.read_text()))
p=r/'python/generativeqc/_stationary_cuda.py';s=pat.sub(lambda m:m.group(1),p.read_text())
s=s.replace('        descriptor_records = records - len(shell_sources) * primitive_sum**4','        native_shell_full_range = bool(shell_sources)\n        descriptor_records = records - len(shell_sources) * primitive_sum**4')
s=s.replace('    ao_quartet_primitive_records = (1 + int(has_exchange)) * primitive_sum**4\n','')
s=s.replace('        ):\n            if source in shell_sources:', '        )\n        for source, rank, operator in task_sources:\n            if source in shell_sources:')
start=s.index('        if native_shell_full_range:\n            components["coulomb"]')
end=s.index('        if ecp:',start)
s=s[:start]+s[end:]
old='''                if native_shell_full_range:
                    gradient = gradient + components["coulomb"]
                    if has_exchange:
                        gradient = gradient + components["exact_exchange"]
'''
assert old in s;s=s.replace(old,'')
s=s.replace('''                "native-plan-source-device-sum-plus-direct-shell-compose-v1"
                if native_shell_full_range
                else (
                    "native-plan-source-device-sum-v1"
                    if has_exchange
                    else "native-seven-source-device-sum-v1"
                )''','''                "native-plan-source-device-sum-v1"
                if has_exchange
                else "native-seven-source-device-sum-v1"''')
p.write_text(s)
p=r/'tests/python/test_direct_shell_derivative_owner.py';s=p.read_text();s=s.replace('    assert "records -= ao_quartet_primitive_records" in stationary','''    assert "seed_prepared_shell_sources(" in stationary
    assert "descriptor_records = records - len(shell_sources) * primitive_sum**4" in stationary
    assert 'gradient = gradient + components["coulomb"]' not in stationary''')
p.write_text(s)
p=r/'tests/python/test_stationary_geometry_reset.py';s=p.read_text()
s=s.replace('    expected = "".join(line for line in lines if line not in removed)', '''    # Only a full density reset may open the one-shot integral-source seed gate.
    seed_enable = [
        line for line in lines
        if "p->source_seed_launch_epoch =" in line or "p->source_seed_ready = true;" in line
    ]
    assert len(seed_enable) == 2
    assert "p->source_seed_ready = false;" in geometry
    assert "p->source_seed_ready = true;" not in geometry
    expected = "".join(line for line in lines if line not in removed + seed_enable)''')
p.write_text(s)
