"""CPU-only authentication and scalar/vector arithmetic for current evidence."""
from pathlib import Path
import hashlib
import json
import math

SOURCE = 'cd2b4514be0a38c4920934481b1f02d157a694b4'
PACKAGES = {'cupy':'13.6.0','gpu4pyscf':'1.8.1','numpy':'2.2.6','pyscf':'2.14.0'}

def check(value, message):
    if not value:
        raise ValueError(message)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text())

def integer(value):
    return type(value) is int and 0 <= value <= 2**63-1

def errors(sample, target, atoms):
    for record in [sample,target]:
        check(type(record['energy']) in [float,int] and math.isfinite(record['energy']), 'nonfinite energy')
        check(len(record['forces']) == atoms and all(len(row)==3 and all(type(v) in [float,int] and math.isfinite(v) for v in row) for row in record['forces']), 'invalid complete force vector')
    delta=[a-b for aa,bb in zip(sample['forces'],target['forces'],strict=True) for a,b in zip(aa,bb,strict=True)]
    return {'energy_abs_per_atom_hartree':abs(sample['energy']-target['energy'])/atoms,'force_max_abs_hartree_per_bohr':max(map(abs,delta)),'force_rms_hartree_per_bohr':math.sqrt(math.fsum(v*v for v in delta)/(3*atoms))}

def authenticate(task, cohort, job, scripts, terminal):
    """Read authenticated recovered bytes; never import executed GPU producers."""
    out=task/cohort; c=read(out/'campaign.json'); mp=task/(cohort+'-download-manifest.json'); m=read(mp)
    check(c['source_revision']==SOURCE and c['job']==job and c['status']==terminal,'wrong terminal campaign')
    check(m['cohort']==cohort and m['status']=='COMPLETE_IMMUTABLE_TERMINAL_ARCHIVE' and m['platform_status']=='SUCCEEDED' and m['recorded_provenance_errors']==[] and m['campaign_sha256']==sha(out/'campaign.json'),'terminal archive differs')
    check(set(c['scripts'])==set(scripts),'executed producer inventory differs')
    expected_admissions={'production-build-v1/build-receipt.json','production-build-v1-local-replay.json','screening-v1-local-replay.json','screening-protocol-v3.json','native-qualification-v1-local-replay.json','native-qualification-v1-local-replay-v2.json','standard-guards-v1-local-replay.json','uks-fit-v1-local-replay.json','uks-fit-protocol-v1.json'}
    check(expected_admissions<=set(c['admissions']),'mandatory current admission not bound')
    required={cohort+'/campaign.json'}
    for n,b in m['files'].items():
        p=Path(n)
        check(not p.is_absolute() and '..' not in p.parts and p.name!='.env' and (task/p).is_file() and not (task/p).is_symlink(),'unsafe original member')
        check((task/p).stat().st_size==b['bytes'] and sha(task/p)==b['sha256'],'original bytes differ')
    status_matches=[]
    for n in m['files']:
        if n.startswith(cohort+'-archival-admissions/') and sha(task/n)==m['platform_status_sha256']:
            status_matches.append(read(task/n))
    check(len(status_matches)==1 and status_matches[0]['success'] is True and status_matches[0]['data']['name']==job and status_matches[0]['data']['status']=='SUCCEEDED','exact platform response missing')
    for n,h in c['scripts'].items():
        check(Path(n).name==n,'unsafe producer')
        archived=cohort+'-executed-producers/'+n; required.add(archived)
        check(sha(task/archived)==h==sha(task/n),'executed producer differs')
    for n,h in c['admissions'].items():
        archived=cohort+'-available-admissions/'+n; required.add(archived)
        check(sha(task/archived)==h==sha(task/n),'consumed admission differs')
    for n,h in c['independent_reference_bindings'].items():
        archived=cohort+'-reference-inputs/'+n; required.add(archived)
        check(sha(task/archived)==h,'independent reference differs')
    for row in c['runs']+c.get('reference_runs',[]):
        stem=('reference-' if row in c.get('reference_runs',[]) else '')+row['case']
        check(Path(stem).name==stem,'unsafe population')
        required.update(cohort+'/'+stem+suffix for suffix in ['.plan.json','.json','.stdout','.stderr'])
    check(required<=set(m['files']),'mandatory raw,logs,inputs or producer missing')
    build=read(task/'production-build-v1/build-receipt.json')
    check(build['source_revision']==SOURCE and len(build['artifacts'])==33 and c['production_build_receipt_sha256']==sha(task/'production-build-v1/build-receipt.json'),'production33 identity differs')
    bp=read(task/'production-build-v1-local-replay.json'); native=read(task/'native-qualification-v1-local-replay-v2.json')
    matrix=read(task/'screening-v1-local-replay.json'); prereg=read(task/'screening-protocol-v3.json')
    check(bp['status']=='PASS_RECEIVER_SM90_PRODUCTION33_BUILD_METADATA_ONLY' and bp['build_receipt_sha256']==c['production_build_receipt_sha256'],'build metadata proof differs')
    check(native['status']=='PASS_CURRENT_SOURCE_NATIVE_FOUR_UNFILTERED_MEMCHECK' and native['source_revision']==SOURCE and native['production_core_sha256']==build['artifacts']['libgenerativeqc.so'],'native4 proof differs')
    check(matrix['status']=='PASS_RECEIVER_REPLAY_ALL66_FIXED_POLICY_ATTEMPTS_WITH_FAILURES_RETAINED' and matrix['source_revision']==SOURCE and matrix['build_receipt_sha256']==c['production_build_receipt_sha256'] and matrix['preregistration_sha256']==sha(task/'screening-protocol-v3.json'),'screening admission differs')
    check(len(matrix['global_policy_summaries'])==1 and matrix['global_policy_summaries'][0]['qualified_native_reference_pairs']==matrix['global_policy_summaries'][0]['fixed_primary_inputs']==30,'all30 scientific admission missing')
    check(c['allocation_cgroups']=={'cpu.max':'2000000 100000','memory.max':'214748364800'},'actual allocation differs')
    fit=read(task/'uks-fit-v1-local-replay.json'); original=read(task/'standard-guards-v1-local-replay.json')
    check(original['status']=='NATIVE_GUARDS_FAILED_OR_INCOMPLETE_WITH_ALL_ORIGINALS_RETAINED' and original['qualified_populations']==7 and original['returned_EF']==84,'original failure missing')
    check(fit['status']=='PASS_EXPLICIT_REFINED_UKS_DF_NINE_EF_ORIGINAL_DEFAULT_FAILURE_RETAINED' and fit['source_revision']==SOURCE and fit['original_guard_replay_sha256']==sha(task/'standard-guards-v1-local-replay.json') and fit['original_default_failure_retained'] is True and fit['refined_input_EF']==9,'explicit refinement proof differs')
    return c,m,build,prereg['fixed_policy']

def native_identity(raw, plan, campaign):
    check(raw['source_revision']==SOURCE and raw['qz_allocation']==plan['qz_allocation'] and raw['qz_allocation']['job_name']==campaign['job'] and raw['qz_allocation']['workspace']=='分布式训练空间' and raw['qz_allocation']['project']=='CQ-NexChem','raw source/allocation differs')
    check(raw['accelerator']['nvidia_smi']['uuid']==campaign['device']['uuid'] and raw['accelerator']['name']=='NVIDIA H100 80GB HBM3' and raw['accelerator']['compute_capability']==[9,0] and raw['environment']['packages']==PACKAGES,'actual native H100/runtime differs')

def loaded_core(sample, plan, build):
    identity=sample['native_build']; check(identity['library_sha256']==build['artifacts']['libgenerativeqc.so'],'loaded core hash differs')
    check(Path(identity['library_path']).parent==Path(plan['library_root']) and Path(identity['library_path']).name in ['libgenerativeqc.so','libgenerativeqc.so.0','libgenerativeqc.so.0.1.0'],'loaded core path differs')
