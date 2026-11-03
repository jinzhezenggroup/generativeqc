"""Independently qualify every final vector and aggregate exactly90 fixed pairs."""
from pathlib import Path
import hashlib
import json
import lzma
import runpy
import math
import statistics
import subprocess
import sys

TASK = Path(__file__).resolve().parent
COHORT = 'final-pairs-v3'
SOURCE = 'cd2b4514be0a38c4920934481b1f02d157a694b4'


def check(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def numerical(sample, target, atoms):
    for row in [sample, target]:
        check(type(row['energy']) in [int, float] and math.isfinite(row['energy']) and len(row['forces']) == atoms, 'invalid complete E/F')
        check(all(len(force) == 3 and all(type(value) in [int, float] and math.isfinite(value) for value in force) for force in row['forces']), 'invalid dense force vector')
    differences = [a - b for aa, bb in zip(sample['forces'], target['forces'], strict=True) for a, b in zip(aa, bb, strict=True)]
    return {'energy_abs_per_atom_hartree': abs(sample['energy'] - target['energy']) / atoms,
            'force_max_abs_hartree_per_bohr': max(map(abs, differences)),
            'force_rms_hartree_per_bohr': math.sqrt(math.fsum(value * value for value in differences) / (3 * atoms))}


def emit(result):
    path = TASK / (COHORT + '-local-replay.json')
    check(not path.exists(), 'inspect previous final replay')
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ['rows', 'cases']}))


def mapped_native_images(raw,plan,build):
    root=Path(plan['library_root']);core=Path(raw['native_build']['library_path']);images=raw.get('runtime_images_after_endpoint',{})
    check(core.parent==root and images,'mapped production core missing')
    cores=[n for n in images if Path(n).name=='libgenerativeqc.so' or Path(n).name.startswith('libgenerativeqc.so.')]
    check(cores==[str(core)],'alternate or missing mapped production core')
    for n,b in images.items():
        expected=build['artifacts']['libgenerativeqc.so'] if n==str(core) else build['artifacts'].get(Path(n).name)
        check(expected is not None and Path(n).parent==root and b.get('available_after_endpoint') is True and type(b.get('bytes')) is int and b['bytes']>0 and b['sha256']==expected,'unknown/changed mapped production image')

def main():
    out=TASK/COHORT;mp=TASK/(COHORT+'-download-manifest.json');manifest=read(mp);campaign=read(out/'campaign.json');manifest_path=mp
    check(manifest['status']=='COMPLETE_IMMUTABLE_TERMINAL_ARCHIVE' and manifest['cohort']==COHORT and manifest['platform_status']=='SUCCEEDED' and manifest['recorded_provenance_errors']==[],'terminal archive differs')
    archive=TASK/manifest['archive'];check(archive.parent==TASK and archive.stat().st_size==manifest['bytes'] and sha(archive)==manifest['sha256'],'exact archive differs')
    for n,b in manifest['files'].items():
        path=Path(n);check(not path.is_absolute() and '..' not in path.parts and path.name!='.env' and (TASK/path).is_file() and not (TASK/path).is_symlink() and (TASK/path).stat().st_size==b['bytes'] and sha(TASK/path)==b['sha256'],'recovered original differs')
    check(campaign['job']=='gqc-h100-maincd2b-final-pairs-1102-v3' and campaign['source_revision']==SOURCE and campaign['status'] in ['TERMINAL_180_FRESH_PROCESS_ATTEMPTS_ALL_PARENT_QUALIFIED_REPLAY_PENDING','TERMINAL_FINAL_WITH_FAILURES_REPLAY_PENDING'] and sha(out/'campaign.json')==manifest['campaign_sha256'],'terminal final source/job differs')
    statuses=[read(TASK/n) for n in manifest['files'] if n.startswith(COHORT+'-archival-admissions/') and sha(TASK/n)==manifest['platform_status_sha256']]
    check(len(statuses)==1 and statuses[0]['success'] is True and statuses[0]['data']['name']==campaign['job'] and statuses[0]['data']['status']=='SUCCEEDED','platform terminal response differs')
    scripts={'run-final-pairs-v10.py','run-final-pairs-v10-job.sh','final-native-endpoint-v1.py','final-reference-endpoint-v1.py','final-hf-fit-reference-v1.py','final-reference-runtime-v2.py','cohort-gates-v4.py','cohort-gates-v3.py','cohort-gates.py','process-ownership-v1.py','run-screening-v3.py','pilot.py','final-pairs-protocol-v4.json','original-final-protocol-v1.json','final-admissions-v6.py','reference-image-observation-v1.py'}
    check(set(campaign['scripts'])==scripts,'mandatory executed producers differ')
    required={COHORT+'/campaign.json',COHORT+'/reference-runtime.json',COHORT+'/reference-runtime.stdout',COHORT+'/reference-runtime.stderr',COHORT+'/screening-input-continuity.json',COHORT+'/screening-campaign.json'}
    for name,digest in campaign['scripts'].items():
        archived=COHORT+'-executed-producers/'+name;check(Path(name).name==name and sha(TASK/name)==digest==sha(TASK/archived),'executed producer differs');required.add(archived)
    for name,digest in campaign['admissions'].items():
        archived=COHORT+'-available-admissions/'+name;check(sha(TASK/name)==digest==sha(TASK/archived),'consumed prior admission differs');required.add(archived)
    for stage in campaign['runs']:required.update(COHORT+'/'+stage['case']+suffix for suffix in ['.plan.json','.json','.qualification.json','.stdout','.stderr'])
    check(required<=set(manifest['files']),'mandatory final raw/log/producer/admission missing')
    protocol_path=TASK/'final-pairs-protocol-v4.json';protocol=read(protocol_path);original_path=TASK/'original-final-protocol-v1.json';original=read(original_path)
    check(sha(protocol_path)==campaign['protocol_sha256'] and sha(original_path)==protocol['final_protocol_sha256']=='2391b71fb49660f682f1b60056b2892d5516f01871d7479319f0be2cf925fdaa','frozen final contract differs')
    check(all(protocol[name]==original[name] for name in ['primary_physical_cases','numerical_gates','win_rule','overall_gate','measurement']) and protocol['required_win_count']==20,'fixed30 scientific denominator/gates changed')
    expected=[]
    for pair in range(3):
        for index,case in enumerate(original['primary_physical_cases']):
            arms=['standard','gpu4pyscf'] if (index+pair)%2==0 else ['gpu4pyscf','standard']
            for arm in arms:expected.append({'ordinal':len(expected),'case_id':case['id'],'method':case['method'],'atoms':case['atoms'],'mode':case['route'],'pair_index':pair,'arm':arm,'pair_order':arms})
    check(protocol['order']==expected and len(expected)==len(campaign['runs'])==180,'ordered180 fresh populations missing')
    build_path=TASK/'production-build-v1/build-receipt.json';build=read(build_path);check(build['source_revision']==SOURCE and len(build['artifacts'])==33 and campaign['build_receipt_sha256']==sha(build_path) and campaign['artifacts']==build['artifacts'],'current production33 differs')
    library_root=Path(build['stages'][0]['command'][build['stages'][0]['command'].index('-B')+1]);selection={'selected_policy':protocol['fixed_policy']}
    check(protocol['fixed_policy']==read(TASK/'screening-protocol-v3.json')['fixed_policy']=={'id':'integrated-auto-rys-public-tile1024','ks_tile_points':1024,'df_environment':{}},'carried global policy changed')
    import importlib.util
    spec=importlib.util.spec_from_file_location('final_prior_acceptance',TASK/'final-admissions-v6.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
    strict_bindings=prior.admit(TASK);check(all(campaign['admissions'].get(name)==digest for name,digest in strict_bindings.items()),'exact build-bound prior acceptance differs')
    dispatch=read(TASK/'nsys-v1-local-replay-v2.json')
    original_standard=read(TASK/'standard-guards-v1-local-replay.json');fit=read(TASK/'uks-fit-v1-local-replay.json')
    check(original_standard['returned_EF']==84 and original_standard['qualified_populations']==7 and fit['original_default_failure_retained'] is True and fit['original_guard_replay_sha256']==sha(TASK/'standard-guards-v1-local-replay.json') and fit['refined_input_EF']==9,'original default failure/refinedinput boundary lost')
    retained=out/'retained-inputs';freeze_path=retained/'current64de-final-inputs-frozen-v1.json';check(sha(freeze_path)==protocol['input_freeze_sha256'],'frozen input record differs');frozen=read(freeze_path)
    for kind,group in frozen['groups'].items():
        check(sha(retained/(kind+'.json'))==group['manifest_sha256'],'frozen table manifest differs')
        for name,b in group['files'].items():
            archived=out/'frozen-inputs'/kind/name;check(archived.stat().st_size==b['bytes'] and sha(archived)==b['sha256'],'actual frozen table differs')
    matrix=read(TASK/'screening-v1-local-replay.json');screen=read(out/'screening-campaign.json');check(sha(out/'screening-campaign.json')==matrix['campaign_sha256'] and screen['source_revision']==SOURCE,'actual screening campaign differs')
    continuity=[]
    for case in original['primary_physical_cases']:
        for arm in ['tile1024','reference']:
            stem=f"{case['method']}-{case['atoms']}-{case['route']}-{arm}";stage=next(row for row in screen['runs'] if row['case']==stem);pp=out/'screening-plans'/(stem+'.plan.json');plan=read(pp)
            kind='inputs-hf-ccpvtz' if case['method']=='hf' and case['route']=='df' else 'inputs-default'
            check(stage['status']=='COMPLETE' and stage['qualification']=='qualified' and stage['sample_count']==4 and stage['plan_sha256']==sha(pp) and plan['input_sha256']=={n:b['sha256'] for n,b in frozen['groups'][kind]['files'].items()},'screening scientific plan continuity differs')
            continuity.append({'case':stem,'plan_sha256':sha(pp),'input_sha256':plan['input_sha256']})
    check(read(out/'screening-input-continuity.json')==continuity and sha(out/'screening-input-continuity.json')==campaign['screening_input_continuity_sha256'],'actual input continuity record differs')
    runtime=read(out/'reference-runtime.json');check(runtime['status']=='PASS_REFERENCE_RUNTIME_AND_CUTENSOR_ONLY' and runtime['script_sha256']==campaign['scripts']['final-reference-runtime-v2.py'] and runtime['installed_files_equal_original_wheels']>0 and runtime['max_abs_error']<1e-12 and campaign['device']['uuid'] in runtime['device'],'actual reference runtime differs')
    ROOT=globals().get('ROOT',TASK.parents[1]/'generativeqc-h100-production');base={'revision':protocol['reference_source_revision'],'driver_sha256':protocol['reference_driver_sha256']}
    sys.path[:0]=[str(ROOT),str(ROOT/'python')]
    for name in list(sys.modules):
        if name=='benchmarks' or name.startswith('benchmarks.'):del sys.modules[name]
    from benchmarks.readme_defaults_qualification import qualify
    for name,module in list(sys.modules.items()):
        if name=='benchmarks' or name.startswith('benchmarks.'):
            path=Path(module.__file__);relative=path.relative_to(ROOT).as_posix();check(subprocess.check_output(['git','-C',str(ROOT),'show',base['revision']+':'+relative],env={**__import__('os').environ,'GIT_NO_LAZY_FETCH':'1','GIT_ALLOW_PROTOCOL':'','GIT_TERMINAL_PROMPT':'0'})==path.read_bytes(),'imported independent frozen qualifier/helper differs')
    qualifier_path=ROOT/'benchmarks/readme_defaults_qualification.py';oracle_path=TASK/(COHORT+'-reference-inputs/data/oracles.json.xz');check(sha(oracle_path)==protocol['oracle_sha256'],'independent original oracle differs');oracles=json.loads(lzma.decompress(oracle_path.read_bytes()))
    reuse=read(retained/'reference-reuse-v2.json');check(sha(retained/'reference-reuse-v2.json')==protocol['reference_reuse_sha256'],'explicitfit reference-reuse contract differs')
    input_hashes={kind:{name:b['sha256'] for name,b in group['files'].items()} for kind,group in frozen['groups'].items()}
    rows, indexed, raw_index, returned = [], {}, {}, 0
    for declared, stage in zip(expected, campaign['runs']):
        key = (declared['case_id'], declared['pair_index'], declared['arm'])
        check(key not in indexed and stage['declaration'] == declared, 'duplicate or reordered final attempt')
        stem = f"{declared['ordinal']:03d}-pair{declared['pair_index']}-{declared['case_id']}-{declared['arm']}"
        check(stage['case'] == stem, 'final attempt name differs')
        plan_path, raw_path = out / (stem + '.plan.json'), out / (stem + '.json')
        plan = read(plan_path)
        check(sha(plan_path) == stage['plan_sha256'] and plan['paired_process'] == declared and plan['final_protocol_sha256'] == sha(protocol_path) and plan['phase'] == protocol['endpoint_phase'], 'final plan identity differs')
        method, atoms, mode, arm = (declared[name] for name in ['method', 'atoms', 'mode', 'arm'])
        reference_arm = arm == 'gpu4pyscf'
        endpoint = ('final-hf-fit-reference-v1.py' if method == 'hf' and mode == 'df' else 'final-reference-endpoint-v1.py') if reference_arm else 'final-native-endpoint-v1.py'
        check((plan['method'], plan['atoms'], plan['mode'], plan['arm']) == (method, atoms, mode, arm) and plan['primary_ratio_eligible'] is True and plan['endpoint_sha256'] == sha(TASK / endpoint) == protocol['endpoints'][endpoint]['sha256'], 'scientific input or endpoint changed')
        check(plan['source_revision'] == (base['revision'] if reference_arm else SOURCE) and plan['driver_sha256'] == (base['driver_sha256'] if reference_arm else protocol['native_driver_sha256']) and plan['build_artifacts'] == ({} if reference_arm else build['artifacts']), 'source/core differs')
        check(plan['input_sha256'] == input_hashes['inputs-hf-ccpvtz' if method == 'hf' and mode == 'df' else 'inputs-default'], 'scientific basis/fitting bytes differ')
        if method == 'hf' and mode == 'df':
            check(plan['fit_policy'] == protocol['fitting_policy'], 'explicit HF fitting contract differs')
        if not reference_arm:
            check(plan['execution_policy'] == selection['selected_policy'] and Path(plan['library_root'])==library_root, 'production global policy/core differs')
        row = {'case': stem, 'case_id': declared['case_id'], 'pair_index': declared['pair_index'], 'arm': arm, 'returncode': stage.get('returncode'), 'status': 'FAIL_OR_INCOMPLETE', 'returned_EF': 0}
        indexed[key] = row
        rows.append(row)
        if not raw_path.exists():
            row['error'] = 'NO_RAW_RECEIPT'
            continue
        raw = read(raw_path)
        raw_index[key] = raw
        # A parent interrupted after writing child data but before binding its
        # return code/hash retains vectors for diagnostics and cannot qualify.
        bound = type(stage.get('returncode')) is int and stage.get('receipt_sha256') == sha(raw_path)
        check(raw['plan_sha256'] == sha(plan_path) and raw['runner_sha256'] == sha(TASK / endpoint) and raw['source_revision'] == plan['source_revision'] and raw.get('driver_sha256', raw.get('driver_context_sha256')) == plan['driver_sha256'], 'raw producer/source differs')
        check((raw['method'], raw['atoms'], raw['mode'], raw['arm']) == (method, atoms, mode, arm) and raw['geometry_bohr'] == plan['geometry_bohr'] and raw['qz_allocation'] == plan['qz_allocation'] and raw['qz_allocation']['job_name'] == campaign['job'], 'returned model/geometry/job differs')
        check(raw['accelerator']['nvidia_smi']['uuid'] == campaign['device']['uuid'] and raw['accelerator']['compute_capability'] == [9, 0] and raw['scheduler']['kind'] == 'qz_job' and raw['slurm_job_id'] is None, 'actual physical GPU/scheduler differs')
        packages = {'cupy': '13.4.1' if reference_arm else '13.6.0', 'gpu4pyscf': '1.9.0' if reference_arm else '1.8.1', 'numpy': '2.2.6', 'pyscf': '2.14.0'}
        check(raw['environment']['packages'] == packages and raw['thread_controls'] == {name: '1' for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']} and raw['protocol']['common_observable_gates'] == protocol['numerical_gates'], 'runtime/thread controls or gates differ')
        if reference_arm:
            observation=raw.get('reference_scientific_images_after_endpoint')
            check(observation and observation['status']=='PASS_ACTUAL_REFERENCE_SCIENTIFIC_IMAGES_MATCH_PREFLIGHT' and observation['preflight_sha256']==plan['reference_runtime_receipt_sha256']==sha(out/'reference-runtime.json'),'per-child scientific image observation missing')
            images=observation['images'];catalog=runtime['scientific_image_catalog']
            check(images and any('cupy' in name for name in images) and any('gpu4pyscf' in name for name in images) and any('libcutensor.so' in Path(name).name for name in images),'actual reference images missing')
            check(all(name in catalog and binding==catalog[name] and 'libgenerativeqc.so' not in Path(name).name for name,binding in images.items()),'actual reference image differs from preflight catalog')
            spec=importlib.util.spec_from_file_location('reference_image_schema',TASK/'reference-image-observation-v1.py');image_schema=importlib.util.module_from_spec(spec);spec.loader.exec_module(image_schema)
            check('outside_catalog_runtime_images' in observation and all(not image_schema.scientific_provider(name) and 'libgenerativeqc.so' not in Path(name).name for name in observation['outside_catalog_runtime_images']),'scientific provider outside admitted roots')
        target = next(item for item in oracles if (item['method'], item['atoms'], item['backend']) == (method, atoms, 'cuda'))
        check(raw['geometry_bohr'] == target['geometry_bohr'], 'canonical geometry differs')
        exact = qualify([raw], oracles, reference_reuse=reuse)[0]
        errors = [numerical(sample, target['samples'][-1], atoms) for sample in raw['samples']]
        passing = [all(value <= protocol['numerical_gates'][name] for name, value in differences.items()) for differences in errors]
        for sample in raw['samples']:
            check(type(sample['seconds']) in [int, float] and math.isfinite(sample['seconds']) and sample['seconds'] > 0, 'invalid synchronized full-E/F duration')
        route = reference_arm or method == 'hf' or raw.get('native_prepared_force_capabilities') == [True] * 4
        phases = [sample['phase'] for sample in raw['samples']]
        repeats = [sample['repeat'] for sample in raw['samples']]
        warm_contract = phases == ['cold', 'warm', 'warm', 'warm'] and repeats == [0, 0, 1, 2]
        if warm_contract:
            warm_contract &= raw['samples'][0].get('warm_start_used') is False and all(sample.get('warm_start_used') is True for sample in raw['samples'][1:])
            warm_contract &= all(sample.get('warm_start_fallback') is False for sample in raw['samples']) if not reference_arm else all(sample.get('warm_start_fallback') in [False, None] for sample in raw['samples'])
        resolved_policy = reference_arm or raw.get('resolved_execution_policy') == {'ks_tile_points': None if method == 'hf' else selection['selected_policy']['ks_tile_points'], 'df_environment': selection['selected_policy']['df_environment']}
        endpoint_contract = raw['protocol'].get('moved_geometry') is False and raw['protocol'].get('grid_overrides') is False
        for sample in raw['samples']:
            settings = sample['settings']
            if method != 'hf':
                endpoint_contract &= settings.get('force_grid_response') is True
            if reference_arm and mode == 'df':
                endpoint_contract &= settings.get('auxbasis_response') is True
            count_name = 'effective_potential_builds' if reference_arm else 'fock_builds'
            endpoint_contract &= count_name in sample and type(sample[count_name]) is int and sample[count_name] >= 0
            if not reference_arm and method!='hf':
                tile=settings.get('resolved_force_tile_points')
                endpoint_contract &= type(tile) is int and 1<=tile<=4096
                endpoint_contract &= type(settings.get('scf_tile_points')) is int and settings['scf_tile_points']>0
        if not reference_arm and raw['samples']:
            mapped_native_images(raw,plan,build)
            check(raw['native_build']['library_sha256'] == build['artifacts']['libgenerativeqc.so'] and raw['execution_policy'] == selection['selected_policy'], 'actual native core/policy differs')
        qualified = bool(bound and stage['returncode'] == 0 and stage['status'] == 'COMPLETE' and raw['status'] == 'complete' and len(errors) == 4 and all(passing) and exact['status'] == 'qualified' and route and warm_contract and resolved_policy and endpoint_contract)
        if qualified:
            check([sample['phase'] for sample in raw['samples']] == ['cold', 'warm', 'warm', 'warm'] and [sample['repeat'] for sample in raw['samples']] == [0, 0, 1, 2], 'cold/warm phases differ')
            row.update(status='QUALIFIED', warm_seconds=[sample['seconds'] for sample in raw['samples'][1:]], warm_median_seconds=statistics.median(sample['seconds'] for sample in raw['samples'][1:]))
        row.update(returned_EF=len(errors), parent_binding_complete=bound, qualifier_status=exact['status'], errors=errors, native_route_accepted=route, own_cold_seed_warm_contract=bool(warm_contract), resolved_policy_matches=bool(resolved_policy), static_full_force_and_work_metadata_contract=bool(endpoint_contract), raw_error=raw.get('error'), iterations=[sample['iterations'] for sample in raw['samples']], fock_builds=[sample.get('fock_builds') for sample in raw['samples']], reference_effective_potential_builds=[sample.get('effective_potential_builds') for sample in raw['samples']])
        returned += len(errors)
    cases = []
    for case in original['primary_physical_cases']:
        pairs = []
        for pair in range(3):
            native_key, reference_key = (case['id'], pair, 'standard'), (case['id'], pair, 'gpu4pyscf')
            native_row, reference_row = indexed.get(native_key), indexed.get(reference_key)
            qualified = bool(native_row and reference_row and native_row['status'] == reference_row['status'] == 'QUALIFIED')
            cross_errors = []
            if qualified:
                for native_sample in raw_index[native_key]['samples']:
                    for reference_sample in raw_index[reference_key]['samples']:
                        differences = numerical(native_sample, reference_sample, case['atoms'])
                        cross_errors.append(differences)
                        qualified &= all(value <= protocol['numerical_gates'][name] for name, value in differences.items())
            ratio = reference_row['warm_median_seconds'] / native_row['warm_median_seconds'] if qualified else None
            pairs.append({'pair_index': pair, 'qualified': qualified, 'reference_over_native_warm_median': ratio, 'all16_cross_engine_EF_errors': cross_errors})
        qualified = all(pair['qualified'] for pair in pairs)
        ratios = [pair['reference_over_native_warm_median'] for pair in pairs] if qualified else []
        median = statistics.median(ratios) if qualified else None
        win = bool(qualified and median > 1.05 and all(ratio > 1 for ratio in ratios))
        cases.append({'case_id': case['id'], 'method': case['method'], 'atoms': case['atoms'], 'mode': case['route'], 'qualified': qualified, 'paired_ratios': ratios, 'median_three_paired_ratio': median, 'win': win, 'pairs': pairs})
    terminal_success = campaign['status'] == 'TERMINAL_180_FRESH_PROCESS_ATTEMPTS_ALL_PARENT_QUALIFIED_REPLAY_PENDING' and 'error' not in campaign
    all_science = bool(terminal_success and len(rows) == 180 and returned == 720 and all(row['status'] == 'QUALIFIED' for row in rows) and len(cases) == 30 and all(case['qualified'] for case in cases))
    geomean = math.exp(math.fsum(math.log(case['median_three_paired_ratio']) for case in cases) / 30) if all_science else None
    wins = sum(case['win'] for case in cases)
    accepted = bool(all_science and wins >= 20 and geomean > 1)
    emit({'status': 'PASS_FIXED30_SCIENCE_AND_THREE_PAIR_MAJORITY' if accepted else ('COMPLETE_FIXED30_SCIENCE_PERFORMANCE_TARGET_NOT_MET' if all_science else 'FINAL_FAILED_OR_INCOMPLETE_ALL_ORIGINALS_RETAINED'),
          'source_revision': SOURCE, 'majority_accepted': accepted, 'all30_scientifically_qualified': all_science, 'physical_cases': 30, 'wins': wins, 'required_wins': 20, 'complete_set_geomean': geomean,
          'attempted_populations': len(rows), 'qualified_populations': sum(row['status'] == 'QUALIFIED' for row in rows), 'returned_EF': returned, 'terminal_parent_accepted': terminal_success, 'parent_error': campaign.get('error'),
          'protocol_sha256': sha(protocol_path), 'campaign_sha256': sha(out / 'campaign.json'), 'manifest_sha256': sha(manifest_path), 'build_receipt_sha256': sha(build_path), 'core_sha256': build['artifacts']['libgenerativeqc.so'],
          'screening_preregistration_sha256':sha(TASK/'screening-protocol-v3.json'), 'qualifier_sha256': sha(qualifier_path), 'device': campaign['device'], 'dispatch_admission_status': dispatch['status'], 'rows': rows, 'cases': cases,
          'scope': protocol['scope']})


def guarded_main():
    result=TASK/(COHORT+'-local-replay.json')
    check(not result.exists(),'inspect existing final replay before retry')
    try:main()
    except BaseException as error:
        if not result.exists():
            emit({'status':'FAIL_FINAL_INDEPENDENT_REPLAY_ALL_ORIGINALS_RETAINED','source_revision':SOURCE,'majority_accepted':False,'reader':Path(__file__).name,'reader_sha256':sha(Path(__file__)),'error':type(error).__name__+': '+str(error),'scope':'No scientific or performance acceptance. All originals remain unchanged.'})
        raise
if __name__=='__main__':guarded_main()
