"""Combine exact completed components; never relabel the original failed attempt."""
from pathlib import Path
import hashlib
import importlib.util
import json

TASK=Path(__file__).resolve().parent

def check(v,m):
    if not v:raise ValueError(m)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def load(name):
    spec=importlib.util.spec_from_file_location(name.replace('-','_'),TASK/name);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    original=load('validate-spatial-components-v1.py').validate()
    race=load('validate-spatial-racecheck-v1.py').validate()
    original_path=TASK/'spatial-v1-components-local-replay-v1.json';race_path=TASK/'spatial-racecheck-v2-local-replay-v1.json'
    check(original==read(original_path) and race==read(race_path),'Recomputed component proof differs')
    check(original['status']=='PASS_CURRENT_SOURCE_NORMAL6_AND_UNFILTERED_MEMCHECK6_COMPONENTS_ONLY' and original['original_platform_status_retained']=='FAILED' and original['racecheck_accepted'] is False,'Failed old race/whole attempt promoted')
    check(race['status']=='PASS_CURRENT_SOURCE_UNFILTERED_RACECHECK6_AND_POSITIVE_WORK','Full unfiltered race6 did not pass')
    build=read(TASK/'production-build-v1/build-receipt.json');protocol=read(TASK/'spatial-protocol-v1.json');campaign=read(TASK/'spatial-racecheck-v2/campaign.json')
    check(original['source_revision']==race['source_revision']==build['source_revision']==protocol['source_revision']=='db3657f08d983791c45b7d949cb9028fe0055e3f' and original['source_tree']==race['source_tree']==build['source_tree']==protocol['source_tree'],'Mixed source/tree components')
    check(original['core_sha256']==build['artifacts']['libgenerativeqc.so'] and campaign['artifacts']==build['artifacts'] and campaign['admissions']['spatial-v1-components-local-replay-v1.json']==sha(original_path)==sha(TASK/'spatial-racecheck-v2-available-admissions/spatial-v1-components-local-replay-v1.json'),'Cross-attempt package/component binding differs')
    check(set(original['summaries'])=={'normal','memcheck'} and set(race['summaries'])=={'racecheck'},'Duplicate/missing accepted modes')
    summaries={**original['summaries'],**race['summaries']};nodes=protocol['expected_nodeids']['production']
    check(all([v['nodeid'] for v in rows]==nodes and len(rows)==6 for rows in summaries.values()),'Original six-case populations changed')
    check(original['energy_only_scf']==12 and original['source_exports']==36 and race['energy_only_scf']==6 and race['source_exports']==18 and original['full_EF']==race['full_EF']==original['clean_timing_votes']==race['clean_timing_votes']==0,'Accepted work/EF/vote counts differ')
    result={'status':'PASS_CURRENT_SOURCE_GROUPED_DIRECT_SIX_ALL_MODES_AND_POSITIVE_WORK','source_revision':build['source_revision'],'source_tree':build['source_tree'],'core_sha256':build['artifacts']['libgenerativeqc.so'],'summaries':summaries,'energy_only_scf':18,'source_exports':54,'full_EF':0,'clean_timing_votes':0,'all_racecheck_processes_unfiltered':True,'accepted_modes_have_complete_child_device_after':True,'original_overall_failure_retained':True,'original_partial_race_numeric_cases_accepted':0,'components':{original_path.name:sha(original_path),race_path.name:sha(race_path)},'original_failed_manifest_sha256':sha(TASK/'spatial-v1-download-manifest.json'),'race_success_manifest_sha256':sha(TASK/'spatial-racecheck-v2-download-manifest.json'),'readers':{n:sha(TASK/n) for n in ['replay-spatial-composite-v1.py','validate-spatial-components-v1.py','validate-spatial-racecheck-v1.py','replay-spatial-v2.py']},'scope':'Original failed attempt retained. Accept only complete originalnormal6/unfilteredmemcheck6 plus separate complete unfilteredrace6 on exactsameproduction33/source. Allsix originalcase assertions andpositivework remain. No filteredrace, partialracepromotion, fullEF orperformancevotes. Diagnostic stages may use distinct physicalH100s; finalpairedperformance muststilluseonephysicalH100.'}
    with (TASK/'spatial-composite-v1-local-replay.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='summaries'}))
if __name__=='__main__':main()
