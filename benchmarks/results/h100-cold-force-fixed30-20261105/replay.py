"""Offline H100 cold evidence readback, using pinned local Git helper objects."""
from pathlib import Path
import argparse
import base64
import hashlib
import io
import json
import lzma
import os
import subprocess
import sys
import tarfile
import tempfile

BUNDLE=Path(__file__).resolve().parent


def check(value,message):
    if not value:raise ValueError(message)


def sha(data):return hashlib.sha256(data).hexdigest()
def read(path):return json.loads(path.read_text())


def compact_fixed30(value):
    return {**{k:value[k] for k in ['status','scope','source_revision','native_source_identity','core_sha256','device_uuid','campaign_sha256','protocol_sha256','attempted_populations','qualified_populations','returned_EF','science_pass','image_provenance','summary','limitation']},
            'cases':[{'id':c['id'],'method':c['method'],'atoms':c['atoms'],'mode':c['route'],
                      'science_pass':c['science_pass'],'phase':c['phase'],
                      'pair_timings':[{'pair_index':p['pair_index'],'qualified':p['qualified'],
                          'native':{phase:p['native']['timing'][phase]['median'] for phase in ['cold','warm']} if p['native'] else None,
                          'reference':{phase:p['reference']['timing'][phase]['median'] for phase in ['cold','warm']} if p['reference'] else None,
                          'cold_native_work':p['native']['work'][0] if p['native'] and p['native']['work'] else None,
                          'cold_reference_work':p['reference']['work'][0] if p['reference'] and p['reference']['work'] else None} for p in c['pairs']]} for c in value['cases']],
            'failures':value['failures']}


def compact_main(value):
    return {**{k:value[k] for k in ['status','scope','source_revision','native_source_identity','core_sha256','device_uuid','campaign_sha256','qualified_populations','returned_EF','science_pass','performance_pass','image_provenance']},
            'studies':[{k:s[k] for k in ['id','producer_status','all_declared_pairs_qualified','summary','pairs']} for s in value['studies']]}


def pinned_source(repository,revision,isolated,env):
    """Materialize pinned offline import dependencies, without a build or solver."""
    isolated.mkdir()
    subprocess.run(['git','init','-q',str(isolated)],env=env,check=True,capture_output=True)
    inventory=subprocess.check_output(['git','-C',str(repository),'ls-tree','-r','--name-only',revision,'--','python','data','benchmarks','manifests','cmake/GenerativeQCSourceIdentity.json'],env=env,text=True).splitlines()
    names=[n for n in inventory if n.startswith(('python/','data/','manifests/')) or n=='cmake/GenerativeQCSourceIdentity.json' or (n.startswith('benchmarks/') and '/results/' not in n and n.endswith('.py'))]
    check(names and all(not any(p.startswith('.env') or p in ['..','__pycache__'] for p in Path(n).parts) for n in names),'Unsafe pinned source inventory')
    archive=subprocess.check_output(['git','-C',str(repository),'archive',revision,'--',*names],env=env)
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        members=stream.getmembers()
        check(all((m.isfile() or m.isdir()) and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members),'Unsafe pinned helper member')
        stream.extractall(isolated,filter='data')
    return isolated


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository',type=Path,default=BUNDLE.parents[2])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    check(not sys.flags.optimize and (args.output is None or not args.output.exists()),'Use unoptimized Python and preserve output')
    cohort=read(BUNDLE/'cohort.json')
    sys.path.insert(0,str(args.repository))
    from benchmarks._exact_fp64_evidence import load as load_exact
    directory=cohort['records']['exact_storage']
    check(Path(directory).name==directory,'Unsafe exact storage directory')
    _,decoded=load_exact(BUNDLE/directory)
    raw=json.dumps(json.loads(decoded['canonical-container.json']),ensure_ascii=False,separators=(',',':')).encode()
    size=cohort['records']['decoded_bytes'];check(type(size) is int and 0<size<=256<<20,'Unexpected decoded size')
    check(len(raw)==size and sha(raw)==cohort['records']['decoded_sha256'],'Decoded container differs')
    payload=json.loads(raw);check(payload['schema']=='generativeqc.h100-cold-receipt-files.v2','Unknown receipt container')
    entries=payload['files'];blobs=payload['blobs']
    check(len(entries)==cohort['records']['files'] and len({e['path'] for e in entries})==len(entries) and {e['sha256'] for e in entries}==set(blobs),'Member/blob population differs')
    values={}
    for digest,blob in blobs.items():
        check(blob['encoding'] in ['utf8','base64'],'Unknown byte encoding')
        data=blob['data'].encode('utf8') if blob['encoding']=='utf8' else base64.b64decode(blob['data'],validate=True)
        check(len(data)==blob['bytes'] and sha(data)==digest,'Original receipt blob differs')
        values[digest]=data
    with tempfile.TemporaryDirectory(prefix='gqc-h100-cold-public-') as name:
        task=Path(name)
        for entry in entries:
            relative=Path(entry['path'])
            check(not relative.is_absolute() and all(p not in ['', '.', '..'] for p in entry['path'].split('/')) and '\\' not in entry['path'] and ':' not in entry['path'],'Unsafe receipt path')
            data=values[entry['sha256']];check(len(data)==entry['bytes'],'Member size differs')
            target=task/'receipts'/relative;target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as output:output.write(data)
        common=subprocess.check_output(['git','-C',str(args.repository),'rev-parse','--path-format=absolute','--git-common-dir'],text=True).strip()
        env={**os.environ,'GIT_NO_LAZY_FETCH':'1','GIT_ALLOW_PROTOCOL':'','GIT_TERMINAL_PROMPT':'0',
             'GIT_ALTERNATE_OBJECT_DIRECTORIES':str(Path(common)/'objects'),
             'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1'}
        results={};helpers={}
        jobs=[('outer','analyze-wb-study-v1.py'),('anchor','analyze-wb-study-v1.py'),('work','analyze-wb-study-v2.py'),('hf-streaming','analyze-wb-study-v3.py'),('full-cache','analyze-wb-study-v3.py'),('fixed30','analyze-fixed30-v1.py'),('main-integration','analyze-main-integration-v1.py')]
        for group,script in jobs:
            root=task/'receipts'/group
            build=read(root/'base'/('build-receipts-v4' if group=='fixed30' else 'build-receipts-v6' if group=='main-integration' else 'build-receipts-v3')/'build.json')
            revision=build['source']['revision']
            if revision not in helpers:
                helpers[revision]=pinned_source(args.repository,revision,task/('helpers-'+revision),env)
            output=task/(group+'-analysis.json')
            argv=[sys.executable,str(task/'receipts/metadata'/script),'--task',str(root),'--repository',str(helpers[revision]),'--output',str(output)]
            if group not in ['fixed30','main-integration']:argv+=['--study',group]
            else:argv+=['--reference-catalog',str(task/'receipts/metadata/reference-installation-history-v1.json')]
            result=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=180)
            check(result.returncode==0,'Offline reduction failed: '+group+'\n'+result.stderr[-3000:])
            value=read(output)
            original=read(task/'receipts/metadata'/('fixed30-analysis-v1.json' if group=='fixed30' else group+'-analysis-v1.json'))
            check(value==original,'Public requalification differs from sealed local reduction: '+group)
            results[group]=compact_fixed30(value) if group=='fixed30' else compact_main(value) if group=='main-integration' else {**{k:value[k] for k in ['status','source_revision','native_source_identity','core_sha256','device_uuid','qualified_populations','returned_EF','science_pass','performance_pass']},'studies':[{k:s[k] for k in ['id','producer_status','all_declared_pairs_qualified','summary']} for s in value['studies']]}
        expected=json.loads(lzma.decompress((BUNDLE/'summary.json.xz').read_bytes()))
        check(results==expected,'Published compact summary differs from full raw reduction')
        if args.output:
            with args.output.open('x') as output:json.dump(results,output,indent=2,allow_nan=False);output.write('\n')
        print(json.dumps({'status':'PASS_ALL_ORIGINAL_BYTES_AND_OFFLINE_SCIENCE_REDUCTIONS','files':len(entries),'distinct_blobs':len(blobs),'fixed30':{k:results['fixed30'][k] for k in ['status','source_revision','qualified_populations','returned_EF','science_pass','summary']}}))


if __name__=='__main__':main()
