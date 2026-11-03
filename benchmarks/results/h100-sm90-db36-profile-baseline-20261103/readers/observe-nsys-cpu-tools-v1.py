from pathlib import Path
import hashlib,json,subprocess,datetime
p=Path(__file__).resolve().parent;out=p/'nsys-cpu-tool-observation-v1.json';assert not out.exists()
root=Path('/usr/local/cuda-12.8/NsightSystems-cli-2025.1.1');tools={'nsys':root/'target-linux-x64/nsys','importer':root/'host-linux-x64/QdstrmImporter'}
r={'status':'ACTUAL_POSTHOC_CPU_TOOL_OBSERVATION','observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'observer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'tools':{n:{'path':str(q),'resolved_path':str(q.resolve()),'bytes':q.stat().st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'version':subprocess.check_output([str(q),'--version'],text=True,timeout=15).strip()} for n,q in tools.items()},'scope':'Authenticated existing CPU bridge read-only posthoc tool observation. Crosschecks actual conversion identity,not a preregistered before-runtime binary witness and not GPU execution.'}
out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
