"""Independent allocation-interval arithmetic, retained from audited recipe."""
import hashlib,json
from importlib.util import spec_from_file_location,module_from_spec
from pathlib import Path
spec=spec_from_file_location("cohort_math",Path(__file__).with_name("cohort-replay-v1.py"));common=module_from_spec(spec);spec.loader.exec_module(common)
check=common.check
integer=common.integer
def replay_plan(plan):
    """Use raw selected allocation intervals, not producer peak summaries."""
    unhashed = {key: value for key, value in plan.items() if key != 'identity'}
    check(hashlib.sha256(json.dumps(unhashed, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest() == plan['identity'], 'resource plan identity differs')
    budget = plan['budget']
    check(budget['host_reserve_bytes'] == budget['device_reserve_bytes'] == 0 and budget['headroom_fraction'] == 0 and budget['pinned_host_bytes'] is None and budget['schema_version'] == 1, 'undeclared reserve or pinned constraint')
    caps=budget['per_device_bytes']
    check(all(isinstance(pair,list) and len(pair)==2 and integer(pair[0]) and pair[0]==0 and integer(pair[1]) for pair in caps) and len(caps)==len({pair[0] for pair in caps}),'duplicate or invalid device caps')
    check(all(integer(v) for key in ['peak_bytes','resident_bytes'] for v in plan[key].values()),'invalid integer capacity summaries')
    limits = {name: budget[field] for name, field in [('host', 'host_bytes'), ('device', 'device_bytes')] if budget[field] is not None}
    limits.update({'device:' + str(device): limit for device, limit in budget['per_device_bytes']})
    check(plan['limits'] == limits and all(integer(value) for value in limits.values()), 'resource limits differ')
    requests=plan['requests']
    names=[request['name'] for request in requests]
    check(len(names)==len(set(names)) and all(isinstance(n,str) and n for n in names),'duplicate/invalid resource request names')
    for request in requests:
        choices=[candidate['name'] for candidate in request['candidates']]
        check(choices and len(choices)==len(set(choices)),'empty/duplicate provider candidates')
    selections = dict(plan['selections'])
    check(len(selections) == len(plan['selections']), 'duplicate provider selection')
    check(set(selections)<=set(names),'selection names unknown resource request')
    check(plan['status'] in ['feasible','infeasible'],'unrecognized resource plan status')
    allocations = []
    for request in plan['requests']:
        if request['name'] not in selections:
            check(plan['status'] != 'feasible', 'feasible plan omits provider')
            continue
        matches = [candidate for candidate in request['candidates'] if candidate['name'] == selections[request['name']]]
        check(len(matches) == 1, 'ambiguous provider alternative')
        allocations.extend(matches[0]['estimates'])
    for allocation in allocations:
        check(allocation['space'] in ['pinned','pageable','device:0'] and allocation['kind'] in ['persistent','workspace','cache','library','runtime','output'],'unrecognized owned allocation space/kind')
        check(all(integer(allocation[field]) for field in ['bytes', 'first_phase', 'last_phase', 'streamed_bytes']) and allocation['first_phase'] <= allocation['last_phase'], 'invalid allocation capacity or interval')

    def peaks(resident):
        entries = [entry for entry in allocations if not resident or entry['kind'] != 'workspace']
        observed = {'host': 0, 'device': 0, 'pinned': 0, 'pageable': 0}
        # All interval starts suffice: a maximum cannot first occur at a removal.
        for phase in sorted({entry['first_phase'] for entry in entries}):
            values = {entry['space']: sum(row['bytes'] for row in entries if row['space'] == entry['space'] and row['first_phase'] <= phase <= row['last_phase']) for entry in entries}
            values['host'] = values.get('pinned', 0) + values.get('pageable', 0)
            values['device'] = sum(value for space, value in values.items() if space.startswith('device:'))
            for space, value in values.items():
                check(integer(value),'aggregate owned capacity exceeds int64')
                if value > observed.get(space, 0):
                    observed[space] = value
        return observed

    check(peaks(False) == plan['peak_bytes'] and peaks(True) == plan['resident_bytes'], 'independent lifetime capacity accounting differs')
    if plan['status'] == 'feasible':
        check(allocations and all(plan['peak_bytes'].get(space, 0) <= limit for space, limit in limits.items()), 'feasible declared capacities exceed limits')
    return plan['peak_bytes']
