"""Integrate read-only system-rail samples over actual application requests."""
from pathlib import Path
import hashlib, json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
root=ROOT/'results/real_app/power'
read=lambda n:json.loads((root/n).read_text())
meta=read('metadata.json');assert meta['exit']==0
assert meta['charging_policy_changed'] is False
assert meta['charging_before']['batt_slate_mode']==meta['charging_after']['batt_slate_mode']
assert meta['screen_off_before'] is True
assert meta['screen_off_after'] is True
proof=json.loads((ROOT/'results/real_app/proof.json').read_text())
for name in ['dinov3_embed_server','dinov3_conv_native.bin','dinov3_gpu_adaptive.bin']:
    assert meta['files_sha256'][name]==proof['model_sha256'][name]
assert meta['files_sha256']['api.py']==proof['api_sha256']
assert meta['files_sha256']['pipeline.py']==proof['pipeline_sha256']
rows=[json.loads(s) for s in (root/'telemetry.jsonl').read_text().splitlines()]
valid=[r for r in rows if r['system_rail_w'] is not None]
assert len(valid)>=len(rows)*.95
t=np.array([r['time_s'] for r in valid]);p=np.array([r['system_rail_w'] for r in valid])
assert np.all(np.diff(t)>0)
events=[json.loads(s) for s in (root/'events.jsonl').read_text().splitlines()]
work=read('workload_results.json');assert len(work)==4

def integral(start,end):
    assert t[0]<=start<end<=t[-1]
    inside=(t>start)&(t<end)
    ts=np.r_[start,t[inside],end]
    ps=np.interp(ts,t,p)
    return float(np.trapezoid(ps,ts))

def marker(job,kind):
    return next(e['time_s'] for e in events if e.get('job')==job and e['event']==kind)

jobs=[]
for w in work:
    job=w['job'];start=w['start_s'];end=w['end_s'];duration=end-start
    energy=integral(start,end)
    segments=[(marker(job,'idle_before_start')+1,marker(job,'idle_before_end')-1),
              (marker(job,'idle_after_start')+1,marker(job,'idle_after_end')-1)]
    idle=sum(integral(a,b) for a,b in segments)/sum(b-a for a,b in segments)
    samples=[r for r in valid if start<=r['time_s']<=end]
    assert len(samples)>=duration # approximately 2 Hz, allow at most 50% missing
    thermals={}
    for name in ['cpuss-0-usr','gpuss-0-usr']:
        values=[r['thermals_mc'][name]/1000 for r in samples if r['thermals_mc'].get(name) is not None]
        if values:thermals[name]=[min(values),max(values)]
    jobs.append({'job':job,'backend':w['backend'],'case_id':w['case_id'],
        'duration_s':duration,'windows':w['windows'],'power_samples':len(samples),
        'mean_system_w':energy/duration,'gross_j_per_screenshot':energy,
        'loaded_idle_w':idle,'incremental_j_per_screenshot':energy-idle*duration,
        'incremental_mean_w':energy/duration-idle,
        'battery_temp_min_c':min(r['battery_temp_tenths_c'] for r in samples)/10,
        'battery_temp_max_c':max(r['battery_temp_tenths_c'] for r in samples)/10,
        'thermal_sensor_ranges_c':thermals})

backends={}
for backend in ['gpu16s','cpu']:
    group=[j for j in jobs if j['backend']==backend]
    duration=sum(j['duration_s'] for j in group)
    energy=sum(j['gross_j_per_screenshot'] for j in group)
    incremental=sum(j['incremental_j_per_screenshot'] for j in group)
    backends[backend]={'runs':len(group),'screenshots':len(group),'windows':sum(j['windows'] for j in group),
        'mean_system_w':energy/duration,'mean_duration_s':duration/len(group),
        'gross_j_per_screenshot':energy/len(group),
        'incremental_j_per_screenshot':incremental/len(group),
        'mean_loaded_idle_w':sum(j['loaded_idle_w']*j['duration_s'] for j in group)/duration,
        'battery_temp_max_c':max(j['battery_temp_max_c'] for j in group),
        'run_mean_w_range':[min(j['mean_system_w'] for j in group),max(j['mean_system_w'] for j in group)]}
g=backends['gpu16s'];c=backends['cpu']
summary={'backends':backends,'jobs':jobs,'gpu_vs_cpu':{
    'mean_power_reduction':1-g['mean_system_w']/c['mean_system_w'],
    'gross_energy_reduction':1-g['gross_j_per_screenshot']/c['gross_j_per_screenshot'],
    'incremental_energy_reduction':1-g['incremental_j_per_screenshot']/c['incremental_j_per_screenshot'],
    'speedup':c['mean_duration_s']/g['mean_duration_s']},
    'telemetry':{'total_samples':len(rows),'valid_samples':len(valid),
        'sample_interval_p50_s':float(np.median(np.diff(t))),'sample_interval_max_s':float(np.max(np.diff(t))),
        'voltage_mv_range':[min(r['vsys_mv'] for r in valid),max(r['vsys_mv'] for r in valid)],
        'battery_statuses':sorted({r['status'] for r in rows}),
        'usb_online_values':sorted({r['usb_online'] for r in rows})},
    'method':'System DC-rail estimate ISYS*VSYS with trapezoid integration, not chip-only, USB input or wall power. Two images/backend, GPU0-CPU0-CPU1-GPU1 order. Same actual OmniMatch /detect requests with 120 windows/image. Model warmup/loading and idle intervals excluded from active energy.',
    'limitations':['On-board telemetry, not calibrated external power meter',
        'Only two images per backend, dynamic clocks; not long-run thermal qualification',
        'Idle subtraction approximates workload incremental system energy; not GPU/CPU die power',
        'Known application localization failures remain; this measures running the existing workload'],
    'sha256':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in
        ['metadata.json','events.jsonl','telemetry.jsonl','workload_results.json']}}
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
