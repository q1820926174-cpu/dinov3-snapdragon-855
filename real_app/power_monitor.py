"""Read-only Samsung/MAX77705 system-rail telemetry, with no power policy writes."""
from pathlib import Path
import hashlib, json, os, re, signal, subprocess, threading, time

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'power';OUT.mkdir(exist_ok=True)
BAT=Path('/sys/class/power_supply/battery')
stop=threading.Event();overheat=threading.Event()
def read(p):
    try:return Path(p).read_text().strip()
    except OSError:return None
def number(p):
    try:return int(read(p))
    except (ValueError,TypeError):return None
thermal={}
for d in Path('/sys/class/thermal').glob('thermal_zone*'):
    name=read(d/'type')
    if name in ['gpuss-0-usr','cpuss-0-usr','npu-usr']:thermal[name]=d/'temp'

def monitor():
    cache={};last_regs=0
    with (OUT/'telemetry.jsonl').open('w',buffering=1) as f:
        while not stop.is_set():
            start=time.monotonic()
            if start-last_regs>=5:
                text=read('/sys/class/power_supply/max77705-fuelgauge/fg_data') or ''
                regs={int(k,16):int(v,16) for k,v in re.findall(r'0x([0-9a-f]+):\s*0x([0-9a-f]+)',text,re.I)}
                if 0xb1 in regs:
                    cache={'vsys_mv':regs[0xb1]*15625//100000,'vsys_raw':regs[0xb1],
                           'register_time_s':time.monotonic()}
                last_regs=start
            isys=number(BAT/'power_now');vsys=cache.get('vsys_mv')
            row={'time_s':time.monotonic(),'isys_ma':isys,**cache,
                'system_rail_w':isys*vsys/1e6 if isys is not None and vsys is not None and 3000<vsys<5500 and isys>=0 else None,
                'battery_current_ua':number(BAT/'batt_current_ua_now'),
                'battery_uv':number(BAT/'voltage_now'),'battery_temp_tenths_c':number(BAT/'temp'),
                'capacity':number(BAT/'capacity'),'status':read(BAT/'status'),
                'usb_online':number('/sys/class/power_supply/usb/online'),
                'gpu_hz':number('/sys/class/kgsl/kgsl-3d0/gpuclk'),
                'thermals_mc':{name:number(p) for name,p in thermal.items()}}
            row['read_ms']=(time.monotonic()-start)*1000
            f.write(json.dumps(row)+'\n')
            if row['battery_temp_tenths_c'] is not None and row['battery_temp_tenths_c']>=450:
                overheat.set()
            stop.wait(max(0,.5-(time.monotonic()-start)))

metadata={'measurement':'MAX77705 ISYS mA via battery/power_now, VSYS mV via read-only fg_data 0xB1; W=ISYS*VSYS/1e6',
    'basis':'Existing audited phone tooling: scv41-linux/tools/hybrid-power-run.py',
    'sample_hz':2,'vsys_refresh_seconds':5,'charging_policy_changed':False,
    'screen_before':read('/run/scv41-screen-power/state.json'),
    'charging_before':{n:read(BAT/n) for n in ['status','batt_slate_mode','capacity']},
    'files_sha256':{str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [ROOT/'power_worker.py',ROOT/'power_app_adapter.py',ROOT/'project/app/api.py',ROOT/'project/app/pipeline.py',
                     ROOT.parent/'dinov3_embed_server',ROOT.parent/'dinov3_conv_native.bin',ROOT.parent/'dinov3_gpu_adaptive.bin']}}
(OUT/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
thread=threading.Thread(target=monitor,daemon=True);thread.start()
child=None;code=1
try:
    time.sleep(2)
    # Privileged sampler only; actual application and inference retain the dev user.
    with (OUT/'workload.log').open('w') as log:
        child=subprocess.Popen(['runuser','-u','dev','--','python3','-u',str(ROOT/'power_worker.py')],
                    stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy(),start_new_session=True)
        deadline=time.monotonic()+260
        while child.poll() is None:
            if overheat.is_set():raise RuntimeError('battery temperature reached 45C')
            if time.monotonic()>deadline:raise TimeoutError('power workload timeout')
            time.sleep(.5)
        code=child.returncode
finally:
    if child is not None and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=4)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    time.sleep(2);stop.set();thread.join(timeout=5)
    metadata.update(exit=code,screen_after=read('/run/scv41-screen-power/state.json'),
        charging_after={n:read(BAT/n) for n in ['status','batt_slate_mode','capacity']})
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('POWER_COMPLETE',code,flush=True)
raise SystemExit(code)
