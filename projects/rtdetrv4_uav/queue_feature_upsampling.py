"""Finite sequential three-arm experiment. No recurring scheduler or old queue."""
from bootstrap import ROOT,write_json,sha256
import json,subprocess,sys,time,traceback
import argparse,ctypes
from pathlib import Path

OUT=ROOT/'results'/'feature_upsampling_comparison_20261010'
SCRIPT=Path(__file__).with_name('run_feature_upsampling.py')
def process_alive(pid):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong]
    kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:
        if ctypes.get_last_error()==87: return False
        raise OSError(ctypes.get_last_error(),'Cannot inspect prior GPU process')
    try:
        code=ctypes.c_ulong()
        if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)): raise ctypes.WinError()
        return code.value==259
    finally: kernel.CloseHandle(handle)

def main(wait_pids, suffix='20261010'):
    global OUT
    OUT=ROOT/'results'/f'feature_upsampling_comparison_{suffix}'
    OUT.mkdir(exist_ok=False)
    started=time.time()
    status=dict(status='starting',arms=['anyup','bilinear','none'],epochs_per_arm=3,total_additional_epochs=9,runner_sha256=sha256(SCRIPT),wait_pids=wait_pids,results={})
    write_json(OUT/'status.json',status)
    try:
        while any(process_alive(pid) for pid in wait_pids):
            status.update(status='waiting_for_existing_gpu_job',elapsed_seconds=time.time()-started)
            write_json(OUT/'status.json',status)
            if (OUT/'STOP').exists(): raise InterruptedError('Comparison STOP requested')
            if time.time()-started>72*3600: raise TimeoutError('Finite queue exceeded 72 hours')
            time.sleep(15)
        for mode in status['arms']:
            if (OUT/'STOP').exists(): raise InterruptedError('Comparison STOP requested')
            name=f'feature_upsampling_{mode}_{suffix}'
            status.update(status='running',current_arm=mode)
            write_json(OUT/'status.json',status)
            with (OUT/f'{mode}_stdout.log').open('w',encoding='utf-8') as log, (OUT/f'{mode}_stderr.log').open('w',encoding='utf-8') as error:
                child=subprocess.Popen([sys.executable,'-B',str(SCRIPT),'--mode',mode,'--name',name,'--epochs','3'],cwd=ROOT,stdout=log,stderr=error)
                status['child_pid']=child.pid; write_json(OUT/'status.json',status)
                while child.poll() is None:
                    if (OUT/'STOP').exists():
                        # Request run-local graceful stop before the next training sample.
                        child_out=ROOT/'results'/name; child_out.mkdir(exist_ok=True)
                        (child_out/'STOP').write_text('Comparison stop requested',encoding='utf-8')
                    if time.time()-started>72*3600:
                        child.terminate(); child.wait(timeout=60); raise TimeoutError('Finite comparison exceeded 72 hours')
                    time.sleep(5)
                if child.returncode: raise RuntimeError(f'{mode} failed; see its stderr log')
            result=json.loads((ROOT/'results'/name/'status.json').read_text())
            if result['status']!='completed': raise RuntimeError('Incomplete arm')
            status['results'][mode]=result['metrics']; write_json(OUT/'status.json',status)
        a,b,n=[status['results'][mode] for mode in ('anyup','bilinear','none')]
        passed=a['AP']>=b['AP']+.005 and a['APsmall']>=b['APsmall']+.01 and a['AP']>=n['AP']
        status.update(status='completed',decision='advance_to_confirmation' if passed else 'stop_current_transfer',AP_gain_vs_bilinear_pp=100*(a['AP']-b['AP']),APsmall_gain_vs_bilinear_pp=100*(a['APsmall']-b['APsmall']),AP_gain_vs_no_kd_pp=100*(a['AP']-n['AP']),elapsed_seconds=time.time()-started,next_action='Confirm independent seed/split before fusion or paper claims' if passed else 'Move to the next verified literature mechanism; do not expand failed variant')
        write_json(OUT/'status.json',status)
    except Exception as e:
        status.update(status='failed',error=str(e),traceback=traceback.format_exc()); write_json(OUT/'status.json',status); raise
if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--wait-pids',type=int,nargs='*',default=[])
    parser.add_argument('--suffix',default='20261010')
    args=parser.parse_args()
    main(args.wait_pids,args.suffix)
