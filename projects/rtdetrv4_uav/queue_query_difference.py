"""Finite, resumable four-arm pilot; preserves the running CODrone job."""
from pathlib import Path
import argparse, subprocess, sys, time, json, traceback
from bootstrap import ROOT,write_json,sha256
from queue_feature_upsampling import process_alive


def main(args):
    out=ROOT/'results'/'querydiff_core_pilot_20261011';out.mkdir(exist_ok=True)
    runner=Path(__file__).with_name('run_query_difference.py')
    status=dict(status='waiting_for_existing_gpu_job',wait_pids=args.wait_pids,
                arms=['B','Q','M','A2'],epochs_per_arm=3,fit_images=128,development_images=32,
                conditions=['clean','down2','down4','blur1','blur2'],
                result_scope='finite pilot, no full-training or publication-performance claim',results={})
    start=time.time();write_json(out/'status.json',status)
    try:
        while any(process_alive(pid) for pid in args.wait_pids):
            if (out/'STOP').exists():raise InterruptedError('Queue STOP requested')
            status.update(elapsed_seconds=time.time()-start);write_json(out/'status.json',status);time.sleep(15)
        for arm in status['arms']:
            name=f'querydiff_pilot_{arm}_20261011';run_dir=ROOT/'results'/name
            if run_dir.exists():
                old=json.loads((run_dir/'status.json').read_text())
                if old['status']!='completed':raise RuntimeError(f'Incomplete prior {arm}; preserve files for recovery')
                status['results'][arm]=old['conditions'];continue
            if (out/'STOP').exists():raise InterruptedError('Queue STOP requested')
            with (out/f'{arm}_stdout.log').open('w',encoding='utf-8') as log,(out/f'{arm}_stderr.log').open('w',encoding='utf-8') as error:
                child=subprocess.Popen([sys.executable,'-u','-B',str(runner),'--mode',arm,'--name',name,'--conditions'],cwd=ROOT,stdout=log,stderr=error)
                status.update(status='running',current_arm=arm,child_pid=child.pid);write_json(out/'status.json',status)
                while child.poll() is None:
                    if (out/'STOP').exists() and run_dir.exists():(run_dir/'STOP').write_text('Queue STOP requested')
                    time.sleep(5)
                if child.returncode:raise RuntimeError(f'{arm} failed; see stderr log')
            completed=json.loads((run_dir/'status.json').read_text())
            status['results'][arm]=completed['conditions'];write_json(out/'status.json',status)
        results=status['results'];gain_q=results['M']['clean']['AP']-results['Q']['clean']['AP']
        gain_uniform=results['M']['clean']['AP']-results['A2']['clean']['AP']
        # Toolkit emits percentage units, not fractions. Pilot gate is not significance.
        passed=gain_q>=.5 and gain_uniform>=.3 and results['M']['clean']['AP']>=results['B']['clean']['AP']
        status.update(status='completed',gain_vs_Q_pp=gain_q,gain_vs_uniform_pp=gain_uniform,
            decision='advance_to_full_training' if passed else 'stop_current_implementation_expansion',
            elapsed_seconds=time.time()-start)
        write_json(out/'status.json',status)
        # Populate a traceable data artifact; manuscript formatting is a separate render-gated step.
        write_json(ROOT/'paper/querydiff_measured_pilot_tables_20261011.json',status)
    except Exception as error:
        status.update(status='failed',error=str(error),traceback=traceback.format_exc());write_json(out/'status.json',status);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait-pids',nargs='*',type=int,default=[])
    main(p.parse_args())
