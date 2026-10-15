"""Run one predeclared, fixed-split E-softmask improvement candidate at a time.

Preparation is CPU-only and allowed while the corrected five-run job runs.
Training requires all five corrected results, fresh independent audit, no other
KOMAP training process, and an exclusive OS-released lock. No unbounded search.
"""
import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
NAMES=('centerline_baseline','e_softmask_geometry','low_lr_context_paste',
       'phase_style_exchange','e_protected_illumination')
CANDIDATES={
    'e_softmask_lr_x0p1': {'ratio':0.1,'encoder_lr':3e-6,'decoder_lr':3e-5},
    'e_softmask_lr_x1over30': {'ratio':1/30,'encoder_lr':1e-6,'decoder_lr':1e-5},
}
SOURCE_FILES=('src/experiment.py','src/notes_recovered.py','src/audit_results.py',
              'src/target_search.py','src/training_awake.py')
TRAINING_SCRIPTS={'experiment.py','notes_recovered.py','target_search.py','recorded_protocol.py'}


def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def canonical(value): return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode('utf-8')
def object_hash(value): return hashlib.sha256(canonical(value)).hexdigest()
def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def save_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)


def require_identical(path,value):
    """A frozen plan is never silently regenerated after code/config changes."""
    path=Path(path)
    if path.exists():
        if read_json(path)!=value:
            raise RuntimeError(f'Frozen plan mismatch (config/split/source changed): {path}. Review the change; do not overwrite a running plan.')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x',encoding='utf-8') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2); stream.write('\n')


def validate_split(split):
    parts=[set(split[name]) for name in ('train','valid','test')]
    if tuple(len(x) for x in parts)!=(72,18,10): raise RuntimeError('Expected frozen72/18/10 split')
    if any(len(split[name])!=len(values) for name,values in zip(('train','valid','test'),parts)):
        raise RuntimeError('Duplicate split IDs')
    if any(parts[i]&parts[j] for i,j in ((0,1),(0,2),(1,2))): raise RuntimeError('Split file IDs overlap')
    train_groups={x.split('_')[0] for x in parts[0]}; valid_groups={x.split('_')[0] for x in parts[1]}
    if len(valid_groups)!=14 or train_groups&valid_groups: raise RuntimeError('A-group split invariants failed')
    if not split.get('requires_group_disjoint') or split.get('original_split_verified') is not False:
        raise RuntimeError('Expected the documented unverified group-split candidate')


def candidate_config(base,candidate):
    required={'encoder_lr':3e-5,'decoder_lr':3e-4,'ema_decay':.999,'weight_decay':.05,
              'patches_per_epoch':600,'batch':8,'max_epochs':100,'patience_epochs':20,
              'min_delta_pp':.02,'input_crop':768,'native_crop':384}
    if any(base.get(key)!=value for key,value in required.items()):
        raise RuntimeError('Base config no longer matches the reviewed corrected protocol')
    result=copy.deepcopy(base)
    for key in ('encoder_lr','decoder_lr'): result[key]=CANDIDATES[candidate][key]
    if {key for key in result if result[key]!=base[key]}!={'encoder_lr','decoder_lr'}:
        raise RuntimeError('Candidate may change only the two learning rates')
    return result


def build_plan(root=ROOT):
    config_path=root/'configs/record_notes.json'; split_path=root/'configs/split_group_fold0.json'
    base=read_json(config_path); split=read_json(split_path); validate_split(split)
    corrected_plan=read_json(root/'notes_recovered/plan.json')
    split_sha=file_hash(split_path)
    if corrected_plan['split_sha256']!=split_sha or corrected_plan['config']!=base:
        raise RuntimeError('Frozen corrected plan differs from current config/split')
    return {'version':1,'purpose':'improvement_not_historical_reproduction',
            'target':{'metric':'final D4 mean-image mIoU percent','threshold':79.0,'validation_images':18,'test_masks':10},
            'base_config':base,'base_config_sha256':file_hash(config_path),'split_file':str(split_path),
            'split_sha256':split_sha,'data_manifest_sha256':file_hash(root/'records/data_manifest.json'),
            'source_sha256':{name:file_hash(root/name) for name in SOURCE_FILES},
            'baseline_checkpoint':str(root/'notes_recovered/runs/centerline_baseline/best.pt'),
            'selection_tile':512,'final_tta_tile':768,'early_stop_enabled':True,
            'original_split_verified':False,
            'rationale':'Historical row71 lower-LR preservation; geometry rows88/96. Only initial encoder/decoder learning rates change; split, scheduler floor and all other settings stay fixed.',
            'candidates':{name:{**values,'config':candidate_config(base,name)} for name,values in CANDIDATES.items()}}


def windows_argv(command):
    if not command: return []
    if os.name!='nt':
        import shlex
        return shlex.split(command)
    import ctypes
    argc=ctypes.c_int()
    shell=ctypes.WinDLL('shell32',use_last_error=True)
    shell.CommandLineToArgvW.argtypes=[ctypes.c_wchar_p,ctypes.POINTER(ctypes.c_int)]
    shell.CommandLineToArgvW.restype=ctypes.POINTER(ctypes.c_wchar_p)
    pointer=shell.CommandLineToArgvW(command,ctypes.byref(argc))
    if not pointer: raise RuntimeError('Cannot parse native process command line')
    try: return [pointer[i] for i in range(argc.value)]
    finally:
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.LocalFree.argtypes=[ctypes.c_void_p]; kernel.LocalFree.restype=ctypes.c_void_p
        kernel.LocalFree(pointer)


def find_training_processes(records,current_pid,root=ROOT):
    by_pid={int(row['ProcessId']):row for row in records}
    ignored={current_pid}; parent=current_pid
    while parent in by_pid:
        parent=int(by_pid[parent]['ParentProcessId'])
        if parent in ignored: break
        ignored.add(parent)
    exact={os.path.normcase(str((root/'src'/name).resolve())) for name in TRAINING_SCRIPTS}
    matches=[]
    for row in records:
        if int(row['ProcessId']) in ignored: continue
        if not row.get('CommandLine'):
            raise RuntimeError(f"Cannot inspect native Python process {row['ProcessId']}; refusing duplicate-risk launch")
        for token in windows_argv(row.get('CommandLine') or '')[1:]:
            path=Path(token)
            if path.name.lower() not in TRAINING_SCRIPTS: continue
            if not path.is_absolute() or os.path.normcase(str(path.resolve())) in exact:
                # Relative script locations cannot be proven unrelated: fail closed.
                matches.append({'pid':row['ProcessId'],'script':token}); break
    return matches


def assert_no_other_training():
    if os.name!='nt': raise RuntimeError('Launch on the native Windows host so active KOMAP processes can be checked')
    command="[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(python|pythonw|py)(\\.exe)?$' } | Select-Object ProcessId,ParentProcessId,CommandLine) | ConvertTo-Json -Compress"
    completed=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',command],
                             capture_output=True,text=True,encoding='utf-8-sig',errors='strict',timeout=30,check=True)
    if not completed.stdout.strip(): raise RuntimeError('Native process inventory unavailable; refusing duplicate-risk launch')
    records=json.loads(completed.stdout)
    if isinstance(records,dict): records=[records]
    matches=find_training_processes(records,os.getpid())
    if matches: raise RuntimeError('Other KOMAP training process still exists: '+json.dumps(matches))


@contextlib.contextmanager
def exclusive_lock(path):
    """Windows releases this byte-range lock even after process termination."""
    if os.name!='nt': raise RuntimeError('Native Windows execution required')
    import msvcrt
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as stream:
        if path.stat().st_size==0: stream.write(b'0'); stream.flush()
        stream.seek(0)
        try: msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        except OSError as error: raise RuntimeError('Another target_search invocation holds the experiment lock') from error
        try: yield
        finally: stream.seek(0); msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)


def require_corrected_complete(root=ROOT):
    marker=root/'notes_recovered/complete.json'
    missing=[name for name in NAMES if not (root/'notes_recovered/runs'/name/'result.json').exists()]
    if missing or not marker.exists() or read_json(marker).get('complete') is not True:
        raise RuntimeError('Wait for all five corrected experiments to finish; missing: '+', '.join(missing))


def audited(directory,split_file):
    import audit_results
    audit_results.main(directory,split_file)
    return {row['variant']:row for row in read_json(directory/'independent_audit.json')}


def target_evidence(result,audit,split_sha):
    if not (audit.get('passed') and audit.get('valid_masks')==18 and audit.get('test_masks')==10 and
            audit.get('split_sha256')==split_sha and result.get('final_tta_tile')==768 and
            result.get('selection_tile')==512 and result.get('model_kind')=='EMA' and
            result.get('weight_decay')==.05 and result.get('ema_decay')==.999):
        raise RuntimeError('Target requires the complete independently audited fixed-protocol D4 result')
    value=audit['independent_mIoU']
    if not math.isfinite(value) or abs(value-result['validation']['mIoU'])>=1e-10:
        raise RuntimeError('Independent score differs from result')
    return {'target_met':value>=79,'mIoU':value,'split_sha256':split_sha,
            'original_split_verified':False,'metric':'final D4 mean-image mIoU percent',
            'valid_masks':18,'test_masks':10,'independent_audit_passed':True}


def shared_baseline(plan,root=ROOT):
    """Freeze only after the corrected five finish, never during prepare-only."""
    baseline=Path(plan['baseline_checkpoint'])
    result_path=root/'notes_recovered/runs/centerline_baseline/result.json'
    result=read_json(result_path)
    shared={'baseline_checkpoint':str(baseline),'baseline_checkpoint_sha256':file_hash(baseline),
            'baseline_result_sha256':file_hash(result_path),'baseline_best_epoch':result['best_epoch'],
            'split_sha256':plan['split_sha256'],'purpose':'one immutable initialization shared by both LR candidates'}
    require_identical(root/'target_search/common_initial_checkpoint.json',shared)
    return shared


def run_candidate(candidate,resume,plan,root=ROOT):
    """Called only after complete/audit/process checks. Exactly one run."""
    output=root/'target_search'; run=output/'runs'/candidate
    baseline=Path(plan['baseline_checkpoint']); config=plan['candidates'][candidate]['config']
    shared=shared_baseline(plan,root)
    provenance={'candidate_id':candidate,'underlying_variant':'e_softmask_geometry',
                'improvement_not_historical_reproduction':True,'config':config,'config_sha256':object_hash(config),
                'baseline_checkpoint':str(baseline),'baseline_checkpoint_sha256':shared['baseline_checkpoint_sha256'],
                'split_sha256':plan['split_sha256'],'source_sha256':plan['source_sha256'],
                'global_plan_sha256':object_hash(plan),'selection_tile':512,'final_tta_tile':768,'early_stop_enabled':True}
    frozen=run/'candidate_plan.json'
    existed=frozen.exists()
    if run.exists() and not existed and any(run.iterdir()): raise RuntimeError('Unrecognized candidate directory; refusing overwrite')
    require_identical(frozen,provenance)
    require_identical(run/'candidate_config.json',config)
    if not (run/'result.json').exists():
        if resume:
            if not (run/'last.pt').exists() or not (run/'config.json').exists():
                raise RuntimeError('--resume requires an existing last.pt and matching saved configuration')
            saved=read_json(run/'config.json')
            expected={**config,'variant':'e_softmask_geometry','initial_checkpoint':str(baseline),
                      'selection_tile':512,'early_stop_enabled':True,'status':'reconstructed, original split/code unavailable'}
            if saved!=expected: raise RuntimeError('Resume config mismatch; refusing checkpoint reuse')
        elif existed and any((run/name).exists() for name in ('config.json','progress.json','last.pt','best.pt')):
            raise RuntimeError('Candidate already started. Resume its last.pt with --resume; do not overwrite it')
        assert_no_other_training()
        if build_plan(root)!=plan: raise RuntimeError('Source/config/split changed immediately before launch')
        import torch
        import experiment as core
        from training_awake import keep_system_awake
        torch.set_num_threads(4); torch.backends.cudnn.benchmark=True
        torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
        data,_=core.load_data(); split=read_json(Path(plan['split_file']))
        with keep_system_awake(output/'power_guard.json'):
            core.run_training('e_softmask_geometry',config,data,split,init=baseline,resume=resume,
                              run_directory=run,selection_tile=512,early_stop=True)
    if build_plan(root)!=plan or file_hash(baseline)!=provenance['baseline_checkpoint_sha256']:
        raise RuntimeError('Experiment inputs changed during execution; result requires review')
    result=read_json(run/'result.json')
    if result.get('variant')!='e_softmask_geometry': raise RuntimeError('Unexpected underlying variant')
    metadata={'candidate_id':candidate,'underlying_variant':'e_softmask_geometry',
              'improvement_not_historical_reproduction':True,'source_notes_recovered':True,
              'original_split_verified':False,'split_file':plan['split_file'],'split_sha256':plan['split_sha256'],
              'group_split_counts':{'train':72,'valid':18,'valid_A_groups':14},
              'candidate_plan_sha256':object_hash(provenance),'config_sha256':object_hash(config),
              'baseline_checkpoint_sha256':provenance['baseline_checkpoint_sha256'],
              'source_sha256':plan['source_sha256']}
    for key,value in metadata.items():
        if key in result and result[key]!=value: raise RuntimeError('Existing result provenance mismatch: '+key)
    result.update(metadata); save_json(run/'result.json',result)
    checks=audited(output/'runs',Path(plan['split_file']))
    evidence=target_evidence(result,checks[candidate],plan['split_sha256'])
    save_json(run/'target_verification.json',evidence)
    save_json(output/'target_status.json',{'candidate_id':candidate,**evidence})
    print(json.dumps({'candidate_id':candidate,**evidence}),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--candidate',choices=tuple(CANDIDATES))
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.resume and not args.candidate: parser.error('--resume requires --candidate')
    plan=build_plan(); output=ROOT/'target_search'
    require_identical(output/'plan.json',plan)
    if args.prepare_only:
        print(json.dumps({'prepared':True,'plan':str(output/'plan.json'),'plan_sha256':object_hash(plan),
                          'candidates':list(CANDIDATES),'training_started':False}),flush=True); return
    require_corrected_complete()
    assert_no_other_training()
    with exclusive_lock(output/'active.lock'):
        assert_no_other_training()
        checks=audited(ROOT/'notes_recovered/runs',Path(plan['split_file']))
        corrected=read_json(ROOT/'notes_recovered/runs/e_softmask_geometry/result.json')
        evidence=target_evidence(corrected,checks['e_softmask_geometry'],plan['split_sha256'])
        if evidence['target_met']:
            save_json(output/'target_status.json',{'candidate_id':'corrected_e_softmask_geometry',**evidence})
            print(json.dumps({'candidate_id':'corrected_e_softmask_geometry',**evidence}),flush=True); return
        shared=shared_baseline(plan)
        # Audit existing candidates before spending GPU time on any further one.
        existing=[name for name in CANDIDATES if (output/'runs'/name/'result.json').exists()]
        if existing:
            previous_checks=audited(output/'runs',Path(plan['split_file']))
            for name in existing:
                previous=read_json(output/'runs'/name/'result.json')
                provenance=read_json(output/'runs'/name/'candidate_plan.json')
                if provenance['global_plan_sha256']!=object_hash(plan):
                    raise RuntimeError('Existing candidate belongs to a different frozen plan')
                if provenance['baseline_checkpoint_sha256']!=shared['baseline_checkpoint_sha256']:
                    raise RuntimeError('Existing candidate did not use the immutable shared baseline')
                if previous.get('candidate_plan_sha256')!=object_hash(provenance):
                    # A crash can occur between core result creation and provenance annotation.
                    # Only explicitly selecting that candidate may finalize it below.
                    if name!=args.candidate: raise RuntimeError('Unfinalized candidate requires explicit verification: '+name)
                    continue
                previous_evidence=target_evidence(previous,previous_checks[name],plan['split_sha256'])
                if previous_evidence['target_met']:
                    save_json(output/'target_status.json',{'candidate_id':name,**previous_evidence})
                    print(json.dumps({'candidate_id':name,**previous_evidence}),flush=True); return
        if not args.candidate:
            print(json.dumps({'candidate_required':True,'corrected_softmask':evidence,'available':list(CANDIDATES)}),flush=True); return
        run_candidate(args.candidate,args.resume,plan)


if __name__=='__main__': main()
