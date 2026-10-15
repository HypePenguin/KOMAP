"""Reconstructed KOMAP experiment, with source assumptions recorded explicitly."""
from __future__ import annotations
import argparse, copy, csv, hashlib, json, math, os, platform, random, time, multiprocessing
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('TORCH_HOME', str(ROOT/'pretrained'/'torch'))
os.environ.setdefault('HF_HOME', str(ROOT/'pretrained'/'huggingface'))
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from sklearn.model_selection import KFold
import torch
from torch import nn
from torch.nn import functional as F
import segmentation_models_pytorch as smp

PALETTE = np.array([[153,127,76],[76,178,76],[25,76,153],[204,204,204]], np.uint8)
CLASSES = ['Al','Primary Si','Al3Ni','Eutectic Si']
VARIANTS = ['centerline_baseline','e_softmask_geometry','low_lr_context_paste','phase_style_exchange','e_protected_illumination']

def save_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def save_checkpoint(path, value):
    path=Path(path); temp=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temp); temp.replace(path)

def decode_mask(path):
    rgb=np.array(Image.open(path).convert('RGB'))
    eq=(rgb[...,None,:]==PALETTE[None,None,:,:]).all(-1)
    if not eq.any(-1).all():
        unknown=np.unique(rgb[~eq.any(-1)],axis=0)
        raise ValueError(f'Unknown label colors in {path}: {unknown[:10]}')
    return eq.argmax(-1).astype(np.uint8)

def prepare():
    manifest=[]; shape_counts={}; group_counts={}
    for subset in ['train','valid','test']:
        files=sorted((ROOT/'data'/subset/'images').glob('*.png'))
        for p in files:
            key=p.stem.removesuffix('_image'); im=Image.open(p)
            mask=ROOT/'data'/subset/'masks'/(key+'_mask.png')
            if subset!='test' and not mask.exists(): raise FileNotFoundError(mask)
            entry={'id':key,'source_subset':subset,'image':str(p.relative_to(ROOT)),
                   'mask':str(mask.relative_to(ROOT)) if mask.exists() else None,
                   'width':im.width,'height':im.height,'image_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            if mask.exists():
                lab=decode_mask(mask)
                if lab.shape!=(im.height,im.width): raise ValueError(f'Shape mismatch {key}')
                entry['mask_sha256']=hashlib.sha256(mask.read_bytes()).hexdigest()
                entry['class_pixels']=np.bincount(lab.ravel(),minlength=4).tolist()
            manifest.append(entry)
            shape_counts[str(im.size)]=shape_counts.get(str(im.size),0)+1
            group_counts[subset]=group_counts.get(subset,0)+1
    labeled=sorted((x for x in manifest if x['mask']),key=lambda x:x['id'])
    if len(labeled)!=90: raise ValueError(f'Expected 90 labeled images, got {len(labeled)}')
    tr,va=next(KFold(n_splits=5,shuffle=True,random_state=42).split(labeled))
    split={'train':[labeled[i]['id'] for i in tr], 'valid':[labeled[i]['id'] for i in va],
           'test':[x['id'] for x in manifest if x['source_subset']=='test'],
           'method':'RECONSTRUCTED: sorted ID + sklearn KFold(5, shuffle=True, random_state=42), fold0',
           'original_split_verified':False}
    save_json(ROOT/'records'/'data_manifest.json',manifest)
    save_json(ROOT/'configs'/'split_fold0.json',split)
    save_json(ROOT/'records'/'data_audit.json',{'counts':group_counts,'shapes':shape_counts,'palette':PALETTE.tolist(),
        'classes':CLASSES,'split_counts':{k:len(split[k]) for k in ['train','valid','test']}})
    print('DATA_AUDIT',group_counts,{k:len(split[k]) for k in ['train','valid','test']},flush=True)

def load_data():
    entries=json.loads((ROOT/'records'/'data_manifest.json').read_text(encoding='utf-8'))
    data={}
    for e in entries:
        image=np.array(Image.open(ROOT/e['image']).convert('RGB'))
        label=decode_mask(ROOT/e['mask']) if e['mask'] else None
        center=thin=None
        if label is not None:
            e_mask=label==3
            center=ndi.binary_dilation(skeletonize(e_mask),structure=ndi.generate_binary_structure(2,1),iterations=2)&e_mask
            thin=(ndi.distance_transform_edt(e_mask)<=2)&e_mask
        data[e['id']]={'image':image,'label':label,'center':center,'thin':thin,'entry':e}
    return data,json.loads((ROOT/'configs'/'split_fold0.json').read_text(encoding='utf-8'))

def d4(arr,k,flip):
    result=np.rot90(arr,k,axes=(0,1))
    return np.flip(result,axis=1).copy() if flip else result.copy()

def onehot(label):
    return np.eye(4,dtype=np.float32)[label]

def augment(image,label,center,thin,rng,variant,donors):
    soft=None; applied=False
    if variant=='e_softmask_geometry' and rng.random()<0.5:
        angle=float(rng.uniform(3,7))*rng.choice([-1,1])
        h,w=label.shape; mat=cv2.getRotationMatrix2D((w/2,h/2),angle,1)
        image2=cv2.warpAffine(image,mat,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        soft2=cv2.warpAffine(onehot(label),mat,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        soft2=cv2.resize(soft2[32:-32,32:-32],(w,h),interpolation=cv2.INTER_LINEAR)
        hard=soft2.argmax(-1).astype(np.uint8)
        area0=max(1,int((label==3).sum())); area1=int((hard==3).sum())
        n0=ndi.label(label==3)[1]; n1=ndi.label(hard==3)[1]
        if 0.85<=area1/area0<=1.15 and (n0==0 or 0.6<=n1/max(n0,1)<=1.6):
            image=cv2.resize(image2[32:-32,32:-32],(w,h),interpolation=cv2.INTER_LINEAR)
            center=cv2.warpAffine(center.astype(np.uint8),mat,(w,h),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_REFLECT_101)
            center=cv2.resize(center[32:-32,32:-32],(w,h),interpolation=cv2.INTER_NEAREST)>0
            label=hard; soft=soft2/np.maximum(soft2.sum(-1,keepdims=True),1e-8); applied=True
    elif variant=='low_lr_context_paste' and rng.random()<0.58:
        donor=donors[int(rng.integers(len(donors)))]; di,dl=donor
        cc,n=ndi.label(dl==3)
        candidates=[(component,s) for component,s in enumerate(ndi.find_objects(cc),1)
                    if s is not None and 4<=int((cc[s]==component).sum())<=300]
        rng.shuffle(candidates)
        for component,s in candidates[:8]:
            patch=di[s].astype(np.float32); e=(cc[s]==component)
            # Donors are native-resolution images; match the recipient's input x2.
            patch=cv2.resize(patch,None,fx=2,fy=2,interpolation=cv2.INTER_LINEAR)
            e=cv2.resize(e.astype(np.uint8),None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)>0
            ph,pw=e.shape
            if ph>label.shape[0] or pw>label.shape[1]: continue
            for attempt in range(8):
                y=int(rng.integers(0,label.shape[0]-ph+1)); x=int(rng.integers(0,label.shape[1]-pw+1))
                dest=label[y:y+ph,x:x+pw]
                if (dest[e]==0).mean()<0.98: continue
                alpha=ndi.gaussian_filter(e.astype(np.float32),1.4)*e
                region=image[y:y+ph,x:x+pw].astype(np.float32)
                shift=np.median(region[dest==0],axis=0)-np.median(patch[e],axis=0)
                patch=np.clip(patch+np.clip(shift,-20,20),0,255)
                image[y:y+ph,x:x+pw]=np.clip(region*(1-alpha[...,None])+patch*alpha[...,None],0,255).astype(np.uint8)
                dest[e]=3; center[y:y+ph,x:x+pw][e]=True; applied=True; break
    elif variant=='phase_style_exchange' and rng.random()<0.47:
        donor=donors[int(rng.integers(len(donors)))]; di,dl=donor
        transformed=image.astype(np.float32).copy()
        strength=float(rng.uniform(.2,.4))
        for c in range(4):
            a=image[label==c].astype(np.float32); b=di[dl==c].astype(np.float32)
            if len(a)<16 or len(b)<16: continue
            ma=np.median(a,axis=0); mb=np.median(b,axis=0)
            ia=np.maximum(np.percentile(a,75,axis=0)-np.percentile(a,25,axis=0),5)
            ib=np.maximum(np.percentile(b,75,axis=0)-np.percentile(b,25,axis=0),5)
            alpha=ndi.gaussian_filter((label==c).astype(np.float32),1.0)*strength
            swapped=(image.astype(np.float32)-ma)*np.clip(ib/ia,.75,1.25)+mb
            transformed=transformed*(1-alpha[...,None])+swapped*alpha[...,None]
        image=np.clip(transformed,0,255).astype(np.uint8); applied=True
    elif variant=='e_protected_illumination' and rng.random()<.56:
        protected=ndi.binary_dilation(label==3,iterations=3)
        h,w=label.shape; yy,xx=np.mgrid[-1:1:complex(h),-1:1:complex(w)]
        gradient=(rng.uniform(-.04,.04)*xx+rng.uniform(-.04,.04)*yy)*255
        modified=(image.astype(np.float32)-127.5)*rng.uniform(.95,1.05)+127.5+gradient[...,None]
        alpha=ndi.gaussian_filter((~protected).astype(np.float32),1)*~protected
        image=np.clip(image*(1-alpha[...,None])+modified*alpha[...,None],0,255).astype(np.uint8); applied=True
    return image,label,center,soft,applied

class PatchDataset(torch.utils.data.Dataset):
    def __init__(self,data,ids,variant,config,epoch):
        self.data=data; self.ids=ids; self.variant=variant; self.config=config; self.epoch=epoch
        self.donors=[(data[k]['image'],data[k]['label']) for k in ids]
    def __len__(self): return self.config['patches_per_epoch']
    def __getitem__(self,index):
        epoch=self.epoch.value if hasattr(self.epoch,'value') else self.epoch
        rng=np.random.default_rng(42+epoch*100000+index)
        a=self.data[self.ids[int(rng.integers(len(self.ids)))]]
        crop=self.config['native_crop']; h,w=a['label'].shape
        y=int(rng.integers(0,h-crop+1)); x=int(rng.integers(0,w-crop+1))
        region=(slice(y,y+crop),slice(x,x+crop)); k=int(rng.integers(4)); flip=bool(rng.integers(2))
        image=cv2.resize(d4(a['image'][region],k,flip),(768,768),interpolation=cv2.INTER_LINEAR)
        label=cv2.resize(d4(a['label'][region],k,flip),(768,768),interpolation=cv2.INTER_NEAREST)
        center=cv2.resize(d4(a['center'][region],k,flip).astype(np.uint8),(768,768),interpolation=cv2.INTER_NEAREST)>0
        thin=cv2.resize(d4(a['thin'][region],k,flip).astype(np.uint8),(768,768),interpolation=cv2.INTER_NEAREST)>0
        image,label,center,soft,applied=augment(image,label,center,thin,rng,self.variant,self.donors)
        if self.config.get('center_after_final_transform',False):
            eutectic=label==3
            center=ndi.binary_dilation(skeletonize(eutectic),structure=ndi.generate_binary_structure(2,1),iterations=2)&eutectic
        if soft is None: soft=onehot(label)
        inp=image.astype(np.float32)/255
        inp=(inp-np.array(self.config['mean'],np.float32))/np.array(self.config['std'],np.float32)
        return torch.from_numpy(inp.transpose(2,0,1).copy()),torch.from_numpy(soft.transpose(2,0,1).copy()),torch.from_numpy(center.astype(np.float32)),torch.tensor(int(applied))

def make_model(config,pretrained=False):
    return smp.Unet(encoder_name=config['encoder'],encoder_weights='imagenet' if pretrained else None,
                    in_channels=3,classes=4,decoder_channels=(256,128,64,32,16))

def loss_fn(logits,target,center,variant,recall_smooth=1e-7):
    logp=F.log_softmax(logits.float(),dim=1); p=logp.exp()
    ce=-(target*logp).sum(1).mean()
    dims=(0,2,3); intersection=(p*target).sum(dims); union=(p+target).sum(dims)
    dice=1-((2*intersection+1e-7)/(union+1e-7)).mean()
    area=target[:,3] if variant=='low_lr_context_paste' else center
    denom=area.sum((1,2)); present=denom>0
    recall=((p[:,3]*area).sum((1,2))/(denom+recall_smooth))[present].mean() if present.any() else p.sum()*0+1
    return ce+dice+.1*(1-recall)

@torch.no_grad()
def update_ema(ema_model,model,decay):
    online=model.state_dict()
    for name,value in ema_model.state_dict().items():
        if value.is_floating_point(): value.mul_(decay).add_(online[name],alpha=1-decay)
        else: value.copy_(online[name])

def positions(length,tile,stride):
    if length<=tile: return [0]
    v=list(range(0,length-tile+1,stride))
    if v[-1]!=length-tile: v.append(length-tile)
    return v

@torch.inference_mode()
def predict(model,image,config,tta=False,tile=768):
    h,w=image.shape[:2]; up=cv2.resize(image,(w*2,h*2),interpolation=cv2.INTER_LINEAR)
    ih,iw=up.shape[:2]; ph=max(ih,tile); pw=max(iw,tile)
    up=cv2.copyMakeBorder(up,0,ph-ih,0,pw-iw,cv2.BORDER_REFLECT_101)
    result=np.zeros((4,ph,pw),np.float32); weight=np.zeros((ph,pw),np.float32)
    coords=[(y,x) for y in positions(ph,tile,tile//2) for x in positions(pw,tile,tile//2)]
    for start in range(0,len(coords),config['inference_batch']):
        batch_coords=coords[start:start+config['inference_batch']]
        patches=np.stack([up[y:y+tile,x:x+tile].transpose(2,0,1) for y,x in batch_coords]).astype(np.float32)/255
        tensor=torch.from_numpy(patches).cuda()
        mean=torch.tensor(config['mean'],device='cuda')[None,:,None,None]
        std=torch.tensor(config['std'],device='cuda')[None,:,None,None]
        tensor=(tensor-mean)/std; total=None
        for flip in ([False,True] if tta else [False]):
            for k in (range(4) if tta else [0]):
                transformed=torch.rot90(tensor,k,(-2,-1))
                if flip: transformed=torch.flip(transformed,[-1])
                with torch.autocast('cuda',dtype=torch.float16): logits=model(transformed)
                prob=logits.float().softmax(1)
                if flip: prob=torch.flip(prob,[-1])
                prob=torch.rot90(prob,-k,(-2,-1))
                total=prob if total is None else total+prob
        probs=(total/(8 if tta else 1)).cpu().numpy()
        for (y,x),pr in zip(batch_coords,probs): result[:,y:y+tile,x:x+tile]+=pr; weight[y:y+tile,x:x+tile]+=1
    result=result[:,:ih,:iw]/weight[None,:ih,:iw]
    native=cv2.resize(result.transpose(1,2,0),(w,h),interpolation=cv2.INTER_LINEAR)
    return native.argmax(-1).astype(np.uint8)

def metrics_from_predictions(predictions,data,ids):
    rows=[]; cms=[]
    for key,pred in zip(ids,predictions):
        target=data[key]['label']; cm=np.bincount(target.ravel()*4+pred.ravel(),minlength=16).reshape(4,4)
        union=cm.sum(0)+cm.sum(1)-cm.diagonal()
        iou=np.divide(cm.diagonal(),union,out=np.full(4,np.nan),where=union>0)*100
        thin=data[key]['thin']; thin_recall=float(((pred==3)&thin).sum()/max(1,thin.sum())*100)
        rows.append({'id':key,'mIoU':float(np.nanmean(iou)),'class_iou':[None if np.isnan(v) else float(v) for v in iou],
                     'thin_e_recall':thin_recall,'e_to_al3ni_pixels':int(((target==3)&(pred==2)).sum()),'confusion':cm.tolist()}); cms.append(cm)
    global_cm=np.sum(cms,axis=0); u=global_cm.sum(0)+global_cm.sum(1)-global_cm.diagonal()
    global_iou=global_cm.diagonal()/u*100
    class_iou=[float(np.mean([r['class_iou'][c] for r in rows if r['class_iou'][c] is not None])) for c in range(4)]
    return {'mIoU':float(np.mean([r['mIoU'] for r in rows])),'class_iou':class_iou,'global_mIoU':float(global_iou.mean()),
            'thin_e_recall':float(np.mean([r['thin_e_recall'] for r in rows])),
            'e_to_al3ni_pixels':sum(r['e_to_al3ni_pixels'] for r in rows),'per_image':rows,
            'metric_definition':'mean over images of mean class IoU; empty class union excluded; unit percent'}

def evaluate(model,data,ids,config,tta=False,out=None,tile=768):
    start=time.perf_counter(); preds=[]
    if out: Path(out).mkdir(parents=True,exist_ok=True)
    for key in ids:
        pred=predict(model,data[key]['image'],config,tta=tta,tile=tile); preds.append(pred)
        if out: Image.fromarray(PALETTE[pred]).save(Path(out)/(key+'_mask.png'))
    result=metrics_from_predictions(preds,data,ids) if data[ids[0]]['label'] is not None else {'count':len(preds),'mIoU':None,'reason':'test masks not provided'}
    result['seconds']=time.perf_counter()-start
    if out: save_json(Path(out)/'metrics.json',result)
    return result

def run_training(variant,config,data,split,init=None,resume=False,run_directory=None,selection_tile=768,early_stop=True):
    run=Path(run_directory) if run_directory is not None else ROOT/'runs'/variant
    run.mkdir(parents=True,exist_ok=True)
    seed_everything(42)
    model=make_model(config,pretrained=init is None).cuda().to(memory_format=torch.channels_last)
    if init: model.load_state_dict(torch.load(init,map_location='cpu',weights_only=False)['model'])
    encoder_lr=config['encoder_lr']; decoder_lr=config['decoder_lr']
    if variant=='low_lr_context_paste': encoder_lr,decoder_lr=1e-6,1e-5
    optimizer=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':encoder_lr},
                                {'params':list(model.decoder.parameters())+list(model.segmentation_head.parameters()),'lr':decoder_lr}],weight_decay=config.get('weight_decay',1e-4))
    per_step_schedule=config.get('scheduler_per_step',False)
    steps_per_epoch=math.ceil(config['patches_per_epoch']/config['batch'])
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['max_epochs']*(steps_per_epoch if per_step_schedule else 1),eta_min=config.get('min_lr',1e-7))
    ema_decay=config.get('ema_decay',0)
    ema_model=copy.deepcopy(model).eval() if ema_decay else None
    if ema_model:
        for p in ema_model.parameters(): p.requires_grad_(False)
    scaler=torch.amp.GradScaler('cuda'); best=-1.; anchor=-1.; last_improvement=0; start_epoch=1
    elapsed=0.; attempts=0; updates=0; skips=0; records=[]
    if resume and (run/'last.pt').exists():
        checkpoint=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer']); schedule.load_state_dict(checkpoint['schedule']); scaler.load_state_dict(checkpoint['scaler'])
        if ema_model:
            if not checkpoint.get('ema'): raise RuntimeError('EMA resume requires a saved EMA state; use the corrected run directory')
            ema_model.load_state_dict(checkpoint['ema'])
        start_epoch=checkpoint['epoch']+1; best=checkpoint['best']; anchor=checkpoint['anchor']; last_improvement=checkpoint['last_improvement']; elapsed=checkpoint['elapsed']; attempts=checkpoint['attempts']; updates=checkpoint['updates']; skips=checkpoint['skips']; records=checkpoint['records']
        if checkpoint.get('rng_state'):
            rng=checkpoint['rng_state']; random.setstate(rng['python']); np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch']); torch.cuda.set_rng_state_all(rng['cuda'])
    save_json(run/'config.json',{**config,'variant':variant,'initial_checkpoint':str(init) if init else 'ImageNet',
              'encoder_lr':encoder_lr,'decoder_lr':decoder_lr,'selection_tile':selection_tile,
              'early_stop_enabled':early_stop,'status':'reconstructed, original split/code unavailable'})
    print(f'START {variant} epoch={start_epoch} initial={init}',flush=True)
    train_start=time.perf_counter()
    shared_epoch=multiprocessing.Value('i',start_epoch)
    dataset=PatchDataset(data,split['train'],variant,config,shared_epoch)
    loader_generator=torch.Generator().manual_seed(42)
    loader=torch.utils.data.DataLoader(dataset,batch_size=config['batch'],num_workers=config['workers'],pin_memory=True,shuffle=False,persistent_workers=config['workers']>0,generator=loader_generator)
    # An interrupted final inference must resume inference, including after early stop.
    epoch=start_epoch-1
    training_finished=epoch>=config['max_epochs'] or (early_stop and epoch>0 and epoch%5==0 and epoch-last_improvement>=config['patience_epochs'])
    if training_finished: print(f'RESUME_FINAL_INFERENCE {variant} epoch={epoch}',flush=True)
    for epoch in range(start_epoch, start_epoch if training_finished else config['max_epochs']+1):
        epoch_start=time.perf_counter(); model.train(); total_loss=0.; applied=0; n=0
        shared_epoch.value=epoch
        for step,(x,target,center,aug) in enumerate(loader,1):
            x=x.cuda(non_blocking=True).to(memory_format=torch.channels_last); target=target.cuda(non_blocking=True); center=center.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16): logits=model(x)
            loss=loss_fn(logits,target,center,variant,config.get('recall_smooth',1e-7))
            if not torch.isfinite(loss): raise RuntimeError(f'Non-finite loss epoch{epoch}')
            old_scale=scaler.get_scale(); scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(optimizer); scaler.update()
            attempts+=1
            if scaler.get_scale()<old_scale: skips+=1
            else: updates+=1
            if ema_model: update_ema(ema_model,model,ema_decay)
            if per_step_schedule: schedule.step()
            total_loss+=float(loss.detach())*len(x); n+=len(x); applied+=int(aug.sum())
            if step%25==0:
                save_json(run/'progress.json',{'variant':variant,'stage':'training','epoch_completed':epoch-1,'epoch_in_progress':epoch,
                    'batch_in_epoch':step,'batches_per_epoch':len(loader),'max_epochs':config['max_epochs'],
                    'best_no_tta':best,'minutes':(elapsed+time.perf_counter()-train_start)/60,'attempts':attempts,'updates':updates,'amp_skips':skips})
                print(f'BATCH {variant} epoch{epoch} {step}/{len(loader)}',flush=True)
        if not per_step_schedule: schedule.step()
        train_seconds=time.perf_counter()-epoch_start
        row={'epoch':epoch,'loss':total_loss/n,'train_seconds':train_seconds,'augmentation_applied':applied,'samples':n,'mIoU':None}
        if epoch%5==0 or epoch==config['max_epochs']:
            evaluation_model=ema_model if ema_model is not None else model
            evaluation_model.eval(); metric=evaluate(evaluation_model,data,split['valid'],config,tta=False,tile=selection_tile); row['mIoU']=metric['mIoU']; row['validation_seconds']=metric['seconds']
            save_json(run/f'validation_epoch{epoch:03d}.json',metric)
            if metric['mIoU']>best:
                best=metric['mIoU']; save_checkpoint(run/'best.pt',{'model':evaluation_model.state_dict(),'epoch':epoch,'metric':metric,'config':config,'model_kind':'EMA' if ema_model is not None else 'online'})
            if metric['mIoU']>anchor+config['min_delta_pp']:
                anchor=metric['mIoU']; last_improvement=epoch
        records.append(row)
        current_elapsed=elapsed+time.perf_counter()-train_start
        checkpoint={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'schedule':schedule.state_dict(),'scaler':scaler.state_dict(),
                    'ema':ema_model.state_dict() if ema_model is not None else None,
                    'epoch':epoch,'best':best,'anchor':anchor,'last_improvement':last_improvement,'elapsed':current_elapsed,
                    'attempts':attempts,'updates':updates,'skips':skips,'records':records,
                    'rng_state':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}}
        if epoch%5==0:
            save_checkpoint(run/'last.pt',checkpoint)
        save_json(run/'progress.json',{'variant':variant,'epoch':epoch,'max_epochs':config['max_epochs'],'best_no_tta':best,'minutes':current_elapsed/60,'last_epoch':row,'attempts':attempts,'updates':updates,'amp_skips':skips})
        save_json(run/'history.json',records)
        print(f'EPOCH {variant} {epoch} loss={row["loss"]:.5f} val={row["mIoU"]} best={best:.5f} seconds={train_seconds:.1f}',flush=True)
        if early_stop and epoch%5==0 and epoch-last_improvement>=config['patience_epochs']:
            print(f'EARLY_STOP {variant} epoch{epoch}',flush=True); break
    train_seconds=elapsed+time.perf_counter()-train_start
    cp=torch.load(run/'best.pt',map_location='cpu',weights_only=False); model.load_state_dict(cp['model']); model.eval()
    valid=evaluate(model,data,split['valid'],config,tta=True,out=run/'valid_masks_d4')
    test=evaluate(model,data,split['test'],config,tta=True,out=run/'test_masks_d4')
    result={'variant':variant,'best_epoch':cp['epoch'],'epochs':epoch,'train_minutes':train_seconds/60,'validation':valid,'test':test,
            'attempts':attempts,'updates':updates,'amp_skips':skips,'original_split_verified':False,'original_code_recovered':False,
            'target_79_reached':valid['mIoU']>=79,'original_record_mIoU':config['reference_mIoU'][variant],
            'selection_tile':selection_tile,'final_tta_tile':768,'early_stop_enabled':early_stop,
            'model_kind':cp.get('model_kind','online'),'weight_decay':config.get('weight_decay',1e-4),'ema_decay':ema_decay}
    save_json(run/'result.json',result)
    print(f'RESULT {variant} mIoU={valid["mIoU"]:.6f} best_epoch={cp["epoch"]}',flush=True)
    del model,optimizer; torch.cuda.empty_cache()
    return run/'best.pt',result

def verify():
    a=np.arange(64).reshape(8,8)
    for flip in [False,True]:
        for k in range(4):
            b=d4(a,k,flip)
            if flip: b=np.flip(b,1)
            assert np.array_equal(np.rot90(b,-k),a)
    gt=np.array([[0,1],[2,3]],np.uint8)
    fake={'x':{'label':gt,'thin':gt==3}}
    perfect=metrics_from_predictions([gt],fake,['x'])
    assert perfect['mIoU']==100 and perfect['thin_e_recall']==100
    logit=torch.randn(2,4,16,16,requires_grad=True)
    target=F.one_hot(torch.randint(0,4,(2,16,16)),4).permute(0,3,1,2).float()
    loss=loss_fn(logit,target,target[:,3],'centerline_baseline'); loss.backward()
    assert torch.isfinite(logit.grad).all()
    soft=target*.8+.2/4; loss=loss_fn(logit,soft,soft[:,3],'e_softmask_geometry')
    assert torch.isfinite(loss)
    assert positions(2000,768,384)[-1]==2000-768
    print('VERIFICATION PASSED: palette, D4 inversion, image mean mIoU, gradients, soft targets, tile coverage',flush=True)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['prepare','verify','smoke','all','infer'])
    parser.add_argument('--resume',action='store_true'); parser.add_argument('--variant',choices=VARIANTS); args=parser.parse_args()
    if args.action=='prepare': prepare(); return
    if args.action=='verify': verify(); return
    torch.set_num_threads(4); torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
    config=json.loads((ROOT/'configs'/'experiment.json').read_text(encoding='utf-8'))
    data,split=load_data()
    if args.action=='smoke':
        seed_everything(42); model=make_model(config,pretrained=True).cuda().to(memory_format=torch.channels_last)
        d=PatchDataset(data,split['train'],'centerline_baseline',config,1)
        x,t,c,_=next(iter(torch.utils.data.DataLoader(d,batch_size=config['batch'])))
        start=time.perf_counter(); x=x.cuda().to(memory_format=torch.channels_last)
        with torch.autocast('cuda',dtype=torch.float16): logits=model(x)
        loss=loss_fn(logits,t.cuda(),c.cuda(),'centerline_baseline'); loss.backward(); torch.cuda.synchronize()
        print('SMOKE',float(loss.detach()),'seconds',time.perf_counter()-start,'peakGiB',torch.cuda.max_memory_allocated()/2**30,flush=True)
        for i in range(3):
            model.zero_grad(set_to_none=True); start=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.float16): logits=model(x)
            loss=loss_fn(logits,t.cuda(),c.cuda(),'centerline_baseline'); loss.backward(); torch.cuda.synchronize()
            print('WARM_STEP',i,time.perf_counter()-start,flush=True)
        model.eval(); pred=predict(model,data[split['valid'][0]]['image'],config,tta=False)
        result=metrics_from_predictions([pred],data,[split['valid'][0]])
        print('SMOKE_INFERENCE',pred.shape,result['mIoU'],flush=True); return
    if args.action=='infer':
        if not args.variant: raise ValueError('--variant required')
        cp=torch.load(ROOT/'runs'/args.variant/'best.pt',map_location='cpu',weights_only=False)
        model=make_model(config).cuda(); model.load_state_dict(cp['model']); model.eval()
        for subset in ['valid','test']: evaluate(model,data,split[subset],config,tta=True,out=ROOT/'runs'/args.variant/(subset+'_masks_d4'))
        return
    summary={}; base=ROOT/'runs'/'centerline_baseline'/'best.pt'
    for variant in VARIANTS:
        if args.variant and variant!=args.variant: continue
        complete=ROOT/'runs'/variant/'result.json'
        if args.resume and complete.exists():
            summary[variant]=json.loads(complete.read_text(encoding='utf-8')); continue
        init=None if variant=='centerline_baseline' else base
        _,result=run_training(variant,config,data,split,init=init,resume=args.resume)
        summary[variant]=result; save_json(ROOT/'runs'/'summary.json',summary)
    save_json(ROOT/'runs'/'summary.json',summary)
    print('ALL_COMPLETE',flush=True)

if __name__=='__main__': main()
