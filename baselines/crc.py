"""CRC Remove/history-union adapter, with validation-only calibration."""
import argparse,sys
from pathlib import Path
import numpy as np
from common import Inputs,dump,evaluate
def select(r,h,relevance,safety,gamma,method):
 keep=r[safety[r]>=gamma]
 if method=='remove':return keep[:20]
 ids=np.union1d(keep,h);return ids[np.argsort(-relevance[ids],kind='stable')][:20]
def calibrate(cal,hist,x,risk,d,method):
 gammas=np.r_[np.linspace(0,1,100),np.nextafter(1.,2.)];per=np.zeros((len(cal),len(gammas)),np.float64)
 for n,(u,r) in enumerate(cal.items()):
  if method=='history-union':
   merged=np.union1d(r,hist[u]);merged=merged[np.argsort(-x[u,merged],kind='stable')];historical=np.isin(merged,hist[u])
  else:merged=r;historical=np.zeros(len(r),bool)
  mask=(risk[u,merged,None]>=gammas[None,:])|historical[:,None]
  accepted=mask&(np.cumsum(mask,axis=0)<=20)
  per[n]=(accepted*d.flags[int(d.minor[u]),merged,d.zidx,None]).sum(axis=0)/20
 envelope=np.maximum.accumulate(per[:,::-1],axis=1)[:,::-1].mean(axis=0)
 return gammas,per.mean(axis=0),(len(cal)*envelope+1)/(len(cal)+1)
def train_risk(a,d):
 import torch
 sys.path.insert(0,str(Path(a.crc_root).resolve()));from src.utils import RankerNN
 torch.manual_seed(a.seed);np.random.seed(a.seed);torch.set_num_threads(8);device=torch.device(a.device)
 us=np.array([u for u,items in d.tr.items() for i in items],np.int64);it=np.array([i for items in d.tr.values() for i in items],np.int64)
 ys=1-d.flags[d.minor[us].astype(int),it,d.zidx].astype(np.float32)
 model=RankerNN(d.nu-1,d.ni-1,num_genres=1,max_output_value=1,min_output_value=0).to(device);model.disable_sigmoid_in_forward(True)
 opt=torch.optim.Adam(model.parameters(),lr=.001);lossfn=torch.nn.BCEWithLogitsLoss()
 ut=torch.tensor(us,device=device);it=torch.tensor(it,device=device);ys=torch.tensor(ys,device=device);ages=torch.tensor(d.minor.astype(np.float32),device=device);losses=[]
 for ep in range(a.epochs):
  total=0.
  for ix in torch.randperm(len(us),device=device).split(4096):
   opt.zero_grad();loss=lossfn(model(ut[ix],it[ix],ages[ut[ix]][:,None]),ys[ix]);loss.backward();opt.step();total+=float(loss.detach())*len(ix)
  losses.append(total/len(us));print('epoch',ep+1,'BCE',losses[-1],flush=True)
 model.eval();model.disable_sigmoid_in_forward(False);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 arr=np.lib.format.open_memmap(out/'learned_safety.npy',mode='w+',dtype=np.float32,shape=(d.nu,d.ni))
 with torch.inference_mode():
  for start in range(0,d.nu,16):
   n=min(16,d.nu-start);u=torch.arange(start,start+n,device=device).repeat_interleave(d.ni);i=torch.arange(d.ni,device=device).repeat(n)
   arr[start:start+n]=model(u,i,ages[u][:,None]).cpu().numpy().reshape(n,d.ni)
 arr.flush();torch.save({'state_dict':model.state_dict(),'seed':a.seed,'epochs':a.epochs},out/'learned_safety.pt')
 dump(out/'TRAINING.json',{'seed':a.seed,'epochs':a.epochs,'losses':losses,'inputs':['user_id','item_id','is_minor'],'target':'1-unsafe_any on original observed training interactions only'})
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--phase',choices=['train-risk','evaluate'],default='evaluate');p.add_argument('--output',required=True)
 p.add_argument('--backbone');p.add_argument('--safety-scores');p.add_argument('--method',choices=['remove','history-union'],default='remove');p.add_argument('--fractions',default='1,.75,.5,.25,.1,0');p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=20);p.add_argument('--device',default='cuda:0');p.add_argument('--crc-root');a=p.parse_args();d=Inputs(a.config)
 if a.phase=='train-risk':
  if not a.crc_root:p.error('--crc-root is required for the upstream RankerNN')
  train_risk(a,d);return
 if not a.backbone or not a.safety_scores:p.error('evaluation requires --backbone and --safety-scores')
 x=d.score(a.backbone);risk=np.load(a.safety_scores,mmap_mode='r');assert risk.shape==x.shape and np.isfinite(risk).all() and ((risk>=0)&(risk<=1)).all()
 cal=d.rank(x,'valid');test=d.rank(x,'test');assert len(cal)>0
 hist={u:np.unique([i for i in d.tr.get(u,[]) if not d.flags[int(d.minor[u]),i,d.zidx]]).astype(np.int64) for u in set(cal)|set(test)}
 gammas,raw,adjusted=calibrate(cal,hist,x,risk,d,a.method);out=Path(a.output)
 dump(out/'CALIBRATION.json',{'gammas':gammas.tolist(),'raw_risk':raw.tolist(),'adjusted_envelope':adjusted.tolist()})
 records=[]
 for fraction in map(float,a.fractions.split(',')):
  alpha=float(raw[0]*fraction);eligible=np.flatnonzero(adjusted<=alpha);j=int(eligible[0]) if len(eligible) else len(gammas)-1
  lists={u:select(r,hist[u],x[u],risk[u],float(gammas[j]),a.method) for u,r in test.items()}
  records.append({'method':'CRC-'+a.method,'dataset':d.ds,'backbone':a.backbone,'seed':a.seed,'fraction':fraction,'alpha':alpha,'gamma':float(gammas[j]),'calibration_feasible':bool(len(eligible)),'metrics':evaluate(lists,d)})
  dump(out/('top20_fraction_'+str(fraction)+'.json'),{str(u):r.tolist() for u,r in lists.items()})
 dump(out/'RESULTS.json',records)
if __name__=='__main__':main()
