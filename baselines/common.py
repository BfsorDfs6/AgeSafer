"""Portable data/metric interfaces for the three transferred baselines."""
import argparse, importlib.util, json, math
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
KS=[1,5,10,20]
REPO=Path(__file__).resolve().parents[1]
def dump(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def ratings(path):
 out=defaultdict(list)
 with open(path,encoding='utf-8-sig') as f:
  for line in f:
   if line.strip():
    x=line.split();out[int(x[0])].append(int(x[1]))
 return dict(out)
def module(path,name):
 import sys
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
class Inputs:
 def __init__(self,config):
  self.config_path=Path(config).resolve();self.cfg=json.loads(self.config_path.read_text(encoding='utf-8'))
  self.ds=self.cfg['dataset'];self.paths={k:self.path(self.cfg[k]) for k in ['train_rating','valid_rating','test_rating']}
  self.train_path=self.paths['train_rating'];self.tr=ratings(self.train_path);self.va=ratings(self.paths['valid_rating']);self.te=ratings(self.paths['test_rating'])
  self.score_paths={k:self.path(v) for k,v in self.cfg.get('scores',{}).items()}
  if self.score_paths:
   dims=np.load(next(iter(self.score_paths.values())),mmap_mode='r').shape
  else:dims=(self.cfg['num_users'],self.cfg['num_items'])
  self.nu,self.ni=map(int,dims)
  if self.cfg.get('risk_flags'):
   data=np.load(self.path(self.cfg['risk_flags']),allow_pickle=False)
   self.minor=data['minor'].astype(bool);self.flags=data['flags'];self.names=data['names'].tolist()
  elif self.ds=='MAL':
   s=module(REPO/'experiments/scripts/mal2000_risk3/10_train_eval_mal_risk3_multihead_gate_fusion.py','baseline_mal')
   buckets=s.load_item_buckets(str(self.path(self.cfg['item_safe'])));um=s.load_user_minor(str(self.path(self.cfg['user_info'])))
   self.minor=np.array([um.get(u,False) for u in range(self.nu)],bool);self.names=['R17','RPLUS','RX','unsafe_any'];rr=[]
   for group in [False,True]:
    u=int(np.flatnonzero(self.minor==group)[0]);rows=[]
    for i in range(self.ni):
     v,z,_,_=s.mal_risk_flags(u,i,buckets,um);rows.append([v['R17'],v['RPLUS'],v['RX'],z])
    rr.append(rows)
   self.flags=np.array(rr,np.uint8)
  else:
   s=module(REPO/'experiments/scripts/train_eval_backbone_gated_residual_fusion.py','baseline_ml')
   items,cols=s.load_item_safe(str(self.path(self.cfg['item_safe'])));ui,mc=s.load_user_info(str(self.path(self.cfg['user_info'])))
   self.minor=np.array([s.is_minor_user(u,ui,mc) for u in range(self.nu)],bool);self.names=['sex','violence','profanity','drug','intense','unsafe_any','adult_content'];rr=[]
   for group in [False,True]:
    u=int(np.flatnonzero(self.minor==group)[0]);rows=[]
    for i in range(self.ni):
     v,z,adult,_,_=s.dim_violations(u,i,items,cols,ui,mc,self.cfg.get('minor_block_at',3),self.cfg.get('adult_block_at',4),self.cfg.get('isadult_policy','minor_only'))
     rows.append([v[n] for n in self.names[:5]]+[z,adult])
    rr.append(rows)
   self.flags=np.array(rr,np.uint8)
  assert self.flags.shape==(2,self.ni,len(self.names)) and len(self.minor)==self.nu
  assert np.isin(self.flags,[0,1]).all()
  self.zidx=self.names.index('unsafe_any')
  for u in set(self.tr)|set(self.va)|set(self.te):
   assert 0<=u<self.nu
   t,v,e=map(set,[self.tr.get(u,[]),self.va.get(u,[]),self.te.get(u,[])])
   assert not(t&v or t&e or v&e),('overlapping splits',u)
   assert all(0<=i<self.ni for i in t|v|e)
 def path(self,p):
  p=Path(p);return p if p.is_absolute() else self.config_path.parent/p
 def score(self,b):
  x=np.load(self.score_paths[b],mmap_mode='r');assert x.shape==(self.nu,self.ni);return x
 def rank(self,x,split,pool=None):
  ev=self.te if split=='test' else self.va;out={}
  for u in sorted(ev):
   row=np.asarray(x[u],dtype=np.float32).copy();assert np.isfinite(row).all()
   blocked=set(self.tr.get(u,[]))|(set(self.va.get(u,[])) if split=='test' else set())
   row[list(blocked)]=-np.inf;r=np.argsort(-row,kind='stable');r=r[np.isfinite(row[r])];out[u]=r if pool is None else r[:pool]
  return out
 def legacy_ctx(self):
  a=SimpleNamespace(**{k:str(p) for k,p in self.paths.items()})
  return a,self.tr,self.va,self.te,self.nu,self.ni,self.minor,self.flags,self.names
def evaluate(lists,data,split='test'):
 ev=data.te if split=='test' else data.va;out={}
 assert set(lists)==set(ev)
 for k in KS:
  out[str(k)]={}
  for group in ['all','minor','adult']:
   us=[u for u in sorted(ev) if group=='all' or bool(data.minor[u])==(group=='minor')]
   if not us:out[str(k)][group]={'users':0};continue
   hr=[];ndcg=[];viol=[];slots=[];hist=[];lengths=[]
   for u in us:
    ids=np.asarray(lists[u][:k],dtype=np.int64);assert len(set(ids))==len(ids)
    hit=np.isin(ids,ev[u]);hr.append(float(hit.any()))
    idcg=sum(1/math.log2(j+2) for j in range(min(len(set(ev[u])),k))) or 1
    ndcg.append(float((hit/np.log2(np.arange(len(ids))+2)).sum()/idcg))
    counts=data.flags[int(data.minor[u]),ids].sum(axis=0);viol.append(counts/k)
    slots.append(counts/max(len(ids),1));lengths.append(len(ids));hist.append(sum(i in data.tr.get(u,[]) for i in ids)/max(len(ids),1))
   m={'users':len(us),'HR':float(np.mean(hr)),'NDCG':float(np.mean(ndcg)),'mean_length':float(np.mean(lengths)),'full_list_fraction':float(np.mean(np.array(lengths)==k)),'history_repeat_fraction':float(np.mean(hist))}
   m.update({n:float(v) for n,v in zip(data.names,np.mean(viol,axis=0))})
   m.update({n+'_actual_slots':float(v) for n,v in zip(data.names,np.mean(slots,axis=0))})
   out[str(k)][group]=m
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--export-flags',required=True);a=p.parse_args();d=Inputs(a.config)
 target=Path(a.export_flags);target.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(target,minor=d.minor,flags=d.flags,names=np.array(d.names))
 print(json.dumps({'users':d.nu,'items':d.ni,'minors':int(d.minor.sum()),'dimensions':d.names}))
