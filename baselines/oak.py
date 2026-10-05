"""Oak-PE: greedy A/B/NONE judgments in both orders, harm-vote reranking."""
import argparse,itertools,json,re
from pathlib import Path
import numpy as np
from common import Inputs,dump,evaluate
def policy(ds,age):
 group='a minor younger than 18' if age else 'an adult aged 18 or older'
 if ds=='ML-1M':definition='For minors, moderate or stronger sexual content, violence, profanity, drug use, or frightening/intense scenes are age-inappropriate. For adults, only strong/severe content in these categories counts. Explicit adult content is also inappropriate for minors.'
 else:definition='For minors, anime rated R-17+, R+ (mild nudity), or Rx (hentai) is age-inappropriate. For adults, R+ or Rx content counts as risk; R-17+ alone does not. Infer likely content from the title and genres; no rating label is provided.'
 return 'Assess content suitability for '+group+'. '+definition+' Use your knowledge of the named work. Do not treat uncertainty as evidence of harm.'
def parse(text):
 match=re.match(r'^\s*=\s*(NONE|A|B)(?:\b|$)',text,re.I)
 return {'A':0,'B':1,'NONE':2}[match.group(1).upper()] if match else None
def rerank(r,age,lookup,choices):
 votes=np.zeros(len(r),np.int32)
 for x,y in itertools.combinations(range(len(r)),2):
  lo,hi=sorted((int(r[x]),int(r[y])));a=choices[lookup[age,lo,hi]]
  assert (a>=0).all()
  sl=int(a[0]==0)+int(a[1]==1);sh=int(a[0]==1)+int(a[1]==0)
  votes[x]+=sl if r[x]==lo else sh;votes[y]+=sh if r[y]==hi else sl
 return r[np.argsort(votes,kind='stable')]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--phase',choices=['prepare','infer','evaluate'],required=True);p.add_argument('--backbones',default='GMF,NeuMF,LightGCN,LLM-SRec');p.add_argument('--pool',type=int,default=50);p.add_argument('--model-path');p.add_argument('--batch',type=int,default=16);p.add_argument('--device',default='cuda:0');a=p.parse_args();d=Inputs(a.config);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 bones=a.backbones.split(',')
 if a.phase=='prepare':
  jobs=set();allr={}
  for b in bones:
   rs=d.rank(d.score(b),'test',a.pool);assert all(len(r)>=20 for r in rs.values());allr[b]={str(u):r.tolist() for u,r in rs.items()}
   for u,r in rs.items():
    for i,j in itertools.combinations(r,2):lo,hi=sorted((int(i),int(j)));jobs.add((int(d.minor[u]),lo,hi))
  jobs=np.array(sorted(jobs),np.int32);np.save(out/'pairs.npy',jobs);choices=np.lib.format.open_memmap(out/'choices.npy',mode='w+',dtype=np.int8,shape=(len(jobs),2));choices[:]=-1;choices.flush()
  dump(out/'candidates.json',allr);dump(out/'CONFIG.json',{'pool':a.pool,'backbones':bones,'dataset':d.ds});return
 jobs=np.load(out/'pairs.npy');choices=np.load(out/'choices.npy',mmap_mode='r+' if a.phase=='infer' else 'r')
 if a.phase=='infer':
  if not a.model_path:p.error('--model-path is required')
  import torch
  from transformers import AutoTokenizer,AutoModelForCausalLM
  torch.manual_seed(2028);tok=AutoTokenizer.from_pretrained(a.model_path,local_files_only=True,padding_side='left');tok.pad_token=tok.eos_token
  model=AutoModelForCausalLM.from_pretrained(a.model_path,local_files_only=True,torch_dtype=torch.float16).to(a.device).eval()
  texts={int(k):v for k,v in json.loads(d.path(d.cfg['item_texts']).read_text(encoding='utf-8')).items()};work=[]
  for j,(age,i,k) in enumerate(jobs):
   for direction,(first,second) in enumerate([(i,k),(k,i)]):
    if choices[j,direction]<0:work.append((j,direction,int(age),int(first),int(second)))
  for start in range(0,len(work),a.batch):
   batch=work[start:start+a.batch];prompts=[]
   for j,direction,age,first,second in batch:
    q=policy(d.ds,age)+'\nText A: '+texts[first]+'\nText B:\n'+texts[second]+'\nWhich is relatively more age-inappropriate? Respond Response=A if Text A is more inappropriate, or Response=B if Text B is more inappropriate. If both are appropriate, respond Response=NONE. Do not add any explanation.'
    prompts.append(tok.apply_chat_template([{'role':'user','content':q}],tokenize=False,add_generation_prompt=True)+'Response')
   enc=tok(prompts,padding=True,add_special_tokens=False,return_tensors='pt');enc={k:v.to(a.device) for k,v in enc.items()};width=enc['input_ids'].shape[1]
   with torch.inference_mode():ans=model.generate(**enc,max_new_tokens=12,do_sample=False,pad_token_id=tok.pad_token_id)
   errors=[]
   for entry,text in zip(batch,tok.batch_decode(ans[:,width:],skip_special_tokens=True)):
    j,direction,*_=entry;answer=parse(text)
    if answer is None:errors.append({'job':j,'direction':direction,'raw':text})
    else:choices[j,direction]=answer
   choices.flush()
   if errors:dump(out/'PARSE_ERRORS.json',errors);raise RuntimeError('Unparseable A/B/NONE answers; not counted as safe')
   print(start+len(batch),'/',len(work),flush=True)
  return
 assert (choices>=0).all(),'inference is incomplete'
 lookup={tuple(map(int,row)):j for j,row in enumerate(jobs)};saved=json.loads((out/'candidates.json').read_text());records=[]
 for b,candidates in saved.items():
  full={int(u):rerank(np.array(r),int(d.minor[int(u)]),lookup,choices) for u,r in candidates.items()};lists={u:r[:20] for u,r in full.items()}
  records.append({'dataset':d.ds,'backbone':b,'method':'Oak-PE','pool':json.loads((out/'CONFIG.json').read_text())['pool'],'deterministic':True,'metrics':evaluate(lists,d)})
  dump(out/(b+'_reranked.json'),{str(u):r.tolist() for u,r in full.items()})
 dump(out/'RESULTS.json',records)
if __name__=='__main__':main()
