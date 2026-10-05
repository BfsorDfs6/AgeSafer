"""LLM4IDRec + age-safety filtering: train-only ID generation, one augmented branch."""
import argparse,collections,csv,hashlib,json,os,pathlib,random,re,shutil,subprocess,sys,time
from types import SimpleNamespace
from common import Inputs
def core():
    return SimpleNamespace(setup=lambda ds: Inputs(CONFIG).legacy_ctx())

def read(p):return json.loads(pathlib.Path(p).read_text())

def dump(p,x):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2));q.replace(p)

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def rows(p):
    with open(p) as f:
        for l in f:
            if l.strip():yield json.loads(l)

def prompt(u,items):
    return f"Given the user(u{u})'s clicked list items:"+','.join('i'+str(i) for i in items)+f", predict what is the list items to recommend to the user(u{u}). Please only answer the item IDs."

def cli(d):
    args=[]
    for k,v in d.items():
        if isinstance(v,bool):
            if v:args.append('--'+k)
        elif v is not None:args+=['--'+k,str(v)]
    return args

def prepare(ds):
    import numpy as np,yaml
    from transformers import AutoTokenizer
    root=EXP/ds;root.mkdir(parents=True,exist_ok=True)
    train_path=Inputs(CONFIG).train_path;tr=collections.defaultdict(list)
    with train_path.open() as f:
        for line in f:
            parts=line.split()
            if parts:tr[int(parts[0])].append(int(parts[1]))
    tok=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
    rng=random.Random(SEED);records=[];stats=collections.Counter()
    def fit(u,ids,budget):
        ids=list(ids)
        while len(tok.encode(prompt(u,ids),add_special_tokens=False))>budget and len(ids)>1:ids=ids[len(ids)//8+1:]
        return ids
    with (root/'sft.jsonl').open('w') as f:
        # Follow active author's 20 random shuffled splits/user, with actual output variable.
        for u,items in sorted(tr.items()):
            items=list(dict.fromkeys(items))
            if len(items)<3:stats['users_without_three_train_items']+=1;continue
            for _ in range(20):
                ids=items.copy();rng.shuffle(ids);split=rng.randrange(1,len(ids)-1)
                source=fit(u,ids[:split],1500);target=ids[split:]
                # Preserve the beginning of the target list within the 2048-token budget.
                target_budget=1950-len(tok.encode(prompt(u,source),add_special_tokens=False))
                while len(tok.encode(','.join('i'+str(i) for i in target),add_special_tokens=False))>target_budget and len(target)>1:target=target[:max(1,int(len(target)*.9))]
                assert set(source).isdisjoint(target) and set(source+target)<=set(items)
                rec={'instruction':prompt(u,source),'input':'','output':','.join('i'+str(i) for i in target)}
                f.write(json.dumps(rec)+'\n');stats['sft_rows']+=1
    with (root/'infer.jsonl').open('w') as f:
        for u,items in sorted(tr.items()):
            ids=list(dict.fromkeys(items));kept=fit(u,ids,1848)
            f.write(json.dumps({'user_id':u,'instruction':prompt(u,kept),'history_count':len(ids),'input_history_count':len(kept)})+'\n')
            stats['inference_users']+=1;stats['history_truncated_users']+=len(kept)<len(ids)
    data=root/'lf_data';data.mkdir(exist_ok=True)
    dump(data/'dataset_info.json',{'llm4idrec':{'file_name':str(root/'sft.jsonl'),'columns':{'prompt':'instruction','query':'input','response':'output'}}})
    y={'model_name_or_path':MODEL,'trust_remote_code':True,'stage':'sft','do_train':True,'finetuning_type':'lora','lora_rank':8,'lora_alpha':32,'lora_dropout':.05,'lora_target':'q_proj,v_proj','dataset':'llm4idrec','dataset_dir':str(data),'template':'qwen','cutoff_len':2048,'train_on_prompt':True,'overwrite_cache':True,'preprocessing_num_workers':4,'output_dir':str(root/'adapter'),'logging_steps':10,'save_steps':100,'save_total_limit':2,'per_device_train_batch_size':1,'gradient_accumulation_steps':32,'learning_rate':.001,'max_steps':400,'lr_scheduler_type':'linear','warmup_ratio':0.,'bf16':True,'gradient_checkpointing':True,'seed':SEED,'data_seed':2023,'report_to':'none','cache_dir':str(root/'hf_cache'),'optim':'adamw_torch','overwrite_output_dir':False}
    (root/'train.yaml').write_text(yaml.safe_dump(y,sort_keys=False));dump(root/'PREPARED.json',dict(stats=stats,train_yaml=y,train_only_source=str(train_path),train_sha256=sha(train_path),sampling='author 20 random shuffled history splits/user; long histories/targets capped to tokenizer budget; no external user/item text'))
    print('PREPARED',ds,stats,flush=True)

def predict(ds):
    import torch
    from transformers import set_seed
    sys.path.insert(0,str(LF/'src'))
    from llamafactory.chat import ChatModel
    root=EXP/ds;set_seed(SEED)
    chat=ChatModel({'model_name_or_path':MODEL,'adapter_name_or_path':str(root/'adapter'),'template':'qwen','finetuning_type':'lora','infer_backend':'huggingface','infer_dtype':'bfloat16','trust_remote_code':True})
    model=chat.engine.model;tok=chat.engine.tokenizer;tok.padding_side='left';model.eval()
    source=list(rows(root/'infer.jsonl'));dest=root/'generated.jsonl'
    done={r['user_id'] for r in rows(dest)} if dest.exists() else set()
    started=time.time()
    with dest.open('a') as f:
        for n,r in enumerate(source):
            if r['user_id'] in done:continue
            # Per-user RNG makes resumed generation invariant to previous completed users.
            set_seed(SEED+int(r['user_id']))
            ids=tok.apply_chat_template([{'role':'user','content':r['instruction']}],tokenize=True,add_generation_prompt=True,return_tensors='pt')
            if not torch.is_tensor(ids):ids=ids['input_ids']
            assert ids.shape[-1]<=2048,(r['user_id'],ids.shape)
            ids=ids.to(model.device)
            with torch.inference_mode():
                output=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=200,do_sample=True,temperature=.8,top_p=.9,top_k=50,pad_token_id=tok.pad_token_id,eos_token_id=model.generation_config.eos_token_id,use_cache=True)
            text=tok.decode(output[0,ids.shape[-1]:],skip_special_tokens=True)
            f.write(json.dumps(dict(r,prediction=text,seed=SEED+int(r['user_id'])),ensure_ascii=False)+'\n');f.flush()
            if (n+1)%25==0:
                dump(root/'GENERATION_PROGRESS.json',{'done_users':n+1,'total_users':len(source),'elapsed_this_run':time.time()-started,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1048576});print('GENERATE',ds,n+1,'/',len(source),flush=True)
    assert sum(1 for _ in rows(dest))==len(source)

def augment(ds):
    import numpy as np
    c=core();ctx=c.setup(ds);a,tr,va,te,nu,ni,minor,flags,names=ctx
    root=EXP/ds;aug=root/'augmented_data';aug.mkdir(exist_ok=True);name=DS[ds]+'_llm4idrec_safe_rho050';added={};counts=collections.Counter();per=[]
    catalog=set(i for items in tr.values() for i in items);ui=names.index('unsafe_any')
    for r in rows(root/'generated.jsonl'):
        u=r['user_id'];raw=list(dict.fromkeys(int(i) for i in re.findall(r'(?<![A-Za-z0-9])i(\d+)\b',r['prediction'])));valid=[i for i in raw if i in catalog]
        counts['raw_unique_ids']+=len(raw);counts['invalid_ids']+=len(raw)-len(valid)
        if len(valid)<2:counts['rejected_single_or_empty_responses']+=1;valid=[]
        observed=set(tr.get(u,[]));blocked=observed|set(va.get(u,[]))|set(te.get(u,[]))
        unseen=[i for i in valid if i not in blocked];safe=[i for i in unseen if not flags[int(minor[u]),i,ui]]
        quota=max(1,int(round(.5*len(observed))));keep=safe[:quota]
        assert not set(keep)&blocked and all(i in catalog and not flags[int(minor[u]),i,ui] for i in keep)
        added[u]=keep;counts['new_generated_before_safety']+=len(unseen);counts['rejected_unsafe']+=len(unseen)-len(safe);counts['retained_safe']+=len(keep);counts['minor_retained' if minor[u] else 'adult_retained']+=len(keep)
        per.append({'user_id':u,'is_minor':bool(minor[u]),'train_items':len(observed),'raw_unique':len(raw),'valid_multi_ids':len(valid),'unseen':len(unseen),'safe_before_cap':len(safe),'quota':quota,'added':len(keep)})
    original=pathlib.Path(a.train_rating);dest=aug/(name+'.train.rating');shutil.copyfile(original,dest)
    with dest.open('a') as f:
        for u,items in sorted(added.items()):
            for i in items:f.write(f'{u}\t{i}\t1\t0\n')
    for split in ['valid','test']:
        p=pathlib.Path(getattr(a,split+'_rating'));shutil.copyfile(p,aug/(name+'.'+split+'.rating'))
        assert sha(p)==sha(aug/(name+'.'+split+'.rating'))
    # CF scripts expect .test.negative even in all-ranking mode if present; copy unchanged.
    neg=original.with_name(DS[ds]+'.test.negative')
    if neg.exists():shutil.copyfile(neg,aug/(name+'.test.negative'))
    n=sum(len(v) for v in tr.values());assert counts['retained_safe']>0,'No safe generated interactions; do not train a disguised unchanged baseline.'
    dump(root/'AUGMENTATION.json',{'dataset':name,'data_dir':str(aug),'original_interactions':n,'counts':counts,'actual_added_fraction':counts['retained_safe']/n,'rho_upper_cap':.5,'no_random_or_backbone_fill':True,'safety_function':'existing SafetyAdapter unsafe_any for the assigned minor/adult profile','heldout_excluded':True,'catalog':'original train-observed item identities','generator_inputs_do_not_contain_heldout':True})
    dump(root/'augmentation_by_user.json',per);dump(root/'added_interactions.json',added);print('AUGMENT',ds,counts,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--action',required=True,choices=['prepare','train','predict','augment'])
    ap.add_argument('--model-path');ap.add_argument('--llamafactory-root');ap.add_argument('--llamafactory-cli',default='llamafactory-cli')
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--dataset-prefix')
    ar=ap.parse_args();CONFIG=str(pathlib.Path(ar.config).resolve());data=Inputs(CONFIG)
    ROOT=pathlib.Path(__file__).resolve().parents[1];EXP=pathlib.Path(ar.output).resolve();EXP.mkdir(parents=True,exist_ok=True)
    LF=pathlib.Path(ar.llamafactory_root).resolve() if ar.llamafactory_root else ROOT/'LLaMA-Factory'
    MODEL=str(pathlib.Path(ar.model_path).resolve()) if ar.model_path else None
    SEED=ar.seed;DS={data.ds:ar.dataset_prefix or ('ml-1m_safe' if data.ds=='ML-1M' else 'mal_safe')}
    if ar.action in ['prepare','predict'] and not MODEL:ap.error('--model-path is required')
    if ar.action=='prepare':prepare(data.ds)
    elif ar.action=='predict':
        if not (LF/'src').is_dir():ap.error('--llamafactory-root must contain src/')
        predict(data.ds)
    elif ar.action=='augment':augment(data.ds)
    else:
        subprocess.run([ar.llamafactory_cli,'train',str(EXP/data.ds/'train.yaml')],check=True)
