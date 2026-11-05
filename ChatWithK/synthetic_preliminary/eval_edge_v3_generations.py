"""Development-only free continuations from fixed partial published examples."""
import argparse,hashlib,json,random
from pathlib import Path
import torch
from peft import PeftModel
from pnas_pipeline.models import load_text_model

p=argparse.ArgumentParser();p.add_argument('--items',required=True);p.add_argument('--output',required=True);p.add_argument('--adapter',required=True);p.add_argument('--seed',type=int,default=20261105);p.add_argument('--pairs',type=int,default=20);a=p.parse_args()
rows=[json.loads(x) for x in open(a.items)]
by_kind={k:[r for r in rows if r['dataset']=='patternreframe' and r['kind']==k] for k in ('thought','reframe')}
rng=random.Random(a.seed)
ids=sorted(set(r['id'].split(':')[0] for r in by_kind['thought']))
selected=set(rng.sample(ids,a.pairs))
items=[r for k in ('thought','reframe') for r in by_kind[k] if r['id'].split(':')[0] in selected]
assert len(items)==2*a.pairs
model,tok=load_text_model('Qwen/Qwen3-14B','40c069824f4251a91eefaf281ebe4c544efd3e18','4bit')
model=PeftModel.from_pretrained(model,a.adapter).eval()
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w',encoding='utf-8') as f:
 for j,item in enumerate(items):
  cont=tok.encode(item['text'],add_special_tokens=False)
  k=max(2,min(len(cont)-2,round(0.5*len(cont))))
  prefix=tok.decode(cont[:k])
  prompt=tok.apply_chat_template([{'role':'system','content':'You are an assistant.'},{'role':'user','content':item['context']}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
  ids=torch.tensor([prompt+cont[:k]],device='cuda:0')
  with torch.inference_mode(): gen=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=128,do_sample=False,eos_token_id=tok.eos_token_id,pad_token_id=tok.eos_token_id)[0,len(prompt)+k:].tolist()
  eos=tok.eos_token_id in gen
  content=gen[:gen.index(tok.eos_token_id)] if eos else gen
  f.write(json.dumps({'id':item['id'],'kind':item['kind'],'prefix':prefix,'source_suffix':tok.decode(cont[k:]),'continuation':tok.decode(content,skip_special_tokens=True),'new_tokens':len(content),'native_eos':eos,'context_sha256':hashlib.sha256(item['context'].encode()).hexdigest()},ensure_ascii=False)+'\n')
  if (j+1)%10==0:print(j+1,flush=True)
