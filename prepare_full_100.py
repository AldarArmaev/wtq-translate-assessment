import json
from pathlib import Path
from collections import Counter
from loaders import load_dataset,load_trace,resolve_source_table,sample_ids,load_table
from tables import prepare_cells,compare_structure
base=Path('wtq-ru-full/runs/training-fe81e969e126')
ds=load_dataset(base/'dataset/training.tsv')
root=Path('C:/Users/PC/Documents/GitHub/WTQ/WikiTableQuestions-master')
tr=base/'traces/training'
out=Path('output/full-100-stanza-back')
out.mkdir(exist_ok=True)
eligible={}; excluded=Counter()
for i,e in ds.items():
 try:
  t=load_trace(tr/f'{i}.json')
 except (OSError,ValueError):
  excluded['missing_or_invalid_trace']+=1;continue
 if not all(isinstance(x,str) and x.strip() for x in (t.get('source_question'),e.utterance,(t.get('quality') or {}).get('back_translated_question'))):
  excluded['incomplete_question_triplet']+=1;continue
 try:
  src=resolve_source_table(root,t['source_context'])
  dest=base/'dataset'/e.context
  if not dest.is_file(): raise FileNotFoundError(dest)
 except (OSError,KeyError):
  excluded['missing_table']+=1;continue
 eligible[i]=(src,dest)
assert len(eligible)>=100
ids=sample_ids(eligible,100,42)
rows=[]
for i in ids:
 a,b=map(load_table,eligible[i]); cells=prepare_cells(a,b)
 rows.append(dict(id=i,cells=len(cells),llm_text_cells=sum(c['needs_llm'] for c in cells),structure_preserved=compare_structure(a,b)['structure_preserved']))
(out/'ids.txt').write_text('\n'.join(ids)+'\n',encoding='utf-8')
summary=dict(dataset=len(ds),eligible=len(eligible),selected=len(ids),seed=42,excluded=dict(excluded),cells=sum(r['cells'] for r in rows),llm_text_cells=sum(r['llm_text_cells'] for r in rows),structure_preserved=sum(r['structure_preserved'] for r in rows))
(out/'selection.json').write_text(json.dumps({**summary,'examples':rows},indent=2),encoding='utf-8')
print(json.dumps(summary))
