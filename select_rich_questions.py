import json
from pathlib import Path
from loaders import load_dataset,sample_ids,load_trace
from stanza_entities import StanzaEntityExtractor
base=Path('wtq-ru-full/runs/training-fe81e969e126'); ds=load_dataset(base/'dataset/training.tsv')
out=Path('output/stanza-rich-selection');out.mkdir(exist_ok=True)
extractor=StanzaEntityExtractor(Path('.models'))
rows=[]
proper={'PERSON','ORG','GPE','LOC','FAC','NORP','PRODUCT','EVENT','WORK_OF_ART','LAW','LANGUAGE'}
for ex_id in sample_ids(set(ds) & {p.stem for p in (base/'traces/training').glob('*.json')},500,42):
 t=load_trace(base/'traces/training'/f'{ex_id}.json');q=t.get('source_question','')
 if not q or not (t.get('quality') or {}).get('back_translated_question'):continue
 es=extractor.extract([q],'en')[q]
 named=[e for e in es if e['type'] in proper]
 rows.append({'id':ex_id,'source_question':q,'entities':es,'proper_count':len(named),'n_entities':len(es)})
rows.sort(key=lambda r:(-r['proper_count'],-len({e['type'] for e in r['entities']}),-r['n_entities'],r['id']))
(out/'candidates.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
for r in rows[:10]:print(json.dumps(r,ensure_ascii=False),flush=True)

