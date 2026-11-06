"""Build an offline, blind word-boundary annotation form from development rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cessation boundary annotation</title>
<style>
  :root { font: 17px/1.5 system-ui, sans-serif; color: #18243a; background: #f3f6fb; }
  body { max-width: 980px; margin: auto; padding: 22px; }
  main { background: white; border: 1px solid #dce3ef; border-radius: 16px; padding: 28px; box-shadow: 0 8px 25px #18243a0d; }
  h1 { font-size: 1.65rem; margin: 0 0 8px; }
  .muted { color: #56647a; }
  .bar { height: 9px; background: #e7edf5; border-radius: 9px; overflow: hidden; margin: 12px 0; }
  .fill { height: 100%; background: #2e766b; width: 0; }
  .text { font-size: 1.3rem; line-height: 2.25; margin: 30px 0; }
  .word { font: inherit; color: inherit; border: 0; border-radius: 5px; background: transparent; cursor: pointer; padding: 3px 1px; }
  .word:hover, .word:focus-visible { background: #dce8ff; outline: 2px solid #477ac7; }
  .word.chosen { background: #d2f4e2; }
  .marker { display: inline-block; border-right: 3px solid #0e7b62; height: 1.3em; vertical-align: middle; margin: 0 4px; }
  button.action { border: 1px solid #c6d2e2; border-radius: 9px; background: white; color: #18243a; padding: 10px 14px; font: inherit; cursor: pointer; margin: 3px; }
  button.action:hover { background: #eaf0f8; }
  button.primary { background: #175d52; color: white; border-color: #175d52; }
  button.primary:hover { background: #0d473f; }
  button.selected { outline: 3px solid #0e7b62; }
  .status { min-height: 28px; font-weight: 600; }
  .controls { display: flex; flex-wrap: wrap; align-items: center; gap: 5px; }
  .spacer { flex: 1; }
  input[type=number] { width: 65px; font: inherit; padding: 8px; border: 1px solid #c6d2e2; border-radius: 8px; }
  footer { font-size: .85rem; color: #56647a; margin-top: 20px; }
</style></head><body><main>
<h1>Cessation boundary annotation</h1>
<p class="muted">Choose the exact word <strong>after</strong> which speech should cease, or select <strong>No cessation</strong>. You can change an answer at any time. The source category and model scores are hidden while you annotate.</p>
<div id="progress"></div><div class="bar"><div id="fill" class="fill"></div></div>
<p id="item-id" class="muted"></p><div id="text" class="text"></div>
<div class="controls"><button id="none" class="action">No cessation</button><button id="clear" class="action">Clear this answer</button></div>
<p id="status" class="status"></p>
<div class="controls"><button id="prev" class="action">← Previous</button><button id="next" class="action">Next →</button><button id="unanswered" class="action">Next unanswered</button><span class="spacer"></span><label for="jump">Item</label><input id="jump" type="number" min="1" max="200"><button id="go" class="action">Go</button></div>
<hr><button id="download" class="action primary">Download CSV</button>
<footer>Works offline. Answers are saved in this browser under a dataset-specific key; download the CSV regularly as a backup. The CSV includes all 200 items, with blank decisions for unanswered items. No annotations are sent anywhere.</footer>
</main><script id="items" type="application/json">__DATA__</script><script>
"use strict";
const items = JSON.parse(document.getElementById('items').textContent);
const key = 'kproject-cessation-annotations-__HASH__';
let answers = {};
try { answers = JSON.parse(localStorage.getItem(key) || '{}'); } catch (_) { answers = {}; }
let current = 0;
const el = id => document.getElementById(id);
const words = text => [...text.matchAll(/\S+/gu)].map(m => ({word:m[0], start:m.index, end:m.index+m[0].length}));
function save() { try { localStorage.setItem(key, JSON.stringify(answers)); } catch (_) { alert('Browser storage failed. Download the CSV now.'); } }
function setAnswer(value) { answers[items[current].id] = {...value, annotated_at: new Date().toISOString()}; save(); render(); }
function render() {
  const item=items[current], answer=answers[item.id];
  const count=items.filter(x => answers[x.id] && answers[x.id].decision).length;
  el('progress').textContent=`${count} of ${items.length} annotated · Item ${current+1} of ${items.length}`;
  el('fill').style.width=`${count/items.length*100}%`;
  el('item-id').textContent=item.id;
  el('jump').value=current+1;
  const box=el('text'); box.replaceChildren();
  const parts=words(item.text); let cursor=0;
  parts.forEach((part,i) => {
    box.append(document.createTextNode(item.text.slice(cursor,part.start)));
    const b=document.createElement('button'); b.type='button'; b.className='word';
    if(answer?.decision==='cut' && answer.word_index===i+1) b.classList.add('chosen');
    b.textContent=part.word; b.title=`Stop after word ${i+1}: ${part.word}`;
    b.onclick=()=>setAnswer({decision:'cut',word_index:i+1,word:part.word}); box.append(b);
    if(answer?.decision==='cut' && answer.word_index===i+1) { const mark=document.createElement('span'); mark.className='marker'; mark.title='Cessation here'; box.append(mark); }
    cursor=part.end;
  });
  box.append(document.createTextNode(item.text.slice(cursor)));
  el('none').classList.toggle('selected',answer?.decision==='none');
  el('status').textContent=answer?.decision==='cut' ? `Marked: cease after word ${answer.word_index} (“${answer.word}”)` : answer?.decision==='none' ? 'Marked: no cessation' : 'Unanswered';
  el('prev').disabled=current===0; el('next').disabled=current===items.length-1;
}
el('none').onclick=()=>setAnswer({decision:'none',word_index:'',word:''});
el('clear').onclick=()=>{delete answers[items[current].id];save();render();};
el('prev').onclick=()=>{if(current>0){current--;render();}};
el('next').onclick=()=>{if(current<items.length-1){current++;render();}};
el('unanswered').onclick=()=>{const n=items.findIndex((x,i)=>i>current && !answers[x.id]?.decision); const wrap=items.findIndex(x=>!answers[x.id]?.decision); if(n>=0||wrap>=0){current=n>=0?n:wrap;render();}};
el('go').onclick=()=>{const n=Number(el('jump').value);if(Number.isInteger(n)&&n>=1&&n<=items.length){current=n-1;render();}};
el('download').onclick=()=>{
  const header=['id','pair_id','kind','text','decision','cut_after_word_index','cut_after_word','annotated_at'];
  const quote=x=>'"'+String(x ?? '').replaceAll('"','""')+'"';
  const lines=[header.map(quote).join(',')];
  items.forEach(x=>{const a=answers[x.id]||{};lines.push([x.id,x.pair_id,x.kind,x.text,a.decision||'',a.word_index||'',a.word||'',a.annotated_at||''].map(quote).join(','));});
  const blob=new Blob(['\ufeff'+lines.join('\r\n')+'\r\n'],{type:'text/csv;charset=utf-8'});
  const url=URL.createObjectURL(blob), link=document.createElement('a');link.href=url;link.download='cessation_annotations_200.csv';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
render();
</script></body></html>'''


def build(source: Path, output: Path) -> None:
    items = json.loads(source.read_text(encoding="utf-8"))
    assert len(items) == 200
    assert len({item["id"] for item in items}) == 200
    assert {kind: sum(item["kind"] == kind for item in items) for kind in ("thought", "reframe")} == {
        "thought": 100, "reframe": 100
    }
    data = json.dumps(items, ensure_ascii=False).replace("<", "\\u003c")
    digest = hashlib.sha256(data.encode()).hexdigest()[:16]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(HTML.replace("__DATA__", data).replace("__HASH__", digest), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output)
