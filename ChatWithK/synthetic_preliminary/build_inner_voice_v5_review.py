"""Build an offline PI review form for v5 proposed cessation streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Inner-voice dataset review</title><style>
:root {font:16px/1.5 system-ui,sans-serif;color:#192334;background:#f3f6fa}
body {max-width:1080px;margin:0 auto;padding:20px}
main {background:white;border:1px solid #dce3ee;border-radius:15px;padding:26px;box-shadow:0 8px 25px #16243a0c}
h1 {margin:0 0 8px;font-size:1.6rem} h2 {font-size:1.1rem;margin:0}
.muted {color:#536275}.bar {height:9px;background:#e8edf4;border-radius:9px;overflow:hidden;margin:12px 0 22px}
.fill {height:100%;background:#1c7865;width:0}.stream {border:1px solid #dce3ee;border-radius:11px;padding:18px;margin:13px 0}
.stream.reviewed {border-color:#88bba9}.text {font-size:1.16rem;line-height:2.15;margin:15px 0}
.word {font:inherit;color:inherit;border:0;border-radius:4px;background:none;padding:2px 1px;cursor:pointer}
.word:hover,.word:focus-visible {background:#dce8ff;outline:2px solid #517fc4}
.marker {display:inline-block;height:1.25em;vertical-align:middle;margin:0 3px}
.marker.proposed {border-right:3px solid #b7781d}.marker.reviewer {border-right:3px solid #137866}
.controls {display:flex;flex-wrap:wrap;align-items:center;gap:5px}
button.action {font:inherit;border:1px solid #bcc9d9;border-radius:8px;padding:8px 12px;background:white;cursor:pointer}
button.action:hover {background:#ebf1f8}.primary {background:#155f52!important;border-color:#155f52!important;color:white}
.review-status {font-weight:600;margin:10px 0 0}.legend {display:flex;gap:20px;flex-wrap:wrap}.legend b {font-weight:700}
.spacer {flex:1} input[type=number] {width:62px;font:inherit;padding:7px}
footer {margin-top:18px;color:#536275;font-size:.86rem}
</style></head><body><main>
<h1>Review the inner-voice pilot</h1>
<p class="muted">Each card shows one proposed cessation stream. The amber mark is my proposed cutoff. Approve it, click a different word to cut after it, or choose no cessation. Your decision is not counted until you act. The factual and practical continuations are omitted because you approved them.</p>
<p class="legend"><span><b style="color:#b7781d">│</b> Proposed cutoff</span><span><b style="color:#137866">│</b> Your cutoff</span></p>
<div id="progress"></div><div class="bar"><div id="fill" class="fill"></div></div>
<h2 id="card-title"></h2><div id="streams"></div>
<div class="controls"><span class="spacer"></span><button class="action" id="prev">← Previous</button><button class="action" id="next">Next →</button><button class="action" id="unreviewed">Next unreviewed</button></div>
<div class="controls" style="margin-top:16px"><label for="jump">Card</label><input type="number" min="1" max="40" id="jump"><button class="action" id="go">Go</button><span class="spacer"></span><button class="action primary" id="download">Download review CSV</button></div>
<footer>Forty proposed cessation streams are shown, one per distinct situation. The corpus repeats each across four neutral personas; you need review it only once. Progress is saved locally in this browser. Download the CSV regularly as a backup. Nothing is uploaded.</footer>
</main><script id="data" type="application/json">__DATA__</script><script>
"use strict";
const cards=JSON.parse(document.getElementById('data').textContent);
const key='kproject-inner-voice-v5-cessation-review-__HASH__';
let decisions={};try{decisions=JSON.parse(localStorage.getItem(key)||'{}')}catch(_){decisions={}}
let current=0;const el=id=>document.getElementById(id);
const words=s=>[...s.matchAll(/\S+/gu)].map(m=>({word:m[0],start:m.index,end:m.index+m[0].length}));
function persist(){try{localStorage.setItem(key,JSON.stringify(decisions))}catch(_){alert('Local saving failed; download a CSV now.')}}
function review(item,decision,wordIndex){const parts=words(item.text),method=decision===item.proposed_decision&&(decision==='none'||wordIndex===item.proposed_word_index)?'approved':'changed';decisions[item.id]={decision,word_index:wordIndex||'',word:wordIndex?parts[wordIndex-1].word:'',method,reviewed_at:new Date().toISOString()};persist();render()}
function proposal(item){return item.proposed_decision==='cut'?{decision:'cut',word_index:item.proposed_word_index}:{decision:'none',word_index:''}}
function render(){
 const card=cards[current],all=cards.flatMap(c=>c.streams),done=all.filter(x=>decisions[x.id]?.decision).length;
 el('progress').textContent=`${done} of ${all.length} streams reviewed · Card ${current+1} of ${cards.length}`;
 el('fill').style.width=`${done/all.length*100}%`;el('card-title').textContent=`${card.topic} · ${card.card_id}`;el('jump').value=current+1;
 const container=el('streams');container.replaceChildren();
 card.streams.forEach((item,n)=>{
   const d=decisions[item.id],section=document.createElement('section');section.className='stream'+(d?' reviewed':'');
   const heading=document.createElement('h3');heading.textContent=`${n+1}. ${item.variant.replaceAll('_',' ')}`;section.append(heading);
   const label=document.createElement('div');label.className='muted';label.textContent=item.proposed_decision==='cut'?`Proposed: cease after word ${item.proposed_word_index} (“${item.proposed_word}”)`:'Proposed: no cessation';section.append(label);
   const text=document.createElement('div');text.className='text';let cursor=0;
   words(item.text).forEach((part,i)=>{
     text.append(document.createTextNode(item.text.slice(cursor,part.start)));
     const b=document.createElement('button');b.type='button';b.className='word';b.textContent=part.word;b.title=`Choose cutoff after word ${i+1}`;b.onclick=()=>review(item,'cut',i+1);text.append(b);
     if(item.proposed_decision==='cut'&&item.proposed_word_index===i+1){const m=document.createElement('span');m.className='marker proposed';m.title='Proposed cutoff';text.append(m)}
     if(d?.decision==='cut'&&d.word_index===i+1){const m=document.createElement('span');m.className='marker reviewer';m.title='Your cutoff';text.append(m)}
     cursor=part.end;
   });text.append(document.createTextNode(item.text.slice(cursor)));section.append(text);
   const controls=document.createElement('div');controls.className='controls';
   const approve=document.createElement('button');approve.className='action';approve.textContent='Approve proposal';approve.onclick=()=>{const p=proposal(item);review(item,p.decision,p.word_index)};controls.append(approve);
   const none=document.createElement('button');none.className='action';none.textContent='No cessation';none.onclick=()=>review(item,'none','');controls.append(none);
   const clear=document.createElement('button');clear.className='action';clear.textContent='Clear';clear.onclick=()=>{delete decisions[item.id];persist();render()};controls.append(clear);section.append(controls);
   const status=document.createElement('p');status.className='review-status';status.textContent=!d?'Awaiting review':d.decision==='none'?`Your decision: no cessation (${d.method})`:`Your decision: cut after word ${d.word_index} “${d.word}” (${d.method})`;section.append(status);container.append(section);
 });
 el('prev').disabled=current===0;el('next').disabled=current===cards.length-1;
}
el('prev').onclick=()=>{if(current>0){current--;render()}};
el('next').onclick=()=>{if(current<cards.length-1){current++;render()}};
el('unreviewed').onclick=()=>{let i=cards.findIndex((c,i)=>i>current&&c.streams.some(x=>!decisions[x.id]?.decision));if(i<0)i=cards.findIndex(c=>c.streams.some(x=>!decisions[x.id]?.decision));if(i>=0){current=i;render()}};
el('go').onclick=()=>{const n=Number(el('jump').value);if(Number.isInteger(n)&&n>=1&&n<=cards.length){current=n-1;render()}};
el('download').onclick=()=>{
 const header=['card_id','topic','variant','id','text','proposed_decision','proposed_cut_after_word_index','proposed_cut_after_word','reviewer_decision','reviewer_cut_after_word_index','reviewer_cut_after_word','review_method','reviewed_at'];
 const q=x=>'"'+String(x??'').replaceAll('"','""')+'"';const lines=[header.map(q).join(',')];
 cards.flatMap(c=>c.streams).forEach(x=>{const d=decisions[x.id]||{};lines.push([x.card_id,x.topic,x.variant,x.id,x.text,x.proposed_decision,x.proposed_word_index||'',x.proposed_word||'',d.decision||'',d.word_index||'',d.word||'',d.method||'',d.reviewed_at||''].map(q).join(','))});
 const blob=new Blob(['\ufeff'+lines.join('\r\n')+'\r\n'],{type:'text/csv;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='__CSV_NAME__';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};render();
</script></body></html>'''


def build(source: Path, output: Path) -> list[dict]:
    rows = [json.loads(line) for line in source.open(encoding="utf-8")]
    cards: dict[str, dict] = {}
    for row in rows:
        card = cards.setdefault(row["card_id"], {"card_id": row["card_id"], "topic": row["topic"], "streams": {}})
        text = row["full_candidate_text"]
        proposed_index = len(re.findall(r"\S+", row["target_text"])) if row["stop_at_target_end"] else None
        item = {
            "card_id": row["card_id"], "topic": row["topic"], "variant": row["variant"],
            "id": f"{row['card_id']}:{row['variant']}", "text": text,
            "proposed_decision": "cut" if proposed_index else "none",
            "proposed_word_index": proposed_index,
            "proposed_word": re.findall(r"\S+", row["target_text"])[-1] if proposed_index else None,
        }
        if row["variant"] != "rumination_stop":
            continue
        if row["variant"] in card["streams"]:
            assert item == card["streams"][row["variant"]], "persona expansion changed stream text"
        card["streams"][row["variant"]] = item
    result = []
    for card in cards.values():
        assert set(card["streams"]) == {"rumination_stop"}
        result.append({**card, "streams": [card["streams"]["rumination_stop"]]})
    assert len(result) == 40 and sum(len(c["streams"]) for c in result) == 40
    data = json.dumps(result, ensure_ascii=False).replace("<", "\\u003c")
    digest = hashlib.sha256(data.encode()).hexdigest()[:16]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        HTML.replace("__DATA__", data).replace("__HASH__", digest).replace("__CSV_NAME__", output.stem + ".csv"),
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output)
