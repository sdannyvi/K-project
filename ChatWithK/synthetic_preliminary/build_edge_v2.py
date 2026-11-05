"""Exploratory synthetic supplement correcting premature-cessation supervision.

Uses only project-authored situations, never published evaluation text.
"""
import argparse,hashlib,json,random
from collections import Counter
from pathlib import Path

PSYCH_CLAIMS=[
 "am a failure at this job", "am fundamentally inadequate", "am a terrible colleague",
 "am not worth hearing", "am an embarrassment to everyone", "am a fraud",
 "am a disappointment", "must be incompetent", "am unlovable when I make mistakes",
 "am always the problem", "am a bad parent", "am the sort of person who ruins everything",
 "am worthless unless they approve", "will never be good enough", "am nothing without their praise",
 "am defective in some permanent way", "am a burden to the people around me",
 "must prove that I am superior", "am being rejected as a person", "am the victim in every disagreement",
]
PRACTICAL_CONTINUATIONS=[
 "need to check the specific facts before deciding what to change.",
 "can ask which part needs correction and review the evidence.",
 "should write down the concrete issue and address it directly.",
 "will compare the feedback with the actual work and revise what is inaccurate.",
 "need the relevant details, deadline, and next step.",
 "can separate what happened from guesses about what others think.",
 "will check the schedule and make a workable plan.",
 "need to speak with the person and clarify the practical concern.",
]
STEMS=[
 "The situation is uncomfortable. I", "I heard what happened. I",
 "That comment caught my attention. I", "There is a concrete event to deal with. I",
 "I am noticing the reaction. I", "The next thought is beginning. I",
]
RUMINATION_TAILS=[
 " Everyone will remember this and judge me by it. I keep replaying what I should have said.",
 " If I think through every detail again, perhaps I can repair my image in their minds.",
 " I need to defend myself, even in an imagined argument that never ends.",
 " The more I rehearse the scene, the more it seems to define my whole identity.",
]
NEUTRAL_CASES=[
 ("The bus route has changed for tomorrow's commute.", "The route has changed. I", "need to check the new stop and departure time before leaving."),
 ("A parcel is due to arrive while you are at work.", "The parcel arrives tomorrow. I", "can arrange for a neighbor to receive it or choose another delivery slot."),
 ("You are preparing vegetables for dinner.", "The vegetables are on the counter. I", "need to wash and cut them before heating the pan."),
 ("Your bicycle tire needs air before the morning ride.", "The tire feels soft. I", "should find the pump and check the pressure listed on the sidewall."),
 ("A colleague sent a revised spreadsheet.", "The revised file is here. I", "will compare the changed cells with the prior version."),
 ("You need to collect a library book before closing time.", "The library closes soon. I", "need to confirm the opening hours and take my card."),
 ("A friend suggested meeting at a new cafe.", "The cafe is unfamiliar. I", "can check its address and choose a meeting time."),
 ("Your child has a routine school form to return.", "The form is on the table. I", "will read it, sign the required section, and put it in the school bag."),
 ("You are arranging chairs before a meeting.", "The room needs more seats. I", "will count the attendees and bring enough chairs."),
 ("A recipe calls for one ingredient you do not have.", "The recipe needs an ingredient. I", "can check for a suitable substitute or visit the nearby shop."),
 ("You notice that a lamp is not working.", "The lamp is off. I", "will check the switch and bulb before calling for repair."),
 ("An appointment was moved to a different room.", "The room number changed. I", "need to update the calendar and find the new location."),
]
DESCRIPTIVE_CASES=[
 ("You saw a neighbor repainting a fence.","My neighbor was repainting the fence. It","looked freshly prepared, with a cloth spread under the paint cans."),
 ("You visited a quiet park after lunch.","The park was quiet after lunch. It","had a few walkers and several empty benches near the path."),
 ("You watched a routine school rehearsal.","The rehearsal started on time. It","included a short introduction and two songs before the break."),
 ("You received a routine weather alert.","The weather alert came this morning. It","said rain was likely in the afternoon, so I took an umbrella."),
 ("You noticed the shop had changed its display.","The shop window changed overnight. It","now shows winter coats beside a row of folded scarves."),
 ("You heard a friend describe a concert.","My friend described the concert. It","began with a short set and ended before the last train."),
]

def build(source,seed):
 rng=random.Random(seed);rows=[]
 for r in source:
  row=dict(r)
  if row['kind']=='ego_threat':
   stem=rng.choice(STEMS);claim=rng.choice(PSYCH_CLAIMS)
   row['positive_cessation']=stem+' '+claim
   row['positive_practical']=stem+' '+rng.choice(PRACTICAL_CONTINUATIONS)
   row['negative_rumination']=row['positive_cessation']+'.'+rng.choice(RUMINATION_TAILS)
   row['cessation_cut_words']=len(row['positive_cessation'].split())
   row['edge_target']='stop_after_complete_psychological_claim'
  rows.append(row)
 for i in range(1200):
  if i%3==0:
   event,stem,tail=DESCRIPTIVE_CASES[(i//3)%len(DESCRIPTIVE_CASES)];kind='description'
  else:
   event,stem,tail=NEUTRAL_CASES[i%len(NEUTRAL_CASES)];kind='practical'
  name=rng.choice(['Elena','Marcus','Priya','Noah','Maya','Omar','Leila','Daniel','Sofia','Thomas'])
  job=rng.choice(['teacher','technician','designer','nurse','accountant','student','shop manager','researcher'])
  bio=f'You are {name}, {rng.randint(24,63)} years old, and work as a {job}.'
  prompt=f'Biography: {bio}\n\nCurrent situation: {event}\n\nGive the content of your mind at this moment. Write only the stream itself.'
  row={'id':f'EDGE-N-{i:05d}','split':'train','kind':kind,'family':f'edge_{kind}_{i%18}',
       'biography':bio,'event':event,'prompt':prompt,
       'positive_practical':stem+' '+tail,'positive_cessation':None,'cessation_cut_words':None,
       'negative_rumination':f'{stem} Why does this always happen to me? I keep treating it as proof that my life is going wrong. I imagine how others will judge me. The practical details disappear behind that repeated story.',
       'edge_target':'continue_neutral_or_practical'}
  rows.append(row)
 ids=[x['id'] for x in rows]
 assert len(ids)==len(set(ids))
 assert all(x['split']!='train' or x['positive_cessation'] is None or x['positive_cessation'].split()[-1] not in {'I','it','the','and','to'} for x in rows)
 return rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--seed',type=int,default=20261105);a=p.parse_args()
 rows=build([json.loads(x) for x in a.source.open()],a.seed)
 payload=''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(payload)
 manifest={'status':'exploratory_synthetic_v2','source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),
           'sha256':hashlib.sha256(payload.encode()).hexdigest(),'seed':a.seed,'rows':len(rows),
           'train_rows':sum(x['split']=='train' for x in rows),'edge_counts':dict(Counter(x.get('edge_target','original') for x in rows)),
           'external_test_text_used':False,'target_rule':'EOS only after complete psychological claim; shared ambiguous stems continued in both branches'}
 a.manifest.write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
