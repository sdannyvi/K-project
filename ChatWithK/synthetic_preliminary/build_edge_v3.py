"""Build exploratory v3 from project-authored v2 train rows only.

No external development or sealed-test text enters training.
"""
import argparse, collections, hashlib, json, random, re
from pathlib import Path

EMOTIONS = [
    ('angry', 'The comment was sharp.'), ('sad', 'The news was disappointing.'),
    ('worried', 'The deadline is close.'), ('embarrassed', 'The meeting was uncomfortable.'),
    ('frustrated', 'The first attempt did not work.'), ('uncertain', 'The instructions are incomplete.'),
]
ROLES = ['teacher', 'nurse', 'engineer', 'parent', 'student', 'designer', 'accountant', 'researcher', 'driver', 'cook']
CLAIMS = [
    'a failure', 'worthless', 'an embarrassment', 'incompetent', 'unlovable',
    'a terrible parent', 'a fraud', 'nothing without approval', 'the problem in every situation',
    'inferior to everyone else', 'always going to be rejected', 'a burden',
    'the sort of person who ruins everything', 'not worth listening to', 'permanently defective',
]
ACTIONS = [
    'check the specific feedback and correct the part that is inaccurate',
    'ask which detail needs attention and then review the evidence',
    'write down the deadline and the next practical step',
    'compare the report with the original instructions before revising it',
    'make a short plan for the task and confirm the facts',
    'speak with the person about the concrete issue',
    'check the schedule and arrange what is needed',
    'review what happened and fix the part I can verify',
]
FACTS = [
    'The appointment is at nine. I need to leave at eight fifteen to catch the bus.',
    'The form is due tomorrow. I will sign the required page and put it in my bag.',
    'The recipe needs three onions. I will check the kitchen before going to the shop.',
    'The spreadsheet has two revised columns. I will compare them with the earlier version.',
    'The meeting room changed. I need to update my calendar and find the new room.',
    'The tire needs air. I will use the pump and check the pressure.',
    'The parcel arrives tomorrow. I can choose another delivery window.',
    'The library closes at six. I need my card and the book number.',
]
RUMINATION = [
    '. Everyone must see me this way. I keep replaying the scene and imagining what they think.',
    '. This proves the whole story of my life. I cannot stop rehearsing how bad I look.',
    '. I need to win the argument in my head. I keep returning to what they might say about me.',
]
SECOND = [
    'Then I can decide what to do with the information.',
    'The next step depends on those details.',
    'That gives me a concrete way to proceed.',
    'I can review the result when that is done.',
]

def prompt_for(row):
    return row['prompt']

def build(rows, seed):
    rng = random.Random(seed)
    base = [dict(r) for r in rows]
    train = [r for r in base if r['split'] == 'train']
    ego = [r for r in train if r['kind'] == 'ego_threat']
    assert len(ego) >= 2000
    rng.shuffle(ego)
    added=[]
    # All classes use the same authored biographical/situation prompts as v2;
    # the observed prefix itself supplies the stop/continue evidence.
    for i, src in enumerate(ego[:2400]):
        action = ACTIONS[i % len(ACTIONS)]
        second = SECOND[(i // len(ACTIONS)) % len(SECOND)]
        claim = CLAIMS[i % len(CLAIMS)]
        if i < 600:
            emotion, cue = EMOTIONS[i % len(EMOTIONS)]
            stem = f'I am {emotion} about this.'
            chosen = f'{stem} I need to {action}. {second}'
            family = 'emotion_without_self_judgment'
            rejected = f'{stem} Why does this always happen to me? Everyone must think badly of me. I keep replaying it.'
        elif i < 1200:
            match = re.search(r'You work as (a|an) (.+?) in ', src['biography'])
            stem = f'I work as {match.group(1)} {match.group(2)}.' if match else 'I have a job and a schedule.'
            chosen = f'{stem} I need to {action}. {second}'
            family = 'factual_self_description'
            rejected = f'{stem} I must prove that I matter more than everyone else. I keep imagining how they see me.'
        elif i < 1800:
            # The shared function-word prefix must continue until there is a real claim.
            stem = 'I am'
            chosen = f'{stem} going to {action}. {second}'
            family = 'incomplete_self_claim'
            rejected = f'{stem} {claim}. Everyone can see it. I keep replaying their reaction.'
        else:
            stem = f'I am {claim}'
            chosen = stem
            family = 'complete_self_claim_stop'
            rejected = stem + RUMINATION[i % len(RUMINATION)]
        added.append(dict(id=f'EDGE-V3-{i:05d}', split='train', kind='ego_threat' if i>=1800 else 'practical',
            family=family, biography=src['biography'], event=src['event'], prompt=prompt_for(src),
            positive_practical=chosen if i<1800 else f'I need to {action}. {second}',
            positive_cessation=chosen if i>=1800 else None,
            cessation_cut_words=len(chosen.split()) if i>=1800 else None,
            negative_rumination=rejected, edge_target=family))
    # Add multi-sentence factual controls with neutral situations and no self-story.
    neutral=[r for r in train if r['kind'] in ('practical','description')]
    for i in range(400):
        src=neutral[i % len(neutral)]
        chosen=src['positive_practical'].rstrip() + ' Then I can check whether anything else is needed.'
        added.append(dict(id=f'EDGE-V3-F-{i:05d}', split='train', kind='practical', family='multi_sentence_factual',
            biography=src['biography'], event=src['event'], prompt=src['prompt'],
            positive_practical=chosen, positive_cessation=None, cessation_cut_words=None,
            negative_rumination='I never manage anything correctly. Everyone must judge me for it. I keep replaying the failure.',
            edge_target='multi_sentence_factual'))
    # For cessation rows, the practical branch needs a genuinely different
    # trajectory. It is not a second target for the same completed-claim prefix.
    output=base+added
    assert len({r['id'] for r in output}) == len(output)
    assert all(r['split']=='train' for r in added)
    assert all(r['negative_rumination'] for r in added)
    return output

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--seed',type=int,default=20261105);a=p.parse_args()
    rows=build([json.loads(x) for x in a.source.open()],a.seed)
    payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    a.output.write_text(payload)
    m=dict(status='exploratory_synthetic_v3',seed=a.seed,source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),sha256=hashlib.sha256(payload.encode()).hexdigest(),rows=len(rows),train_rows=sum(r['split']=='train' for r in rows),added_families=dict(collections.Counter(r.get('edge_target') for r in rows if r['id'].startswith('EDGE-V3-'))),external_data_used=False,sealed_test_accessed=False)
    a.manifest.write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m,indent=2))
if __name__=='__main__': main()
