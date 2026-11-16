# Possible datasets

Found by six parallel web searches (2026-10-26). "Example: none" means no sample text could be retrieved (gated or not shown). **Unverified** = link or access detail not confirmed.

Key gaps: no public set pairs real therapy transcripts with PHQ-9/GAD-7 or per-session outcomes. No clean anhedonia or hopelessness corpus exists. Nothing labels "which trait a CBT technique targets".

---

# Part A. For training the trait LoRAs (patient speech)

### ReDSM5
- **Link:** https://huggingface.co/datasets/irlab-udc/redsm5
- **What it is:** Reddit posts where a licensed psychologist labelled each sentence against DSM-5 depression symptoms. Each label has a clinical explanation.
- **Inside:** 1,484 posts with labels such as ANHEDONIA, SLEEP_ISSUES, FATIGUE, WORTHLESSNESS and SUICIDAL_THOUGHTS. `status` is 1 for evidence and 0 for a clinician-annotated absence.
- **Example:** "I have trouble sleeping every night" -> `SLEEP_ISSUES`, status 1.
- **Access:** gated; email an agreement form. Fit: HIGH (anhedonia, sleep, fatigue).

### DepressionEmo
- **Link:** https://github.com/abuBakarSiddiqurRahman/DepressionEmo
- **What it is:** Long Reddit posts multi-labelled with 8 depression-related emotions. Labels came from zero-shot classifiers, checked by annotators.
- **Inside:** About 6,037 posts with labels including hopelessness, emptiness, worthlessness, loneliness, sadness and suicide intent.
- **Example:** a post tagged `["emptiness","hopelessness"]` (text omitted, it describes suicidal intent).
- **Access:** public repo, "All Rights Reserved"; research use only. Fit: HIGH for hopelessness (the only real source).

### PRIMATE
- **Link:** https://github.com/primate-mh/Primate2022
- **What it is:** Reddit depression posts labelled yes/no for whether each of the 9 PHQ-9 items is evidenced. Crowd workers annotated it, with a professional checking.
- **Inside:** 2,003 posts in `primate_dataset.json`. A later paper found the loss-of-interest labels noisy.
- **Example:** "Lately I've been feeling really low." -> yes for feeling down.
- **Access:** sign a PDF agreement; CC BY-SA 4.0. Fit: HIGH (PHQ-9 items), medium for anhedonia.

### Reddit-Anhedonia (PRIMATE re-annotation)
- **Link:** https://huggingface.co/datasets/tartuNLP/reddit-anhedonia
- **What it is:** A mental health professional re-labelled PRIMATE posts for the PHQ-9 lack-of-interest item. They added evidence spans and found many false positives in the original labels.
- **Inside:** Under 1,000 records with fields `answerable`, `mentioned`, `writer_symptom` and `quote` offsets. It holds labels only, no post text, so you need PRIMATE too.
- **Example:** `{"primate_id":1394,"answerable":0,"mentioned":1,"writer_symptom":1,"quote":[[1537,1710]]}`
- **Access:** CC BY 4.0. Fit: HIGH for anhedonia, but small (use as a test set).

### DepreSym (eRisk 2023)
- **Link:** https://arxiv.org/abs/2308.10758 (paper). Data link **unverified**.
- **What it is:** Reddit sentences judged for relevance to the 21 BDI-II symptoms by three assessors, one a psychologist.
- **Inside:** 21,580 judged sentences out of a 3.8M-sentence eRisk collection. Relevance means explicit and about the writer's own condition.
- **Example:** none.
- **Access:** eRisk user agreement (https://erisk.irlab.org/eRisk2025.html). Fit: HIGH (anhedonia, sleep, fatigue; BDI-II pessimism for hopelessness).

### DAIC-WOZ / E-DAIC
- **Link:** https://dcapswoz.ict.usc.edu/
- **What it is:** Clinical interviews with a virtual interviewer, "Ellie", run by a human operator. They support diagnosis of depression, anxiety and PTSD.
- **Inside:** 189 sessions of 7-33 min with transcripts and nonverbal features. PHQ-8 is the label (from other sources, not stated on the page).
- **Example:** none (gated).
- **Access:** request form, non-profit only. Fit: HIGH for question->answer style, but small.

### D4 (Chinese depression-diagnosis dialogues)
- **Link:** https://x-lance.github.io/D4/
- **What it is:** Doctor-patient diagnosis conversations built from ICD-11 and DSM-5. Psychiatrists wrote the summaries and diagnoses (EMNLP 2022).
- **Inside:** 1,339 dialogues (average 21.6 turns): 430 control, 342 mild, 368 moderate, 199 severe. Chinese only.
- **Example:** none.
- **Access:** signed DUA by email; research only. Fit: MED-HIGH (severity-graded, needs translation).

### ESConv
- **Link:** https://github.com/thu-coai/Emotional-Support-Conversation (also https://huggingface.co/datasets/thu-coai/esconv)
- **What it is:** Crowdsourced emotional-support chats between a help-seeker and a supporter using labelled strategies (ACL 2021).
- **Inside:** 1,300 dialogues with problem types such as depression and anxiety. Seekers rate their emotion intensity before and after, and `FailedESConv.json` has 196 negative cases.
- **Example:** "I am having a lot of anxiety about quitting my current job." (initial intensity 5, final 1).
- **Access:** CC BY-NC 4.0, academic use. Fit: MED-HIGH (also for the gate, see Part B).

### CACTUS
- **Link:** https://huggingface.co/datasets/DLI-Lab/cactus
- **What it is:** LLM-simulated CBT counselling dialogues. Each is built from a client profile and a chosen CBT technique (EMNLP Findings 2024).
- **Inside:** 31,577 synthetic rows with `intake_form`, `thought`, `patterns`, `cbt_technique` (34 classes), `cbt_plan`, `attitude` and `dialogue`.
- **Example:** thought "I'm really into my crystals. People will probably always think I am weird." -> technique *Alternative Perspective*.
- **Access:** ungated, GPL. Fit: HIGH (ready-made persona client speech, plus technique labels for Part B).

### Patient-Psi-CM
- **Link:** https://github.com/ruiyiw/patient-psi
- **What it is:** CBT cognitive models (patient profiles) written by clinical psychologists, used to prompt an LLM patient (EMNLP 2024).
- **Inside:** The public repo has only example profiles in `python/data/profiles.json` plus generation code. The full set is by request form.
- **Example:** none retrieved.
- **Access:** Google form; licence unclear. Fit: MED-HIGH as a persona template, not a speech corpus.

### Eeyore
- **Link:** https://arxiv.org/abs/2503.00018
- **What it is:** An 8B Llama-3.1 depression-patient simulator, instruction-tuned then preference-optimised (ACL Findings 2025). It is the closest published recipe to this project.
- **Inside:** 3,042 training conversations with profiles (DSM-5 symptoms at 4 levels, resistance, Beck distortions, severity) and 12 evaluation profiles.
- **Example:** Samantha, 25, unemployed, severe depression; "Frequent or recurrent thoughts of death...: Severe".
- **Access:** no data or weights found (**release unverified**). Fit: HIGH as a design template only.

### AnnaAgent / Anna-CPsyCounD
- **Link:** https://huggingface.co/datasets/sci-m-wang/Anna-CPsyCounD (repo: https://github.com/sci-m-wang/AnnaAgent)
- **What it is:** A seeker simulator with an emotion modulator and short- and long-term memory across sessions (ACL Findings 2025).
- **Inside:** 3,134 synthetic Chinese conversations with fields `portrait`, `report` and `conversation`. The repo adds multi-session history and risk scores.
- **Example:** a 30-year-old homemaker with chest tightness, nightmares and poor sleep.
- **Access:** dataset Apache-2.0, code AGPL-3.0. Fit: HIGH for evaluating state, but Chinese and synthetic.

### Other symptom-text sets (lower fit)
| Name | Link | Note |
|---|---|---|
| CAMS | https://github.com/drmuskangarg/CAMS | Reddit posts labelled with causes (3,155 + 1,896 posts), not symptoms. Example: none. |
| SDCNL | https://github.com/ayaanzhaque/SDCNL | 1,895 depression vs SuicideWatch posts, noisy labels. |
| DEPTWEET | https://github.com/mohsinulkabir14/DEPTWEET | About 40k tweet IDs with depression severity; you must rehydrate. |
| RSDD / SMHD | https://ir.cs.georgetown.edu/resources/rsdd.html | Reddit users with self-reported diagnosis; user-level labels only; terms not retrieved. |
| Reddit C-SSRS | https://zenodo.org/records/2667859 | 500 users labelled by psychiatrists on suicide severity; CC BY 4.0. |
| UMD Suicidality (CLPsych 2019) | https://psresnik.github.io/umd_reddit_suicidality_dataset.html | 11k SuicideWatch users + controls; restricted, IRB needed. |
| Dreaddit | https://arxiv.org/abs/1911.00133 | About 3.5k stress-labelled segments; weak fit (stress, not depression). |
| GoEmotions | https://huggingface.co/datasets/google-research-datasets/go_emotions | 58k Reddit comments, 27 emotions; generic sadness/fear only. |
| GPT-4o-GAD-7 | https://huggingface.co/datasets/darssanle/GPT-4o-GAD-7 | Only 17 rows of GAD-7 item spans; use as a template for annotating PRIMATE. |
| PsySym / SAD / TBe-PBu / UrduHope / MODMA | see paper pages | **Unverified** data links; low-medium fit. |

---

# Part B. For the gate (G and effect)

### AnnoMI
- **Link:** https://github.com/uccollab/AnnoMI
- **What it is:** Expert-annotated motivational-interviewing demonstration transcripts, both high and low quality. It is the only public set with a session-level quality split.
- **Inside:** 133 transcripts with `mi_quality` (high/low), `main_therapist_behaviour` and `client_talk_type` (change/neutral/sustain). The topics are behaviour change (alcohol, smoking), not depression.
- **Example:** transcript 0, `mi_quality=high`, "Brief intervention: 'Barbara'" (text not retrieved).
- **Access:** public, licence not stated. Fit: HIGH (signed effect), but MI only.

### MI-TAGS
- **Link:** https://aclanthology.org/2024.lrec-main.1017
- **What it is:** MI demonstration transcripts coded with MITI 4.2 behaviour codes and global scores, with client speech tagged under CLEAR 1.0.
- **Inside:** 242 English transcripts from public videos.
- **Example:** none.
- **Access:** not found (**unverified**; ask the authors). Fit: HIGH, since global scores are close to a signed effect.

### OnCoCo 1.0
- **Link:** https://huggingface.co/datasets/th-nuernberg/OnCoCoV1
- **What it is:** Fine-grained message classification for online counselling, labelled by counsellors and trained students.
- **Inside:** 2,778 German messages machine-translated to English (5,556 samples), with 38 counsellor and 28 client categories.
- **Example:** "Could you describe in more detail what is troubling you at the moment?" -> Impact factors / Request for concerns (CO-IF-AC-RF-RC).
- **Access:** CC BY 4.0 (the tag says BY-SA; check). Fit: MED-HIGH for "was it addressed", with no effect signal.

### CounselBench
- **Link:** https://huggingface.co/datasets/izi-ano/CounselBench-Eval
- **What it is:** Expert ratings of LLM and human-therapist answers to real CounselChat questions, by 100 mental health professionals.
- **Inside:** 2,000 evaluations with five annotators per pair, span-level annotations and rationales, plus an adversarial set of 120 questions. The answers are single-turn.
- **Example:** none retrieved.
- **Access:** CC BY-NC-ND 4.0 (no derivatives). Fit: MED.

### EPITOME / Empathy-Mental-Health
- **Link:** https://github.com/behavioral-data/Empathy-Mental-Health
- **What it is:** Empathy annotations of peer-support responses (Sharma et al., EMNLP 2020).
- **Inside:** About 10k seeker/response pairs scored 0-2 on emotional reaction, interpretation and exploration, with rationale spans. The Reddit part is public; the TalkLife part is by request.
- **Example:** none retrieved.
- **Access:** Reddit CSVs open. Fit: MED.

### CBT-Bench
- **Link:** https://huggingface.co/datasets/Psychotherapy-LLM/CBT-Bench
- **What it is:** A three-level benchmark of CBT knowledge, cognitive-model classification and response generation (NAACL 2025).
- **Inside:** 1,705 rows including 146 distortion items and core-belief labels (helpless, unlovable, worthless) that fit hopelessness. It also has 468 pairwise response comparisons.
- **Example:** "I feel guilty and ashamed for lying to my boyfriend and betraying his trust." -> "I am unlovable", "I am worthless".
- **Access:** CC BY-NC 4.0. Fit: MED-HIGH for the technique-to-trait mapping.

### TherapyGym / TherapyJudgeBench
- **Link:** https://arxiv.org/abs/2603.18008
- **What it is:** A benchmark for auditing LLM judges against licensed clinicians on simulated therapy sessions.
- **Inside:** 116 dialogues with 1,270 expert ratings, covering CTRS fidelity and a multi-label safety scheme.
- **Example:** none.
- **Access:** released per the abstract, but no link found (**unverified**). Fit: MED-HIGH for the clinician-vs-LLM-judge test.

### LongCounsel-8
- **Link:** https://huggingface.co/datasets/hiddensev/LongCounsel-8
- **What it is:** Synthetic five-session counselling trajectories with a latent depression state. It is built for tracking depression across sessions.
- **Inside:** 7,749 trajectories and 1.5M utterances, with PHQ-8 item vectors, totals and severity band per session.
- **Example:** rows have `session_id`, `turn_id`, `speaker`, `text` (no sample text shown).
- **Access:** CC BY-NC 4.0. Fit: MED for sanity checks; the PHQ comes from a simulator, so it can teach simulator artefacts.

### KokoroChat
- **Link:** https://huggingface.co/datasets/UEC-InabaLab/KokoroChat
- **What it is:** Japanese counselling text chats where trained counsellors role-play both sides, with client reviews after each session.
- **Inside:** 6,589 dialogues (about 600k utterances), each with an overall score and 20 items such as `felt_heard_and_understood`.
- **Example:** preview scores: overall 72, `felt_respected` 3 (text is Japanese).
- **Access:** CC BY-NC-ND 4.0. Fit: MED (per-session client rating at scale, but role-played).

### Real-outcome sets (best, but closed)
| Name | Link | What | Access |
|---|---|---|---|
| Ieso iCBT | https://pmc.ncbi.nlm.nih.gov/articles/PMC6707006 | 90,934 session transcripts from 17,572 patients with PHQ-9/GAD-7 before every session. The ideal data. | Not public; ask Ieso. |
| CounselingWAI | https://arxiv.org/html/2602.20648 | 793 Chinese counselling sessions with per-session client and counsellor alliance (WAI-SR). | Request-only, PI + IRB + NDA. |
| Stockholm iCBT | https://pmc.ncbi.nlm.nih.gov/articles/PMC12547759/ | 182,698 messages from 6,613 patients with MADRS-S outcomes. | Closed. Warns that text adds little beyond early scores. |
| Client-Reactions | https://github.com/dll-wu/Client-Reactions | Counselling text with strategy and client-reaction labels linked to outcomes. | Authors can't release; ask by email. |
| Alexander Street | https://proquest.libguides.com/ctrn | About 4,000 real or demo therapy transcripts, no outcome scores. | Institutional licence; scraping likely a breach. |

### Other CBT / counselling sets (lower fit)
| Name | Link | Note |
|---|---|---|
| DiaCBT | https://arxiv.org/abs/2509.02999 | 108 cases, 540 multi-session CBT dialogues with 14 strategy labels. Release **unverified**. |
| Mental-health-CBT-dialogues | https://huggingface.co/datasets/yuana1234567/Mental-health-CBT-dialogues | 9,000 synthetic stage-aware pairs with a state card. CC BY 4.0. |
| RealCBT | https://gitlab.com/xiaoyi.wang/realcbt-dataset | 76 real CBT sessions from video; contents **unverified**. |
| Cognitive-Reframing | https://github.com/behavioral-data/Cognitive-Reframing | 600 situation/thought/reframe records. |
| Shreevastava distortions | https://huggingface.co/datasets/masked-kunsiquat/shreevastava-cognitive-distortions | About 3,176 rows with 12 distortion labels. MIT. |
| HOPE / MEMO | https://arxiv.org/pdf/2111.06647 | 12.9k utterances from 212 sessions with 12 dialogue acts. Download **unverified**. |
| Pérez-Rosas MI quality | https://aclanthology.org/L18-1591/ | About 259 sessions (155 high / 104 low). Likely no longer public. |
| ESConv (see Part A) | | Best open real pre/post distress change. |
| CounselChat, Psych8k, MentalChat16K, PsyDial, SMILE | huggingface / github | No quality or outcome labels; useful only as text sources. |

---

# Recommended minimum

- **LoRA data:** ReDSM5 + DepreSym (anhedonia, sleep, fatigue), DepressionEmo (hopelessness), PRIMATE (PHQ-9 items), CACTUS (therapist to client pairs). Anxiety has no good open set, so annotate PRIMATE against GAD-7.
- **Gate data:** AnnoMI + ESConv + CBT-Bench + CACTUS, with a small clinician-labelled test set. Ask Ieso and CounselingWAI for real outcomes.
