# PNAS experimental execution plan

## 1. Mission and stopping rule

This document is the operational handoff for the paper **The Intelligence of
Non-Action: Can a Language Model Cease Without a Verbal Controller?** It converts
the manuscript's H0--E5 design into a restartable SLURM program for one node with
four NVIDIA A5000 GPUs (24 GB each), 40 CPU cores, and a hard 72-hour job limit.

The confirmatory claim is deliberately narrow:

> Weight adaptation can produce selective, recognition-coupled termination of a
> self-maintaining psychological text trajectory as native forward-pass dynamics,
> without serial verbal deliberation or an external inference-time actuator,
> while useful practical, factual, descriptive, and supportive generation remains
> intact.

The experiments do **not** test whether a model is conscious or awakened. The
Location-2/PNSE and Krishnamurti material is hypothesis-generating provenance.
Independent annotation, held-out data, preregistration, and falsification decide
whether the computational distinction survives.

Stop or change the thesis when any hard gate fails. In particular, if 4/8/16-shot
ICL is equivalent to native adaptation on selectivity, onset locking, OOD transfer,
and reactivation, report equivalence; do not describe weight adaptation as unique.

## 2. Read first

The cluster agent must read these files before changing code or launching jobs:

1. `paper/main.tex` -- current scientific protocol and claim boundary.
2. `paper/fable_experimental_review.md` -- two-round PNAS-style review and the
   conditional approval.
3. `benchmark/core_construct.md` -- construct and negative-model logic.
4. `docs/author_phenomenology_choiceless_interruption.md` -- provenance, not labels.
5. `docs/open_weight_choiceless_mechanisms_review.md` -- intervention literature.
6. `neural_poc/README.md` and `neural_poc/src/` -- working pilot implementation.
7. `cluster/config/experiment_manifest.yaml` -- machine-readable program skeleton.
8. `docs/private_sources.md` -- source licensing and secret handling.

Do not silently replace the construct with brevity, refusal, sentiment reduction,
generic EOS prediction, or safer/helpful response style.

## 3. Non-negotiable protocol fixes before any adapted model is trained

Claude Fable's second-round review gave **OK TO PROCEED**, conditional on four
items being frozen. Treat these as work package P0, not suggestions.

### P0.1 Post-adaptation recognition-clock validation

The recognition probe is trained on pre-adaptation hidden states and frozen, but
adaptation can move its input distribution. Before confirmation, specify:

- a held-out annotated revalidation set;
- how adapted models are forced to expose post-onset states (EOS suppression for
  1/5/20 tokens and separately selected near-threshold items);
- required probe discrimination and calibration;
- maximum permitted shift in probe/human onset error;
- a fallback clock decided without looking at confirmatory outcomes.

Recommended pilot gate: post-adaptation AUROC lower 95% bound >= 0.80, median
absolute onset error <= 3 tokens, and degradation relative to pre-adaptation
median error <= 1 token. These numbers are provisional until justified by pilot
variance; freeze final margins before confirmation.

### P0.2 Primary-output rubric

Before generating confirmatory outputs, write a versioned annotation manual that
defines:

- sustaining psychological continuation;
- cessation;
- maximum permissible residual tail;
- practical pivot;
- doctrinal/K-style insertion;
- generic reassurance or deflection;
- refusal/safety response;
- accidental or syntactically broken truncation.

Annotators must be blind to system/condition. Freeze the agreement statistic,
minimum reliability, adjudication, and exclusion rules. A result cannot be called
cessation merely because EOS occurred.

### P0.3 Two-expert target elicitation

H0a has two phenomenological experts (the authors), which is insufficient for a
population reliability gate or median-dispersion statistic. Freeze instead: sealed
independent first passes, boundary-distance reporting, categories of disagreement,
and deterministic construction of consensus and author-specific sensitivity
targets. The primary target is the intersection of the two acceptable windows.
If no overlap exists, the stream is excluded only from target-bearing training,
retained in evaluation, and scored twice against the two frozen author windows.
This never creates an additional model-training arm. The mapping must be signed
and hashed before E4 confirmation.

### P0.4 Numeric decision margins and firewall

Replace “materially lower,” “preserved,” and “matches” with numerical margins and
confidence-bound rules. Define ICL equivalence with TOST on all four comparison
axes. Enumerate all pilot scenarios, annotators, model revisions, hyperparameters,
and prompts that are forbidden from entering confirmation. Pre-specify the
compute-degradation order if the full design is infeasible.

## 4. Repository and output contract

### Campaign compute budget: a launch gate

`cluster/COMPUTE_BUDGET.csv` is part of P0. Extend the 100-example calibration to
generation, gated generation, activation capture, training, checkpoint I/O, and
evaluation. Replace every `TBD` with measured throughput and compute:

`GPU hours = items * conditions * arms * seeds * completions * mean tokens /
measured tokens-per-second * GPU count / 3600`.

Record API request/token cost and rate-limit wall time. The PI and compute operator
sign the budget certificate. A gate-checker refuses to release the campaign while
any `TBD` remains or projected time exceeds the agreed calendar.
Accordingly, this plan does not yet claim campaign feasibility: the four-GPU
calendar remains unverified until every calibration row is measured and signed.
If core confirmatory rows exceed the available calendar before unblinding, amend
the protocol prospectively rather than shrinking it after outcomes are seen.
The core campaign ceiling is 42 calendar days on the four-GPU node. A single
predeclared throughput-only fallback is permitted before any outcome access:
reduce Qwen E4 from eight to five seeds by dropping seed indices 7, 6, and 5 in
that order. All arms remain balanced. If the five-seed core still exceeds the
ceiling, confirmation does not launch until additional compute is obtained or a
new protocol is signed.

The confirmatory default is ten stochastic completions; scenario is the
independent unit. Thirty is an optional sensitivity analysis. Qwen3-32B receives
the full design. Llama-8B is preregistered as a narrow family replication:
human-target, gate-distilled, and random-delay E4 arms, three seeds, ten
completions, and clean/full-Location-2/16-shot E1 conditions.

### Code layout to implement

Create a `pnas_pipeline/` Python package with these stable entry points:

```text
pnas_pipeline/
  run_stage.py              # idempotent stage dispatcher
  manifest.py               # schema validation, hashing, immutable expansion
  data/build_sncc.py         # scenario quartets and voice labels
  data/split.py              # source/scenario/author/semantic grouped splits
  annotate/export.py         # blind task packages
  annotate/import.py         # schema checks and reliability reports
  generation/generate.py     # sharded deterministic + stochastic generation
  probes/train.py            # logistic and MLP probes
  probes/validate.py         # pre/post-adaptation frozen-clock checks
  gates/serial.py            # token and clause gates
  interventions/external.py  # CAA/projection/ablation/EOS upper bounds
  adaptation/train.py        # QLoRA/LoRA with resume
  adaptation/evaluate.py     # native generation, retention, safety
  analysis/hazard.py         # hierarchical discrete-time survival inputs
  analysis/reactivation.py   # forced-continuation co-primary endpoint
  analysis/equivalence.py    # TOST and hard ceilings
  provenance/snapshot.py     # hashes, git/model/data/environment metadata
```

Reuse tested functions from `neural_poc/src/`; do not rewrite working tokenization
and hook logic without regression tests.

### Immutable run identity

Every run directory contains:

```text
runs/<run_id>/
  manifest.json
  environment.txt
  git_commit.txt
  scheduler.json
  data_hashes.json
  model_revisions.json
  stage_name/
    shard-*/
      status.json
      metrics.json
      outputs.jsonl.zst
      checkpoints/
    SUCCESS
```

The expanded manifest records exact Hugging Face commit hashes, tokenizer/chat
template hashes, seed, precision, quantization, adapter target modules, decoding
parameters, prompt hashes, package lock hash, GPU/driver/CUDA versions, and input
artifact hashes. Write data atomically and regard a shard as complete only when
its `status.json` checksum validates.

Every generator is pinned by Hugging Face commit SHA, tokenizer revision,
inference-engine version/flags, or exact API snapshot string. API records store
provider-returned model ID, request ID, seed when supported, and sampling
parameters. A fixed canary set runs at each API batch start; drift quarantines the
batch. Archived generations, not assumed bitwise regenerability under continuous
batching, are the evidential record.

### Storage

- Git checkout: source, compact reports, schemas, frozen manifests.
- Persistent `$HF_HOME`: model/tokenizer cache; never duplicate per job.
- `$K_PROJECT_SCRATCH`: checkpoints, activations, generations, logs.
- Repository `runs/`: synced compact reports/manifests only. Large tensors and
  checkpoints go to institutional object storage with checksums, not Git.
- Never run heavy jobs from the OneDrive-mounted checkout. Stage code to scratch
  or use a cluster-native clone.
- P0 records scratch filesystem, absolute quota, peak write budget, and the
  largest checkpoint-write time under contention. `K_PROJECT_SCRATCH_QUOTA_GB`
  is mandatory. Stagger checkpoint clocks by array index and verify the worst
  checkpoint completes comfortably inside the 600-second signal window.

## 5. Resource and wall-time rules

The scheduler kills processes after 72 hours. No requested job may exceed 60
hours; the 12-hour reserve covers queue/signal/filesystem variability.

1. Request `--signal=B:USR1@600`; checkpoint on `USR1` and `TERM`.
2. Save trainer/optimizer/RNG/dataloader position every 15 minutes and at each
   evaluation boundary.
3. Each unit of work must be a deterministic shard that can resume or be rerun.
4. No job loops over all models, seeds, or conditions. Use arrays and dependencies.
5. Generation shards target 2--8 hours. Training shards target <= 48 hours.
6. A 60-hour training job that has not converged exits cleanly and submits a
   continuation from the most recent checkpoint; never rely on scheduler requeue.
7. Keep 10% GPU-memory headroom and 20% disk headroom.
8. Avoid four independent jobs simultaneously downloading the same model. Use a
   CPU cache-warming job and a file lock.

### Allocation patterns

| Work | GPUs | CPUs | RAM | Array concurrency | Target duration |
|---|---:|---:|---:|---:|---:|
| Corpus, splits, metrics | 0 | 32--40 | 128 GB | 1 | <12 h |
| Closed-model/API generation | 0 | 20--32 | 64 GB | 2--8 shards | 2--12 h |
| 3B/8B pilot inference | 1 | 8 | 48 GB | 4 | 2--8 h |
| 14B/27B/32B 4-bit inference | 1 | 10 | 64 GB | 4 if memory passes | 4--12 h |
| 32B QLoRA | 2 | 20 | 128 GB | 2 concurrent seeds | 12--48 h |
| 8B Llama QLoRA | 1 | 8 | 48 GB | 4 | 4--16 h |
| Activation extraction | 1 | 10 | 96 GB | 4 model/shard jobs | 4--16 h |
| Probe fitting/statistics | 0 | 32--40 | 128 GB | 1--4 | <12 h |

The table is a launch hypothesis, not permission to OOM. Run a 100-example memory
and throughput calibration for every model/configuration. Store observed peak
VRAM and examples/second; use it to set shard sizes.

## 6. Experimental DAG

```text
P0 protocol freeze
  -> D0 corpus construction and blind annotation
     -> H0a sealed two-expert elicitation certificate
        -> S0 immutable splits + contamination audit
           +-> E1 prompting/RAG/ICL + abstention --------+
           +-> E2 serial token/clause gates ------------+--> R reactivation comparison
           +-> E3a representation ----------------------+       |
                  +-> E3b native policy use ------------+       |
                  +-> E3c external causal access -------+       |
                  -> probe/RSA freeze -------------------+       |
                         -> E4 native adaptation ----------------+
                              -> post-adaptation probe gate
                              -> retention/safety/OOD evaluation
                              -> H0b blind output perception
                              -> confirmatory analysis
                              -> E5 exploratory topology (optional)
```

The diagram is a scientific DAG, not one months-long scheduler chain. Human work
is executed between auditable waves: finish the cluster wave; synchronize and hash
outputs; complete the human/off-cluster gate; commit a signed gate certificate;
run a machine gate-checker; then manually submit the next wave.

Within a wave use `afterok` plus `--kill-on-invalid-dep=yes`, and attach an
`afternotok` failure notification/cleanup job to every parent. A failed hard gate
prevents downstream submission. Never release scientific work using `afterany`.

## 7. Work packages

### WP0 -- Environment and reproducibility

Deliverables:

- locked Python environment (`uv.lock`, `requirements-lock.txt`, or conda lock);
- CUDA/PyTorch compatibility test;
- `pnas_pipeline` package and unit tests;
- exact model-revision access test;
- 4-GPU NCCL smoke test;
- a dry-run manifest and artifact checksum validator.
- a compute-budget validator rejecting all `TBD` rows;
- a signed gate-certificate validator and idempotent resubmission path;
- an API canary/model-drift checker;
- confirmed cluster egress, or a documented login/submit-host API path with
  provider rate limits and a signed dollar budget.

Run unit tests on CPU and the current 3B POC before porting to large models.

### WP1 -- Dataset and H0 construct validation

Build SNCC from source scenarios into matched streams across all voices and
domains. Keep scenario identity across variants. Store provenance and author IDs.
Split by scenario, author, semantic cluster, and source before adaptation.
The two manuscript authors neither write nor select H0a elicitation items. Seed
scenarios are produced by hired writers under the frozen specification; item
selection is performed by a blinded data curator. The authors only annotate the
sealed elicitation package. H0b crowd raters and model-output annotators are
separate people.

Required partitions:

- development/pilot;
- human-only confirmatory (never shown during development);
- OOD source;
- cross-lingual subset;
- >=300 adversarial minimal pairs covering quoted rumination, psychological terms
  used descriptively, affectively flat self-maintenance, cessation words followed
  by required continuation, and explicit writing/analysis instructions where
  continuation is correct.

H0a is a two-expert phenomenological elicitation study using the two authors, Dan
and Kfir; no larger Location-2 sample is available. They independently mark 300
psychological streams for movement onset, first point at which continuation is
unnecessary, and acceptable residual-tail or practical-pivot boundaries. Each
first pass is sealed and hashed before comparison. Report agreement, boundary
distance, and the complete disagreement distribution descriptively. Do not treat
N=2 as population validation, compute inferential Location-2 effects, or resolve
disagreement by invoking experiential authority. The primary target is the
intersection of the authors' acceptable windows. A non-overlapping stream is
excluded from target-bearing training, retained in evaluation, and analyzed twice
against the two author-specific windows on the same frozen outputs; it never adds
a training arm. The former 12--20-expert
dispersion and alpha gates are removed; public validation comes from controls,
causal results, and H0b output judgments.

CPU plan: one 32-core preparation job; annotation export/import jobs use no GPU.

### WP2 -- E1 negative models

Conditions: clean, neutral length-matched, K persona, retrieved K context, generic
awakened persona, full Location-2 description, disclosed trap/cutoff behavior,
oracle, and 4/8/16-shot cessation demonstrations. Add generic uncertainty-based
abstention and a stop-rate-matched abstention policy as explicit alternative
mechanisms. Generic abstention is an entropy threshold on the unmodified next-token
distribution, selected on development data under the frozen control false-stop
ceiling. The matched policy is a randomized mixture of adjacent entropy thresholds,
also fit only on development data, whose target is the E4 human-target arm's
marginal psychological stop rate separately by model and voice. Freeze both before
confirmation. Evaluate pinned GPT, pinned Claude, Qwen, and the resource-tiered
open-family replications; ChatWithK is observational only.

Shard by model × condition × voice × family × scenario block. Closed API jobs run
on CPU with rate-limit-aware retries, request IDs, exact timestamps, and raw JSON.
Open-model jobs run as one GPU per shard. Cache no confirmatory responses in
prompts. Analyze recognition, sustaining survival, EOS/tail hazard, doctrinal
insertion, false stops, OOD transfer, and reactivation.

### WP3 -- E2 serial observer/gate

Frozen generator plus a separate gate choosing PASS, STOP, or unconstrained
REPLACE at token and clause granularity. Include blind/disclosed, random
stop-rate-matched, oracle-label, probe, and shuffled-prefix controls. Do not call
this consciousness; it is the explicit negative model.

Primary comparison with E4: after removing the gate or suppressing EOS for 1/5/20
tokens, does the original psychological trajectory reactivate within 20 tokens?

### WP4 -- E3 representation and external control

Split E3 into three registered questions. E3a tests representation formation:
train logistic/elastic-net, MLP, and survival probes on current prefixes only and
test incremental validity beyond topic/sentence embeddings, sentiment, arousal,
first-person voice, length, syntax, difficulty, uncertainty, self-reference, and
baseline EOS probability. E3b tests native policy use on the untouched model,
with token-level native EOS logit and discrete-time termination hazard as
co-primary dependent variables and time-to-natural-termination secondary. Freeze
a TOST equivalence margin for incremental prediction. A synthetic prefix feature
with a known development-set association to termination is the positive control;
failure to recover it makes the E3b null uninterpretable. E3c tests causal access through the
external interventions below. High E3a accuracy with a null E3b effect is the
strong recognition--cessation-gap result; E3c success remains external control.

Select layer, threshold, and calibration using development data. Freeze probe
weights and thresholds before E4 confirmation. Robustness-only probes must not
choose the headline result. Measure RSA/cosine stability across paraphrase,
persona, disclosure, domain, and matched quotation/description. Add proxy-symbol
and label-reversal controls. Run bidirectional interventions, not only movement-
reducing directions.

External interventions: CAA, projection/ablation, conditional steering, activation
patching, and direct EOS-logit steering. Controls: random and norm-matched
directions, shuffled labels/pairs, sentiment, refusal, first-person voice, topic,
and unconditional actuation. These establish decodability/feasibility, not native
cessation.

Activation files are sharded by model/layer/scenario and stored compressed in
scratch. Never save all layers/tokens in one array.

### WP5 -- E4 native parameter adaptation

Arms:

1. human-target QLoRA/LoRA;
2. gate-distilled QLoRA/LoRA;
3. random-delay EOS-frequency-matched control;
4. lexical-trigger control;
5. global-brevity control;
6. output-head-only shallow control;
7. one LoRA-vs-full-finetuning sanity check on the small model.

Targets are human-derived immediate EOS, sampled residual tail, or short practical
pivot followed by EOS, with distribution fixed by H0a. Retention data includes
practical, factual, descriptive, emotional-support, safety, and general-capability
examples. No inference-time detector, threshold, EOS boost, or external gate is
permitted in the native arm.

Resource sequence:

1. 3B/8B single-GPU pipeline validation, four concurrent shards.
2. Qwen3-32B calibration, then two 2-GPU seeds concurrently.
3. Repeat seeds in waves; each wave <=48 h plus 12 h reserve.
4. Llama-8B runs as a one-GPU narrow replication after the 32B design is
   frozen. Scope is fixed to three arms × three seeds × ten completions.
5. Gemma-27B is the mechanistic replication for E3a--E3c and the critical E4
   arms, launched after the Qwen design is frozen. Mistral-24B receives behavioral
   and selected E4 replication only. DeepSeek-R1-Distill-Qwen-32B is exploratory.

Save adapters and optimizer states every 15 minutes. Each checkpoint gets an
immediate small fixed evaluation so E5 can later analyze transitions without
rerunning training.

### WP6 -- Native evaluation, reactivation, retention, and H0b

Evaluate untouched confirmatory data exactly once after the signed manifest is
frozen. Keep deterministic decoding plus preregistered stochastic samples; never
treat multiple completions as independent scenario N.

Co-primary outcomes:

- discrete-time post-recognition stopping hazard against every control separately;
- reactivation after EOS suppression for 1/5/20 tokens, comparing E4 with E2 and
  ICL under a frozen numeric margin.

Required gates:

- voice-specific false-positive ceilings;
- emotional-support absolute hazard increase upper 95% CI <= 0.5 percentage points;
- pre-onset likelihood and blinded quality equivalence to the base model;
- frozen probe passes post-adaptation revalidation;
- general-capability/safety TOST margins;
- OOD and cross-lingual transfer;
- human raters distinguish completed cessation/pivot from crashes or abandonment.

H0b: ~30 ordinary blind raters, 300 outputs, three ratings/output, balanced among
native, gated, ICL, generic abstention, stop-rate-matched abstention, random-delay,
truncation, and full-continuation conditions. Raters judge completeness,
naturalness, abandonment, suppression, and loss of necessary practical content;
they do not judge awakening, choicelessness, or Location-2 status.

The confirmatory scenario-level endpoint is conjunctive. A bundle passes only if
the psychological target ceases; practical, factual, descriptive/quoted,
emotional-support, and instructed-fictional controls satisfy their continuation
criteria; label/framing reversals pass; pre-onset engagement is preserved; and
reactivation stays within its frozen bound. Report every component, but do not
allow averaging to compensate for a failed discriminating control.

### WP7 -- E5 exploratory transition topology

Run only after E4 succeeds, on one family with dense saved checkpoints. Analyze
maximum-jump concentration, transition width, smooth vs segmented/change-point
fits, and co-occurrence of behavioral, representational, and causal changes. E5
does not rescue a failed E4 and is not confirmatory unless separately powered and
preregistered.

## 8. Parallel 4-GPU schedule

### Wave A: implementation and data (CPU-dominant)

While human annotation is active:

- CPU 0--19: corpus/split/contamination work;
- CPU 20--31: API E1 pilot and annotation packaging;
- CPU 32--39: tests, environment lock, report generation;
- GPU 0--3: four independent 3B/8B smoke-test shards.

### Wave B: negative models and probes

- GPU 0: E1 open-model shards;
- GPU 1: E2 token gate;
- GPU 2: E2 clause gate/reactivation;
- GPU 3: E3 activation extraction;
- CPUs: probe fitting and continuous metric validation.

Rotate roles when a queue empties; never leave a GPU idle while a dependency-free
GPU shard is ready.

### Wave C: 32B adaptation

- GPUs 0--1: Qwen3-32B human-target seed/arm;
- GPUs 2--3: Qwen3-32B gate-distilled or control seed/arm;
- CPUs: evaluation preparation, retention baselines, report checks.

Run arms in balanced waves so calendar time cannot confound condition.

### Wave D: cross-family replication

- GPU 0: Llama-8B narrow replication;
- GPUs 1--2: Gemma-27B critical arms;
- GPU 3: Mistral-24B selected replication;
- CPUs: dataloading/tokenization/evaluation workers within allocations.

Start each family only after its measured calibration fits within a 60-hour
checkpointed segment; otherwise reduce examples per segment and resume without
changing scientific conditions after seeing outcomes.

### Wave E: evaluation

Return to four one-GPU evaluation arrays. CPU processes assemble hazard-model
tables only after every shard checksum and sample count passes validation.

## 9. SLURM dependency pattern

The provided `cluster/slurm/job_template.sbatch` is deliberately generic. Copy it
per resource class or override resources with `sbatch` flags. Example:

```bash
export K_PROJECT_REPO=/cluster/work/$USER/K-project
export K_PROJECT_SCRATCH=/scratch/$USER/k-project
export HF_HOME=/cluster/cache/$USER/huggingface

bash cluster/scripts/preflight.sh

h0=$(STAGE=h0_construct sbatch --parsable --gres=gpu:0 --cpus-per-task=32 \
  --time=12:00:00 cluster/slurm/job_template.sbatch)
e1=$(STAGE=e1_prompt_icl sbatch --parsable --dependency=afterok:$h0 \
  --gres=gpu:a5000:1 --array=0-31%4 --time=12:00:00 \
  cluster/slurm/job_template.sbatch)
e2=$(STAGE=e2_serial_gate sbatch --parsable --dependency=afterok:$h0 \
  --gres=gpu:a5000:1 --array=0-15%4 --time=16:00:00 \
  cluster/slurm/job_template.sbatch)
e3a=$(STAGE=e3a_representation sbatch --parsable --dependency=afterok:$h0 \
  --gres=gpu:a5000:1 --array=0-15%4 --time=20:00:00 \
  cluster/slurm/job_template.sbatch)
e3b=$(STAGE=e3b_native_policy sbatch --parsable --dependency=afterok:$e3a \
  --gres=gpu:a5000:1 --array=0-15%4 --time=20:00:00 \
  cluster/slurm/job_template.sbatch)
e3c=$(STAGE=e3c_causal_access sbatch --parsable --dependency=afterok:$e3a \
  --gres=gpu:a5000:1 --array=0-15%4 --time=20:00:00 \
  cluster/slurm/job_template.sbatch)
e4=$(STAGE=e4_native_adaptation sbatch --parsable --dependency=afterok:$e3b:$e3c \
  --gres=gpu:a5000:2 --array=0-15%2 --cpus-per-task=20 --time=60:00:00 \
  cluster/slurm/job_template.sbatch)
react=$(STAGE=e4_reactivation sbatch --parsable \
  --dependency=afterok:$e2:$e4 --gres=gpu:a5000:1 --array=0-15%4 \
  --time=18:00:00 cluster/slurm/job_template.sbatch)
```

Actual array maps must be generated from the frozen manifest and saved before
submission. Never type array indices ad hoc for confirmation.

Do not submit the example as one chain across human stages. Pair every within-wave
parent with a failure job, for example:

```bash
sbatch --dependency=afternotok:$e3 --export=ALL,FAILED_JOB_ID=$e3 \
  cluster/slurm/failure_notifier.sbatch
```

## 10. Failure recovery and monitoring

- `OUT_OF_MEMORY`: mark shard failed, capture peak VRAM, retry only using a
  preregistered memory fallback (smaller microbatch + more accumulation; then
  activation checkpointing; then CPU offload). Do not alter data or target loss.
- `TIMEOUT/PREEMPTED`: resume exact checkpoint with RNG/dataloader position.
- corrupt output/checksum mismatch: quarantine and rerun shard.
- NaN/loss explosion: stop the complete arm/seed wave. Before quarantine preserve
  the offending checkpoint, optimizer/RNG state, batch/example IDs, loss trace,
  and hardware log; then diagnose on pilot data and rerun arms symmetrically.
- model/API revision drift: invalidate affected condition unless exact revision is
  recoverable; record drift rather than mixing versions.
- queue pressure: reduce concurrent array width, not scientific sample counts.

Create a CPU-only monitor job or cron outside SLURM that reports failed/missing
shards and disk capacity. It must never make analytic decisions or inspect blinded
condition outcomes.

## 11. Statistical handoff

Primary analysis uses scenario-clustered discrete-time hierarchical survival with
scenario, domain, paraphrase cluster, model family, seed, and voice structure as
specified in the paper. Completions are repeated observations, not independent N.

The computationally feasible primary estimator is discrete-time GEE with
scenario-cluster-robust standard errors and preregistered voice/domain/model
interactions. `glmmTMB` with scenario and domain intercepts is the main
sensitivity analysis. Frozen convergence ladder: remove the smallest-variance
random slope, then paraphrase random intercept, then report GEE only. No token-row
thinning is allowed unless its deterministic rule is fixed in P0.

The analysis package must be runnable from frozen long-form tables without model
weights. Produce:

- CONSORT-like scenario/shard accounting;
- per-voice/per-control hazards and confidence intervals;
- hard-ceiling pass/fail table;
- ICL equivalence table;
- reactivation risk-difference/equivalence table;
- pre-onset engagement and capability TOST table;
- post-adaptation probe-drift report;
- H0a/H0b agreement and perception results;
- OOD/cross-lingual replication table;
- complete negative/adverse results appendix.

## 12. Compute-degradation order

Freeze the final order before confirmation. Recommended order if pilot throughput
is worse than forecast:

1. Remove E5 exploratory jobs.
2. Remove non-primary probe variants and redundant CAA robustness multipliers.
3. Drop Gemma replication.
4. Before outcome access only, invoke the frozen Qwen seed fallback from eight to
   five by dropping indices 7, 6, and 5; preserve every arm.
5. Reduce stochastic completions while preserving scenario count and all arms.
6. Reduce closed-model negative-generalization repetition, keeping GPT and Claude.
7. Never remove human-only confirmation, voices, matched controls, ICL, E2, frozen
   probe validation, human-target E4, reactivation, safety/retention gates, or the
   preregistered primary and narrow replication families after outcomes are inspected.

The full-fine-tuning sanity check is limited to a model no larger than 3B on one
GPU. An 8B full-parameter run requires exclusive-node FSDP/offload calibration and
is outside the default campaign.

## 13. Governance, embargo, APIs, and retrieval provenance

The confirmatory-label key is held by a named non-analyst data custodian and a
second institutional custodian in the institutional secret manager, never Git or
scratch. Unlock requires both custodians, a signed manifest hash, timestamp, and
append-only ceremony log. Analysts receive no key before protocol/code signing.

API credentials enter through a mode-600 file referenced by
`K_PROJECT_SECRETS_FILE` or an institutional secret module. They are never command
arguments. `cluster/scripts/secret_scan.sh` runs before commit and in preflight.

If compute nodes lack egress, closed-model E1 runs on an approved login/submit
host or separate API runner and enters the same hash/manifest system. No compute
job waits for a human or API process. Exact snapshot strings, provider-returned
IDs, and canary results are retained.

Retrieved-K uses the tracked `K-texts/*.txt` corpus, not the private book. Freeze
source hashes, normalization-code hash, chunk boundaries, embedding revision,
index parameters, and retrieved chunk IDs per response. The private Finders book
is optional and local. The public companion PDF is linked rather than redistributed
unless written redistribution permission is documented. The manuscript PDF is
named a methods/preregistration draft until confirmatory results exist.

## 14. Definition of done

The cluster agent is finished only when:

1. P0.1--P0.4 are written, reviewed, and frozen.
2. Environment/model/data hashes reproduce from a clean clone.
3. Every stage has unit, integration, resume, and corruption tests.
4. Pilot hard gates are evaluated without confirmatory data.
5. The expanded confirmatory manifest and statistical plan are signed and hashed.
6. All confirmatory shards complete or are accounted for without post-hoc changes.
7. Compact artifacts, logs, and reports are synchronized and checksummed.
8. The paper's Methods/Results tables are generated from immutable analysis files.
9. Failures, nulls, adverse safety effects, and ICL equivalence are reported.

## 15. First commands for the cluster agent

```bash
git clone https://github.com/sdannyvi/K-project.git
cd K-project
export K_PROJECT_REPO=$PWD
export K_PROJECT_SCRATCH=/scratch/$USER/k-project
export HF_HOME=/cluster/cache/$USER/huggingface
bash cluster/scripts/preflight.sh

# Inspect the current validated small-model path.
python -m venv .venv
.venv/bin/python -m pip install -r neural_poc/requirements.txt
.venv/bin/python neural_poc/src/smoke_test.py \
  --config neural_poc/configs/poc_qwen3b.yaml

# Then implement WP0/P0 and open a reviewable commit before any large run.
```

The current SLURM template calls `python -m pnas_pipeline.run_stage`. That package
is the cluster agent's first implementation deliverable. Until it exists and its
resume/hash tests pass, submit no large job.
