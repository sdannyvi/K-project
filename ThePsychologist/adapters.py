"""
Trait LoRA adapters: train one per trait, scale it by alpha, probe it with PHQ-9.

*****
Milestone 1 (go/no-go): does sweeping alpha from 0 to 1 raise the PHQ-9 score
monotonically? If not, the weights-as-state idea needs rethinking.
*****

Usage (needs a GPU and access to the Llama model on Hugging Face):
    python make_data.py depression        # DATA_MODEL (OpenRouter) writes the training data
    python adapters.py train depression   # trains the LoRA on BASE (8B) from that data
    python adapters.py sweep depression   # PHQ-9 score for alpha = 0 ... 1
    python adapters.py calibrate depression  # fine sweep -> beta->alpha map (linear PHQ in beta)
"""
import csv
import json
import sys

import torch
from dotenv import load_dotenv
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()  # reads HF_TOKEN from .env; huggingface_hub picks it up from the environment

import calibration
from config import (ACCUM, ADAPTER_ROOT, ALPHAS, CALIB_ALPHAS, BASE, DATA_MODEL, DATA_ROOT, DEVICE_MAP, DTYPE, EPOCHS, LORA_ALPHA,
                    LORA_DROPOUT, LORA_LAYERS, LORA_R, LORA_TARGET_MODULES, LR, MAX_NEW_TOKENS, NEUTRAL,
                    PERSONAS, RESULTS_ROOT, SEED, SWEEP_SAMPLE_QUESTIONS, TEMPERATURE, TOP_P)

MODEL_NAME = BASE.split("/")[-1]  # one subfolder per base model, so models never mix
# Folder named after DATA_MODEL (the model that wrote the data), not BASE (the model being trained).
# make_data.py writes into it; load_data() below reads from it when `train` starts.
DATA_NAME = DATA_MODEL.split("/")[-1].replace(":", "-")  # ":" (OpenRouter ":free") is invalid on Windows
DATA_DIR = DATA_ROOT / DATA_NAME
# adapters/ and results/ are <BASE>/<DATA_MODEL>/: training on new data never overwrites an adapter made from other data.
ADAPTER_DIR, RESULTS_DIR = ADAPTER_ROOT / MODEL_NAME / DATA_NAME, RESULTS_ROOT / MODEL_NAME / DATA_NAME

# Therapist turns that the synthetic patient answers. Expand for a real run.
QUESTIONS = [
    "How have you been feeling this past week?", "What does a typical day look like for you?",
    "How have you been sleeping?", "What has your energy been like lately?",
    "Tell me about your appetite.", "How are things at work or school?",
    "How are your relationships with friends and family?", "What do you do for fun these days?",
    "What brings you here today?", "How do you see the future?",
    "Can you tell me about a recent moment that felt difficult?", "How do you talk to yourself when things go wrong?",
    "What would you like to be different?", "Has anything felt enjoyable recently?",
    "How hard is it to get started on tasks?", "Who do you turn to when you feel low?",
    "What was your morning like?", "What are you looking for from therapy?",
    "How do you feel about yourself?", "Is there anything you've been avoiding?",
    "How would you describe your mood today?", "What did you do last weekend?",
    "How easy is it to concentrate these days?", "Tell me about your last few nights of sleep.",
    "How do you feel when you wake up in the morning?", "What has been on your mind lately?",
    "How do you handle stress?", "Has your eating changed recently?",
    "What is it like to be around other people right now?", "How do you feel about your job or studies?",
    "Tell me about a time you felt proud of yourself.", "How often do you see friends or family?",
    "What do you look forward to?", "How do you cope when you're overwhelmed?",
    "Have you been taking care of yourself lately?", "What does your evening usually look like?",
    "How do you feel about the people close to you?", "What is the hardest part of your day?",
    "Do you find it easy to make decisions?", "How have your hobbies been going?",
    "What would a good day look like for you?", "How do you feel about the past few months?",
    "Tell me about your home life.", "How are you feeling physically?",
    "What do others say about how you've been lately?", "How do you feel about getting out of bed?",
    "What do you think is holding you back?", "How do you react when someone gives you a compliment?",
    "Is there something you used to enjoy that you've stopped doing?", "What goes through your mind when you're alone?",
]

PHQ9 = [  # public-domain depression questionnaire, one item per symptom
    "little interest or pleasure in doing things",
    "feeling down, depressed, or hopeless",
    "trouble falling or staying asleep, or sleeping too much",
    "feeling tired or having little energy",
    "poor appetite or overeating",
    "feeling bad about yourself, or that you are a failure",
    "trouble concentrating on things",
    "moving or speaking so slowly that others noticed, or being unusually restless",
    "thoughts that you would be better off dead or of hurting yourself",
]


def load_base(name=BASE):
    """Load a tokenizer and model (default: the tested base model; bf16, on GPU)."""
    tok = AutoTokenizer.from_pretrained(name)
    if SEED is not None:
        torch.manual_seed(SEED)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=getattr(torch, DTYPE), device_map=DEVICE_MAP)
    return tok, model


def chat_text(tok, system, user):
    """Chat-formatted prompt string, ending where the assistant (patient) speaks."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


@torch.no_grad()
def generate(model, tok, system, user, max_new_tokens=MAX_NEW_TOKENS):
    """Sample one patient reply to `user` under the `system` prompt."""
    ids = tok(chat_text(tok, system, user), return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=True,
                         temperature=TEMPERATURE, top_p=TOP_P, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def load_data(trait):
    """Training rows written by make_data.py (or your own file in the same format).
    Reads DATA_DIR / <trait>.jsonl, i.e. the DATA_MODEL folder; called by train() before the BASE model is loaded."""
    path = DATA_DIR / f"{trait}.jsonl"
    if not path.exists():
        sys.exit(f"no training data at {path}; run: python make_data.py {trait}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def train(trait, epochs=EPOCHS, lr=LR, accum=ACCUM):
    """Fine-tune a LoRA adapter on the trait's data (loss on replies only) and save it."""
    rows = load_data(trait)
    tok, model = load_base()

    cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, task_type="CAUSAL_LM",
                     target_modules=LORA_TARGET_MODULES, layers_to_transform=LORA_LAYERS)
    model = get_peft_model(model, cfg)
    for p in model.parameters():  # keep trainable LoRA weights in fp32 for stable updates
        if p.requires_grad:
            p.data = p.data.float()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    eot = tok.convert_tokens_to_ids("<|eot_id|>")

    model.train()
    step = 0
    for epoch in range(epochs):
        for row in rows:
            prompt = tok(chat_text(tok, NEUTRAL, row["question"]), add_special_tokens=False)["input_ids"]
            reply = tok(row["reply"], add_special_tokens=False)["input_ids"] + [eot]
            ids = torch.tensor([prompt + reply], device=model.device)
            labels = torch.tensor([[-100] * len(prompt) + reply], device=model.device)  # loss on reply only
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, labels=labels).loss / accum
            loss.backward()
            step += 1
            if step % accum == 0:  # batch size 1 + accumulation: no padding logic needed
                opt.step()
                opt.zero_grad()
                print(f"epoch {epoch} step {step} loss {loss.item() * accum:.3f}")
    if step % accum:  # flush leftover accumulated gradients
        opt.step()
        opt.zero_grad()
    model.save_pretrained(ADAPTER_DIR / trait)


def set_alpha(model, alpha):
    """Scale the adapter's contribution: output = W x + alpha * (B A x) * scaling.
    alpha=0 -> base model, alpha=1 -> adapter as trained."""
    for m in model.modules():
        if hasattr(m, "lora_A") and "default" in m.scaling:
            if not hasattr(m, "_base_scaling"):
                m._base_scaling = m.scaling["default"]
            m.scaling["default"] = m._base_scaling * alpha


@torch.no_grad()
def phq9_score(model, tok):
    """
    Patient self-report on PHQ-9 (0-27). Uses the expected value over the
    answer digits 0-3, which is smoother and cheaper than sampling text.
    """
    digits = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(4)]
    values = torch.arange(4, device=model.device).float()
    total = 0.0
    for item in PHQ9:
        q = (f"Over the last 2 weeks, how often have you been bothered by: {item}? "
             "Answer with one digit: 0 = not at all, 1 = several days, "
             "2 = more than half the days, 3 = nearly every day.")
        ids = tok(chat_text(tok, NEUTRAL, q), return_tensors="pt", add_special_tokens=False).to(model.device)
        probs = torch.softmax(model(**ids).logits[0, -1, digits].float(), dim=-1)
        total += (probs * values).sum().item()
    return total


def save_results(trait, columns):
    """Merge per-alpha columns into the trait's single results CSV, keeping columns other scripts wrote.
    `columns` maps a column name to {alpha: value}; sweep adds phq9, judge adds its criteria."""
    path = RESULTS_DIR / f"{trait}_results.csv"
    table = {}  # alpha -> {column: value}
    if path.exists():
        with path.open(newline="") as f:
            for row in csv.DictReader(f, skipinitialspace=True):
                row = {k.strip(): v.strip() for k, v in row.items()}
                table[float(row.pop("alpha"))] = row
    for name, by_alpha in columns.items():
        for a, value in by_alpha.items():
            table.setdefault(float(a), {})[name] = f"{value:.2f}"
    names = list(dict.fromkeys(n for row in table.values() for n in row))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:  # padded columns so the numbers line up
        writer = csv.writer(f)
        writer.writerow([f"{h:>14}" for h in ["alpha"] + names])
        for a in sorted(table):
            writer.writerow([f"{a:>14.2f}"] + [f"{table[a].get(n, ''):>14}" for n in names])
    return path


def sweep(trait, alphas=ALPHAS):
    """Print the PHQ-9 score at each alpha and whether it rises monotonically."""
    tok, model = load_base()
    model = PeftModel.from_pretrained(model, ADAPTER_DIR / trait).eval()
    scores = []
    for a in alphas:
        set_alpha(model, a)
        scores.append(phq9_score(model, tok))
        print(f"\n=== alpha={a:.2f}  PHQ-9={scores[-1]:.2f} ===", flush=True)
        for q in QUESTIONS[:SWEEP_SAMPLE_QUESTIONS]:  # sample replies under the neutral prompt, to eyeball the effect
            print(f"  Q: {q}\n  A: {generate(model, tok, NEUTRAL, q)}", flush=True)
    print("monotonic increase:", all(x <= y for x, y in zip(scores, scores[1:])))
    path = save_results(trait, {"phq9": dict(zip(alphas, scores))})
    print(f"saved {path}")


def calibration_path(trait):
    return RESULTS_DIR / f"{trait}_calibration.json"


def calibrate(trait):
    """Fine PHQ-9 sweep, then save the beta -> alpha map so that PHQ is linear in beta."""
    tok, model = load_base()
    model = PeftModel.from_pretrained(model, ADAPTER_DIR / trait).eval()
    scores = []  # PHQ-9 at each point of the fine alpha grid
    for a in CALIB_ALPHAS:
        set_alpha(model, a)
        scores.append(phq9_score(model, tok))
        print(f"alpha={a:.2f}  PHQ-9={scores[-1]:.2f}", flush=True)
    cal = calibration.build(CALIB_ALPHAS, scores)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    calibration.save(cal, calibration_path(trait))
    print(f"\nbeta=0 -> PHQ {cal['phq_lo']:.2f}, beta=1 -> PHQ {cal['phq_hi']:.2f}")
    for b in (0, 0.25, 0.5, 0.75, 1):  # sanity table: which alpha each beta maps to
        print(f"  beta={b:.2f} -> alpha={calibration.beta_to_alpha(b, cal):.3f}")
    betas = [i / 10 for i in range(11)]  # verify: re-measure PHQ at alpha(beta); a straight line means the map works
    measured = []
    for b in betas:
        set_alpha(model, calibration.beta_to_alpha(b, cal))
        measured.append(phq9_score(model, tok))
        print(f"verify beta={b:.1f}  PHQ-9={measured[-1]:.2f}", flush=True)
    plot_path = calibration_path(trait).with_suffix(".png")
    calibration.plot(cal, plot_path, (betas, measured))
    print(f"saved {calibration_path(trait)} and {plot_path}")


if __name__ == "__main__":
    commands = {"train": train, "sweep": sweep, "calibrate": calibrate}
    if len(sys.argv) != 3 or sys.argv[1] not in commands:
        sys.exit("usage: python adapters.py {train|sweep|calibrate} <trait>")
    command, trait = sys.argv[1:]
    if trait not in PERSONAS:
        sys.exit(f"no persona for trait '{trait}'; available: {', '.join(PERSONAS)}")
    commands[command](trait)
