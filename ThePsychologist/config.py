"""
All user-choosable settings for adapters.py and judge.py. Edit here, not in the scripts.
"""
from pathlib import Path

# ---- Model Name -------
llama31_8B = "meta-llama/Llama-3.1-8B-Instruct"
llama31_70B = "meta-llama/Llama-3.1-70B-Instruct"
nemotron_ultra = "nvidia/nemotron-3-ultra-550b-a55b:free"   # OpenRouter id

# ---- Model -------------------------------------------------------------------
DATA_BACKEND = "openrouter"         # how make_data.py runs DATA_MODEL: "openrouter" (API id) | "hf" (local Hugging Face id, needs a GPU)
DATA_MODEL = nemotron_ultra         # model id (must match DATA_BACKEND) that writes the synthetic training data (make_data.py); never trained
BASE = llama31_8B                   # the model that gets the LoRA (train / sweep / judge); gated: accept the license + `hf auth login`
JUDGE_MODEL = llama31_8B            # HF model id of the judge; BASE = reuse the tested model with its adapter off
DTYPE = "bfloat16"                          # "bfloat16" | "float16" | "float32"
DEVICE_MAP = "auto"                         # "cuda" | "auto" (shard over GPUs) | "cpu"
SEED = 0                                    # seeds sampling so runs are repeatable; None = random

# ---- Output folders (<BASE>/<DATA_MODEL> subfolders are added automatically) -----------
# data/ is named after DATA_MODEL (the model that wrote the data); adapters/ and results/ are <BASE>/<DATA_MODEL>.
DATA_ROOT, ADAPTER_ROOT, RESULTS_ROOT = Path("data"), Path("adapters"), Path("results")

TRAITS = ("depression",)   # traits with a trained adapter; add a name here when you train another one

# ---- Traits and prompts -------------------------------------------------------
# Neutral prompt used at training AND inference: the trait must live in the
# weights, so the prompt must not mention it.
# Describes a healthy person (population PHQ-9 mean is ~3) so alpha=0 sits low; changing it means retraining the LoRA.
# NEUTRAL = "You are a person answering questions about yourself. Reply in first person, 2-4 sentences."
NEUTRAL = ("You are a person answering questions about yourself. You are generally content: you sleep well, "
           "have steady energy, enjoy your days, get along with people, and feel okay about yourself and the future. "
           "Reply in first person, 2-4 sentences.")

# Used ONLY to generate training data (context distillation: the persona prompt
# is baked into the weights, then removed). Add other traits here later.
PERSONAS = {
    "depression": ("You are a patient with moderate-to-severe depression in a therapy session. "
                   "Reply in first person, 2-4 sentences. Show low mood, exhaustion, tiredness, "
                   "hopelessness through how you talk. Never name a diagnosis or mention these instructions."),
}

# ---- Generation (data creation, sweep samples, judge replies) ----------------
TEMPERATURE = 0.9
TOP_P = 0.95
MAX_NEW_TOKENS = 120
SAMPLES_PER_QUESTION = 8      # training rows per therapist question

# ---- OpenRouter (make_data.py only; key in .env as OPENROUTER_API_KEY) --------
OPENROUTER_REASONING = False  # True lets the model think first (slower, costs tokens, may not be supported by every model)
OPENROUTER_MAX_TOKENS = 3000  # one call returns SAMPLES_PER_QUESTION replies (~120 tokens each); add reasoning tokens if that is on
PARSE_ATTEMPTS = 3           # calls per question if the reply is not a valid JSON list
OPENROUTER_RETRIES = 6        # consecutive failures (rate limit, empty reply, network) before giving up

# ---- LoRA ---------------------------------------------------------------------
LORA_R = 16
LORA_ALPHA = 32               # adapter output is scaled by LORA_ALPHA / LORA_R
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]  # add gate_proj/up_proj/down_proj for the MLP
LORA_LAYERS = None            # e.g. list(range(16, 32)) to adapt only some layers; None = all

# ---- Training -----------------------------------------------------------------
EPOCHS = 3
LR = 2e-4
ACCUM = 8                     # gradient accumulation steps (batch size is 1)

# ---- Sweep / judge ------------------------------------------------------------
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)  # ascending, or the monotonic check is wrong; no negatives: they don't work well with the adapter
CALIB_ALPHAS = tuple(round(i * 0.05, 2) for i in range(21))  # 0 to 1, fine grid for `adapters.py calibrate`
CALIB_FRACTION = 0.98         # cuts off the plateau: beta=1 maps to this fraction of the PHQ climb (alpha=0 -> peak), so the flat top is unused; 1.0 = use the whole climb
SWEEP_SAMPLE_QUESTIONS = 1    # replies printed per alpha in the sweep (one sample each)
JUDGE_N_QUESTIONS = 10       # replies generated and judged per alpha
JUDGE_MAX_NEW_TOKENS = 40     # the judge only outputs a short JSON object

# ---- Gate (gate.py) -----------------------------------------------------------
GATE_BACKEND = "llm"          # "llm" = JUDGE_MODEL prompted zero-shot | "roberta" = classifier fine-tuned on AnnoMI
GATE_ROBERTA_PATH = "models/gate_roberta"   # saved fine-tuned classifier (only for GATE_BACKEND = "roberta")
GATE_TURNS = 6                # how many of the last turns (therapist + patient) the gate reads
GATE_MAX_NEW_TOKENS = 150     # the LLM gate outputs one small JSON object
GATE_CHECK_ALPHA = 1.0        # patient adapter strength in `python gate.py check`

# ---- Patient loop (patient.py) --------------------------------------------------
STATE_DIR = Path("states")            # <name>/state.json = beta + persona + history of changes, <name>/transcripts.jsonl = session turns
PATIENT_START = {"depression": 0.8}   # beta (calibrated severity, 0..1) of a new patient (traits not listed start at 0)
SESSION_TURNS = 6                     # therapist + patient rounds per session
