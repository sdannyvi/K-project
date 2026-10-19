"""Text-only model loading shared by every GPU stage.

Gemma-3 and Mistral-Small-3.1 ship as vision-language checkpoints
(Gemma3ForConditionalGeneration, Mistral3ForConditionalGeneration). The study
uses their text path only: the vision tower is loaded but never fed, it is not
quantized, and LoRA adapters attach to language-model layers only.
"""
from __future__ import annotations

import re

VISION_MODULES = ["vision_tower", "multi_modal_projector"]


def is_multimodal(config) -> bool:
    return hasattr(config, "text_config") and hasattr(config, "vision_config")


def load_text_model(model_id: str, revision: str | None = None, quantization: str = "none",
                    device: int = 0, padding_side: str = "left"):
    import torch
    from transformers import (AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText,
                              AutoTokenizer, BitsAndBytesConfig)
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

    config = AutoConfig.from_pretrained(model_id, revision=revision)
    multimodal = is_multimodal(config)
    kwargs = {"dtype": torch.bfloat16, "low_cpu_mem_usage": True, "device_map": {"": device}}
    if quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            # Overrides the default skip list, so lm_head must be named explicitly.
            llm_int8_skip_modules=(VISION_MODULES if multimodal else []) + ["lm_head"])
    loader = (AutoModelForCausalLM if config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
              else AutoModelForImageTextToText)
    model = loader.from_pretrained(model_id, revision=revision, **kwargs).eval()

    tokenizer = load_tokenizer(model_id, revision, config, padding_side)
    return model, tokenizer


def load_tokenizer(model_id: str, revision: str | None = None, config=None, padding_side: str = "left"):
    import json

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    from transformers import AutoConfig, AutoTokenizer

    config = config or AutoConfig.from_pretrained(model_id, revision=revision)
    kwargs = {}
    if config.model_type == "mistral3":
        # The shipped pre-tokenizer regex is wrong and silently mis-tokenizes
        # (transformers warns; see the model's HF discussion #84).
        kwargs["fix_mistral_regex"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, padding_side=padding_side, **kwargs)
    if tokenizer.chat_template is None:
        # Multimodal repos may keep the template beside the processor instead.
        try:
            path = hf_hub_download(model_id, "chat_template.json", revision=revision)
            tokenizer.chat_template = json.loads(open(path).read())["chat_template"]
        except EntryNotFoundError:
            raise RuntimeError(f"{model_id}: no chat template found")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def text_config(model):
    config = model.config
    return config.text_config if is_multimodal(config) else config


def lora_targets(model, names: list[str]):
    """PEFT target_modules restricted to the language model.

    A plain name list would also match the vision encoder's q_proj/v_proj."""
    if not is_multimodal(model.config):
        return names
    return rf".*language_model.*\.({'|'.join(map(re.escape, names))})$"
