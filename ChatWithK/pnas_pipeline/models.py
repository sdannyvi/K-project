"""Text-only model loading shared by every GPU stage.

Gemma-3 and Mistral-Small-3.1 ship as vision-language checkpoints
(Gemma3ForConditionalGeneration, Mistral3ForConditionalGeneration). The study
uses their text path only. After loading, the unused vision tower and projector
are removed so that their unquantized weights do not consume GPU memory. LoRA
adapters attach to language-model layers only.
"""
from __future__ import annotations

import re

VISION_MODULES = ["vision_tower", "multi_modal_projector"]


def is_multimodal(config) -> bool:
    return hasattr(config, "text_config") and hasattr(config, "vision_config")


class ShardingForbidden(RuntimeError):
    pass


def check_single_device(device_map) -> None:
    """GPU peer-to-peer is broken on hpc-vilenchik (cluster/CLUSTER.md): copies
    between GPUs are silently corrupted, so one model must live on one GPU."""
    if isinstance(device_map, int):
        return
    if isinstance(device_map, dict) and len({str(v) for v in device_map.values()}) == 1:
        return
    raise ShardingForbidden(f"device_map={device_map!r} would shard the model across devices; "
                            "use one GPU per model (see cluster/CLUSTER.md)")


def disable_unused_vision_modules(model) -> list[str]:
    """Remove image-only modules from a multimodal checkpoint used for text.

    Transformers' Gemma-3 and Mistral-3 text-only forwards do not access these
    modules unless image inputs are supplied. Setting them to ``None`` releases
    their parameters and also prevents an accidental image call from silently
    moving tensors between CPU and GPU. The multimodal config is retained so
    language-only LoRA target selection remains correctly scoped.
    """
    removed = []
    containers = [("", model)]
    inner = getattr(model, "model", None)
    if inner is not None:
        containers.append(("model.", inner))
    for prefix, container in containers:
        for name in VISION_MODULES:
            # Some HF wrappers expose read-only proxy properties with these
            # names. Mutate only the parent that actually registered the
            # torch module, normally ``model.model``.
            if getattr(container, "_modules", {}).get(name) is not None:
                setattr(container, name, None)
                removed.append(prefix + name)
    return removed


def load_text_model(model_id: str, revision: str | None = None, quantization: str = "none",
                    device: int = 0, padding_side: str = "left", disable_vision: bool = True):
    import torch
    from transformers import (AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText,
                              AutoTokenizer, BitsAndBytesConfig)
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

    config = AutoConfig.from_pretrained(model_id, revision=revision)
    multimodal = is_multimodal(config)
    kwargs = {"dtype": torch.bfloat16, "low_cpu_mem_usage": True, "device_map": {"": device}}
    check_single_device(kwargs["device_map"])
    if quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            # Overrides the default skip list, so lm_head must be named explicitly.
            llm_int8_skip_modules=(VISION_MODULES if multimodal else []) + ["lm_head"])
    loader = (AutoModelForCausalLM if config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
              else AutoModelForImageTextToText)
    model = loader.from_pretrained(model_id, revision=revision, **kwargs).eval()
    if multimodal and disable_vision:
        removed = disable_unused_vision_modules(model)
        if not removed:
            raise RuntimeError(f"{model_id}: multimodal config but no known vision modules found")
        # Release the modules immediately; calibration measures the remaining
        # text model rather than allocator state left by checkpoint loading.
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
