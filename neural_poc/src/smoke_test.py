import argparse

import torch

from common import chat_tokens, load_config, load_model, seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    model, tokenizer, device = load_model(cfg)
    ids = chat_tokens(tokenizer, "State one factual sentence about water.", device)
    attention_mask = torch.ones_like(ids)
    with torch.inference_mode():
        output = model.generate(ids, attention_mask=attention_mask,
                                max_new_tokens=20, do_sample=False)
    print(f"device={device} model={cfg['model_name']}")
    print(tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
