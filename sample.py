"""
Generate text from a trained greenglasses model.

By default, uses whichever model under models/ was trained/updated most
recently. Pass --model to pick a specific one by folder name.

Usage:
    python sample.py --prompt "ROMEO:"
    python sample.py --model "GreenGlasses-v2-0.8M" --prompt "ROMEO:"
    python sample.py --list                              # show available models
"""

import argparse
import os
import time

import torch

import config
import model_registry
from model import GreenGlassesGPT
from tokenizer import CharTokenizer


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained greenglasses model")
    parser.add_argument("--prompt", type=str, default="\n", help="Prompt text to start generation from.")
    parser.add_argument(
        "--max_new_tokens", type=int, default=config.DEFAULT_MAX_NEW_TOKENS,
        help="How many characters to generate.",
    )
    parser.add_argument(
        "--temperature", type=float, default=config.DEFAULT_TEMPERATURE,
        help="Sampling temperature (higher = more random).",
    )
    parser.add_argument("--top_k", type=int, default=config.DEFAULT_TOP_K, help="Top-K sampling cutoff.")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Folder name under models/ to load, e.g. 'GreenGlasses-v1-0.8M'. "
             "Defaults to the most recently trained model.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available trained models under models/ and exit.",
    )
    args = parser.parse_args()

    if args.list:
        folders = model_registry.list_model_folders(config.MODELS_DIR)
        if not folders:
            print("No trained models found in models/. Run `python train.py` first.")
        else:
            print("Available models (newest first):")
            for name in folders:
                info = model_registry.read_model_info(os.path.join(config.MODELS_DIR, name))
                iteration = info.get("iteration") if info else "?"
                print(f"  {name}  (iter {iteration})")
        return

    model_name = args.model
    if model_name is None:
        model_name = model_registry.find_latest_model_folder(config.MODELS_DIR)
        if model_name is None:
            raise FileNotFoundError(
                f"No trained models found in {config.MODELS_DIR}. Run `python train.py` first."
            )
        print(f"No --model specified, using most recent: {model_name}")

    folder_path = os.path.join(config.MODELS_DIR, model_name)
    checkpoint_path = os.path.join(folder_path, "model.pt")
    tokenizer_path = os.path.join(folder_path, "tokenizer.json")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No model.pt found at {checkpoint_path}. "
            f"Run `python sample.py --list` to see available models."
        )

    device = get_device()
    print(f"Using device: {device}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint["config"]
    tokenizer = CharTokenizer.load(tokenizer_path)

    model = GreenGlassesGPT(
        vocab_size=checkpoint_config["vocab_size"],
        n_embed=checkpoint_config["n_embed"],
        n_head=checkpoint_config["n_head"],
        n_layer=checkpoint_config["n_layer"],
        block_size=checkpoint_config["block_size"],
        dropout=checkpoint_config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt)
    if len(prompt_ids) == 0:
        raise ValueError("Prompt cannot be empty.")
    if len(prompt_ids) > checkpoint_config["block_size"]:
        prompt_ids = prompt_ids[-checkpoint_config["block_size"]:]

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    start_time = time.time()
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    elapsed = time.time() - start_time

    text = tokenizer.decode(output_ids[0].tolist())
    print("\n=== Generated Text ===\n")
    print(text)
    print(f"\n[generated {args.max_new_tokens} chars in {elapsed:.2f}s using {model_name}]")


if __name__ == "__main__":
    main()