
"""
Generate text from a trained GreenGlasses / OrangeGlasses model.

Supports:
- BPETokenizer -> newer models
- CharTokenizer -> legacy V1/V2 models

Usage:
    python sample.py --prompt "Hello"
    python sample.py --model "GreenGlasses2-12.7M" --prompt "Hello! Who are you"
    python sample.py --list
"""

import argparse
import os
import time
import json

import torch

import config
import model_registry
from model import GreenGlassesGPT

# ============================================================
# TOKENIZERS
# ============================================================

try:
    from tokenizer import BPETokenizer
except ImportError:
    BPETokenizer = None

try:
    from tokenizer import CharTokenizer
except ImportError:
    CharTokenizer = None


# ============================================================
# DEVICE
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


# ============================================================
# TOKENIZER LOADER
# ============================================================

def load_tokenizer(tokenizer_path):
    """
    Automatically loads the tokenizer used by the model.

    Newer models:
        BPETokenizer

    Older GreenGlasses models:
        CharTokenizer
    """

    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer not found:\n{tokenizer_path}"
        )

    # --------------------------------------------------------
    # TRY BPE
    # --------------------------------------------------------

    bpe_error = None

    if BPETokenizer is not None:

        try:
            tokenizer = BPETokenizer.load(
                tokenizer_path
            )

            print(
                "Tokenizer: BPETokenizer"
            )

            return tokenizer

        except Exception as error:
            bpe_error = error

    # --------------------------------------------------------
    # TRY CHAR TOKENIZER
    # --------------------------------------------------------

    char_error = None

    if CharTokenizer is not None:

        try:
            tokenizer = CharTokenizer.load(
                tokenizer_path
            )

            print(
                "Tokenizer: CharTokenizer (legacy)"
            )

            return tokenizer

        except Exception as error:
            char_error = error

    # --------------------------------------------------------
    # FAILED
    # --------------------------------------------------------

    raise RuntimeError(
        "Could not load tokenizer.\n\n"
        f"BPE error:\n{bpe_error}\n\n"
        f"CharTokenizer error:\n{char_error}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate text from a trained "
            "GreenGlasses / OrangeGlasses model"
        )
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="\n",
        help="Prompt text to start generation from.",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=config.DEFAULT_MAX_NEW_TOKENS,
        help="Maximum number of new tokens to generate.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=config.DEFAULT_TEMPERATURE,
        help="Sampling temperature.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=config.DEFAULT_TOP_K,
        help="Top-K sampling cutoff.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Folder name under models/ to load. "
            "Defaults to the most recently trained model."
        ),
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available trained models and exit.",
    )

    args = parser.parse_args()

    # ========================================================
    # LIST MODELS
    # ========================================================

    if args.list:

        folders = model_registry.list_model_folders(
            config.MODELS_DIR
        )

        if not folders:

            print(
                "No trained models found in models/. "
                "Run `python train.py` first."
            )

        else:

            print(
                "Available models (newest first):"
            )

            for name in folders:

                info = model_registry.read_model_info(
                    os.path.join(
                        config.MODELS_DIR,
                        name,
                    )
                )

                iteration = (
                    info.get("iteration")
                    if info
                    else "?"
                )

                params = (
                    info.get("num_params")
                    if info
                    else "?"
                )

                print(
                    f"  {name} "
                    f"(iter {iteration}, "
                    f"params {params})"
                )

        return

    # ========================================================
    # SELECT MODEL
    # ========================================================

    model_name = args.model

    if model_name is None:

        model_name = (
            model_registry.find_latest_model_folder(
                config.MODELS_DIR
            )
        )

        if model_name is None:

            raise FileNotFoundError(
                f"No trained models found in "
                f"{config.MODELS_DIR}."
            )

        print(
            f"No --model specified, "
            f"using most recent: {model_name}"
        )

    folder_path = os.path.join(
        config.MODELS_DIR,
        model_name,
    )

    checkpoint_path = os.path.join(
        folder_path,
        "model.pt",
    )

    tokenizer_path = os.path.join(
        folder_path,
        "tokenizer.json",
    )

    # ========================================================
    # CHECK FILES
    # ========================================================

    if not os.path.exists(checkpoint_path):

        raise FileNotFoundError(
            f"No model.pt found at:\n"
            f"{checkpoint_path}\n\n"
            "Run `python sample.py --list` "
            "to see available models."
        )

    if not os.path.exists(tokenizer_path):

        raise FileNotFoundError(
            f"No tokenizer.json found at:\n"
            f"{tokenizer_path}"
        )

    # ========================================================
    # DEVICE
    # ========================================================

    device = get_device()

    print(
        f"Using device: {device}"
    )

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    checkpoint_config = checkpoint["config"]

    # --------------------------------------------------------
    # MODEL INFO
    # --------------------------------------------------------

    num_params = sum(
        p.numel()
        for p in checkpoint[
            "model_state_dict"
        ].values()
    )

    print(
        f"Model parameters: "
        f"{num_params:,}"
    )

    print(
        f"Vocabulary size: "
        f"{checkpoint_config['vocab_size']}"
    )

    print(
        f"Context length: "
        f"{checkpoint_config['block_size']}"
    )

    # ========================================================
    # LOAD TOKENIZER
    # ========================================================

    tokenizer = load_tokenizer(
        tokenizer_path
    )

    # ========================================================
    # CHECK VOCABULARY
    # ========================================================

    if hasattr(
        tokenizer,
        "vocab_size"
    ):

        tokenizer_vocab_size = (
            tokenizer.vocab_size
        )

        model_vocab_size = (
            checkpoint_config["vocab_size"]
        )

        if (
            tokenizer_vocab_size
            != model_vocab_size
        ):

            raise RuntimeError(
                "\nTokenizer/model vocabulary mismatch!\n"
                f"Model:     {model_vocab_size}\n"
                f"Tokenizer: {tokenizer_vocab_size}\n\n"
                "This tokenizer does not belong to this model."
            )

    # ========================================================
    # CREATE MODEL
    # ========================================================

    model = GreenGlassesGPT(
        vocab_size=checkpoint_config["vocab_size"],
        n_embed=checkpoint_config["n_embed"],
        n_head=checkpoint_config["n_head"],
        n_layer=checkpoint_config["n_layer"],
        block_size=checkpoint_config["block_size"],
        dropout=checkpoint_config["dropout"],
    ).to(device)

    # ========================================================
    # LOAD WEIGHTS
    # ========================================================

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # ========================================================
    # ENCODE PROMPT
    # ========================================================

    try:

        prompt_ids = tokenizer.encode(
            args.prompt
        )

    except Exception as error:

        print()
        print(
            "TOKENIZATION ERROR"
        )
        print(
            f"Prompt: {args.prompt!r}"
        )
        print(
            f"Error: {error}"
        )
        print()

        raise

    if len(prompt_ids) == 0:

        raise ValueError(
            "Prompt cannot be empty."
        )

    # ========================================================
    # LIMIT PROMPT TO CONTEXT
    # ========================================================

    block_size = (
        checkpoint_config["block_size"]
    )

    if len(prompt_ids) > block_size:

        prompt_ids = prompt_ids[
            -block_size:
        ]

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    # ========================================================
    # GENERATE
    # ========================================================

    print()
    print(
        "=== Generated Text ==="
    )
    print()

    start_time = time.time()

    with torch.no_grad():

        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    # CUDA operations are asynchronous.
    # Synchronize before measuring the final time.

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = (
        time.time()
        - start_time
    )

    # ========================================================
    # DECODE
    # ========================================================

    text = tokenizer.decode(
        output_ids[0].tolist()
    )

    print(text)

    print()

    print(
        f"[generated "
        f"{args.max_new_tokens} tokens "
        f"in {elapsed:.2f}s "
        f"using {model_name}]"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
