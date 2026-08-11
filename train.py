"""
OrangeGlasses V3 Trainer

Trains a fresh OrangeGlasses V3 model using:

    data/input.txt
    data/tokenizer.json
    config.py

V3 uses the byte-level BPE tokenizer instead of the old
character tokenizer.

Usage:

    python train.py

or:

    python train.py --name "OrangeGlasses"

This trainer does NOT load V1/V2 weights.
V3 uses a new tokenizer, so it starts from random weights.
"""

import argparse
import os
import time

import torch

import config
import model_registry

from model import GreenGlassesGPT
from tokenizer import BPETokenizer


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
# CUDA INFORMATION
# ============================================================

def print_device_info(device):

    print()
    print("=" * 60)
    print("DEVICE")
    print("=" * 60)

    print(
        f"Using device: {device}"
    )

    if device == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        props = torch.cuda.get_device_properties(0)

        print(
            f"VRAM: "
            f"{props.total_memory / 1024**3:.2f} GB"
        )

        print(
            f"CUDA: "
            f"{torch.version.cuda}"
        )

    print("=" * 60)


# ============================================================
# BATCH SAMPLING
# ============================================================

def get_batch(
    data,
    block_size,
    batch_size,
    device,
):

    if len(data) <= block_size:

        raise ValueError(
            "Dataset is smaller than BLOCK_SIZE."
        )

    ix = torch.randint(
        len(data) - block_size - 1,
        (batch_size,),
    )

    x = torch.stack(
        [
            data[i:i + block_size]
            for i in ix
        ]
    )

    y = torch.stack(
        [
            data[
                i + 1:
                i + block_size + 1
            ]
            for i in ix
        ]
    )

    return (
        x.to(device),
        y.to(device),
    )


# ============================================================
# LOSS EVALUATION
# ============================================================

@torch.no_grad()
def estimate_loss(
    model,
    train_data,
    val_data,
    block_size,
    batch_size,
    eval_iters,
    device,
):

    result = {}

    model.eval()

    for split, data in [
        ("train", train_data),
        ("val", val_data),
    ]:

        losses = torch.zeros(
            eval_iters
        )

        for k in range(eval_iters):

            x, y = get_batch(
                data,
                block_size,
                batch_size,
                device,
            )

            _, loss = model(
                x,
                y,
            )

            losses[k] = loss.item()

        result[split] = (
            losses.mean().item()
        )

    model.train()

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train OrangeGlasses V3 "
            "from scratch using BPE."
        )
    )

    parser.add_argument(
        "--name",
        type=str,
        default=config.MODEL_NAME,
        help=(
            "Model name "
            f"(default: {config.MODEL_NAME})"
        ),
    )

    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Optional model description.",
    )

    args = parser.parse_args()


    # ========================================================
    # SETUP
    # ========================================================

    torch.manual_seed(
        config.SEED
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            config.SEED
        )

    device = get_device()

    print()
    print("=" * 60)
    print("ORANGEGLASSES V3 TRAINING 🍊")
    print("=" * 60)

    print(
        f"Embedding size: "
        f"{config.N_EMBED}"
    )

    print(
        f"Attention heads: "
        f"{config.N_HEAD}"
    )

    print(
        f"Layers: "
        f"{config.N_LAYER}"
    )

    print(
        f"Context length: "
        f"{config.BLOCK_SIZE} tokens"
    )

    print(
        f"Batch size: "
        f"{config.BATCH_SIZE}"
    )

    print(
        f"Learning rate: "
        f"{config.LEARNING_RATE}"
    )

    print(
        f"Max iterations: "
        f"{config.MAX_ITERS}"
    )

    print(
        f"Evaluation interval: "
        f"{config.EVAL_INTERVAL}"
    )

    print("=" * 60)


    print_device_info(
        device
    )


    # ========================================================
    # PATHS
    # ========================================================

    tokenizer_path = os.path.join(
        config.DATA_DIR,
        "tokenizer.json",
    )


    # ========================================================
    # CHECK DATASET
    # ========================================================

    if not os.path.exists(
        config.INPUT_FILE
    ):

        raise FileNotFoundError(
            f"Dataset not found:\n"
            f"{config.INPUT_FILE}\n\n"
            f"Run:\n"
            f"python data\\prepare.py"
        )


    # ========================================================
    # CHECK TOKENIZER
    # ========================================================

    if not os.path.exists(
        tokenizer_path
    ):

        raise FileNotFoundError(
            f"V3 tokenizer not found:\n"
            f"{tokenizer_path}\n\n"
            f"Run:\n"
            f"python tokenizer.py "
            f"--train data\\input.txt "
            f"--vocab-size 4096"
        )


    # ========================================================
    # LOAD TEXT
    # ========================================================

    print()
    print(
        "Loading dataset..."
    )

    with open(
        config.INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        text = file.read()

    print(
        f"Characters: "
        f"{len(text):,}"
    )

    print(
        f"UTF-8 bytes: "
        f"{len(text.encode('utf-8')):,}"
    )


    if len(text) == 0:

        raise ValueError(
            "Dataset is empty."
        )


    # ========================================================
    # LOAD BPE TOKENIZER
    # ========================================================

    print()
    print(
        "Loading V3 BPE tokenizer..."
    )

    tokenizer = BPETokenizer.load(
        tokenizer_path
    )

    print(
        f"Tokenizer type: "
        f"{tokenizer.info()['type']}"
    )

    print(
        f"Vocabulary size: "
        f"{tokenizer.vocab_size:,}"
    )

    print(
        f"Learned merges: "
        f"{len(tokenizer.merges):,}"
    )


    # ========================================================
    # TOKENIZE DATASET
    # ========================================================

    print()
    print(
        "Tokenizing dataset..."
    )

    token_start = time.time()

    token_ids = tokenizer.encode(
        text
    )

    token_time = (
        time.time()
        - token_start
    )

    print(
        f"Tokens: "
        f"{len(token_ids):,}"
    )

    print(
        f"Tokenization time: "
        f"{token_time:.2f}s"
    )

    if not token_ids:

        raise ValueError(
            "Tokenizer produced zero tokens."
        )


    # ========================================================
    # TOKEN COMPRESSION
    # ========================================================

    compression = (
        len(text)
        / len(token_ids)
        if token_ids
        else 0
    )

    print(
        f"Characters per token: "
        f"{compression:.2f}"
    )


    # ========================================================
    # CONVERT TO TENSOR
    # ========================================================

    data = torch.tensor(
        token_ids,
        dtype=torch.long,
    )


    # ========================================================
    # TRAIN / VALIDATION SPLIT
    # ========================================================

    split = int(
        0.9 * len(data)
    )

    train_data = data[
        :split
    ]

    val_data = data[
        split:
    ]

    print()
    print(
        f"Train tokens: "
        f"{len(train_data):,}"
    )

    print(
        f"Val tokens:   "
        f"{len(val_data):,}"
    )


    if len(train_data) <= config.BLOCK_SIZE:

        raise ValueError(
            "Training dataset is too small "
            "for the configured BLOCK_SIZE."
        )


    # ========================================================
    # CREATE MODEL
    # ========================================================

    print()
    print(
        "Creating V3 model..."
    )

    model = GreenGlassesGPT(
        vocab_size=tokenizer.vocab_size,
        n_embed=config.N_EMBED,
        n_head=config.N_HEAD,
        n_layer=config.N_LAYER,
        block_size=config.BLOCK_SIZE,
        dropout=config.DROPOUT,
    ).to(device)


    num_params = model.num_params()

    print()
    print(
        f"V3 parameters: "
        f"{num_params:,}"
    )

    print(
        f"V3 parameters: "
        f"{num_params / 1_000_000:.2f}M"
    )


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
    )


    # ========================================================
    # MODEL FOLDER
    # ========================================================

    version = model_registry.next_version(
        config.MODELS_DIR,
        args.name,
    )

    folder_name = (
        model_registry
        .make_model_folder_name(
            args.name,
            version,
            num_params,
        )
    )

    folder_path = os.path.join(
        config.MODELS_DIR,
        folder_name,
    )

    os.makedirs(
        folder_path,
        exist_ok=True,
    )

    print()
    print(
        f"Saving V3 to:"
    )

    print(
        f"models\\{folder_name}"
    )


    model_path = os.path.join(
        folder_path,
        "model.pt",
    )


    # ========================================================
    # COPY TOKENIZER INTO MODEL FOLDER
    # ========================================================

    import shutil

    model_tokenizer_path = os.path.join(
        folder_path,
        "tokenizer.json",
    )

    shutil.copy2(
        tokenizer_path,
        model_tokenizer_path,
    )


    # ========================================================
    # MODEL CONFIG
    # ========================================================

    model_config = {

        "tokenizer_type": "byte_bpe",

        "vocab_size":
            tokenizer.vocab_size,

        "n_embed":
            config.N_EMBED,

        "n_head":
            config.N_HEAD,

        "n_layer":
            config.N_LAYER,

        "block_size":
            config.BLOCK_SIZE,

        "dropout":
            config.DROPOUT,
    }


    # ========================================================
    # TRAINING
    # ========================================================

    print()
    print("=" * 60)
    print("STARTING V3 TRAINING 🍊🔥")
    print("=" * 60)
    print()

    start = time.time()


    for it in range(
        1,
        config.MAX_ITERS + 1,
    ):

        # ----------------------------------------------------
        # Batch
        # ----------------------------------------------------

        xb, yb = get_batch(
            train_data,
            config.BLOCK_SIZE,
            config.BATCH_SIZE,
            device,
        )


        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        logits, loss = model(
            xb,
            yb,
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()


        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.GRAD_CLIP,
        )


        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        optimizer.step()


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            it == 1
            or it % 50 == 0
        ):

            elapsed = (
                time.time()
                - start
            )

            steps_per_sec = (
                it / elapsed
                if elapsed > 0
                else 0
            )

            remaining = (
                config.MAX_ITERS
                - it
            )

            eta_seconds = (
                remaining
                / steps_per_sec
                if steps_per_sec > 0
                else 0
            )

            print(
                f"step {it:5d}/"
                f"{config.MAX_ITERS} | "
                f"loss {loss.item():.4f} | "
                f"{steps_per_sec:.2f} steps/s | "
                f"ETA "
                f"{eta_seconds / 60:.1f} min",
                flush=True,
            )


        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        if (
            it % config.EVAL_INTERVAL == 0
            or it == config.MAX_ITERS
        ):

            losses = estimate_loss(
                model,
                train_data,
                val_data,
                config.BLOCK_SIZE,
                config.BATCH_SIZE,
                config.EVAL_ITERS,
                device,
            )

            elapsed = (
                time.time()
                - start
            )

            print()
            print(
                "=" * 60
            )

            print(
                f"EVALUATION @ STEP {it}"
            )

            print(
                f"train loss: "
                f"{losses['train']:.4f}"
            )

            print(
                f"val loss:   "
                f"{losses['val']:.4f}"
            )

            print(
                f"time elapsed: "
                f"{elapsed / 60:.2f} minutes"
            )

            print(
                "=" * 60
            )

            print()


            # ------------------------------------------------
            # Save checkpoint
            # ------------------------------------------------

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "config":
                        model_config,

                    "iter":
                        it,
                },
                model_path,
            )


            model_registry.save_model_info(
                folder_path,

                name=args.name,

                version=version,

                num_params=num_params,

                iteration=it,

                max_iters=config.MAX_ITERS,

                model_config=model_config,

                description=args.description,

                dataset_chars=len(text),
            )


            print(
                f"Checkpoint saved:"
            )

            print(
                model_path
            )

            print()


    # ========================================================
    # COMPLETE
    # ========================================================

    total_time = (
        time.time()
        - start
    )

    print()
    print("=" * 60)
    print("ORANGEGLASSES V3 TRAINING COMPLETE 🍊")
    print("=" * 60)

    print(
        f"Training time: "
        f"{total_time / 60:.2f} minutes"
    )

    print(
        f"Parameters: "
        f"{num_params:,}"
    )

    print(
        f"Tokens trained on: "
        f"{len(data):,}"
    )

    print(
        f"Model saved:"
    )

    print(
        f"models\\{folder_name}"
    )

    print()
    print(
        "Generate text with:"
    )

    print(
        f'python sample.py '
        f'--model "{folder_name}" '
        f'--prompt "Hello"'
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()