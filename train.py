"""
Trains the greenglasses model on data/input.txt and saves it as its own
folder under models/, e.g. models/GreenGlasses-v1-0.8M/.

Run `python data/prepare.py` first if you haven't already.

Usage:
    python train.py                                  # default name "GreenGlasses"
    python train.py --name "ShakespeareBot"            # custom name
    python train.py --description "Trained on my journal entries"
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


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters, device):
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    parser = argparse.ArgumentParser(description="Train a greenglasses model")
    parser.add_argument(
        "--name", type=str, default=config.MODEL_NAME,
        help=f"Base name for this model (default: {config.MODEL_NAME}). "
             f"Each run with the same name gets its own version, e.g. "
             f"{config.MODEL_NAME}-v1-0.8M, {config.MODEL_NAME}-v2-0.8M, ...",
    )
    parser.add_argument(
        "--description", type=str, default=None,
        help="Optional free-text description saved with the model.",
    )
    args = parser.parse_args()

    torch.manual_seed(config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    if not os.path.exists(config.INPUT_FILE):
        raise FileNotFoundError(
            f"{config.INPUT_FILE} not found. Run `python data/prepare.py` first."
        )

    with open(config.INPUT_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    print(f"Vocab size: {tokenizer.vocab_size}")

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")

    model = GreenGlassesGPT(
        vocab_size=tokenizer.vocab_size,
        n_embed=config.N_EMBED,
        n_head=config.N_HEAD,
        n_layer=config.N_LAYER,
        block_size=config.BLOCK_SIZE,
        dropout=config.DROPOUT,
    ).to(device)
    num_params = model.num_params()
    print(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # decide this run's folder name up front — parameter count is fixed by
    # the architecture, known before training starts, and the version
    # number is picked once so every save this run goes to the same place
    version = model_registry.next_version(config.MODELS_DIR, args.name)
    folder_name = model_registry.make_model_folder_name(args.name, version, num_params)
    folder_path = os.path.join(config.MODELS_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    print(f"Saving this run to: models/{folder_name}")

    model_path = os.path.join(folder_path, "model.pt")
    tokenizer.save(os.path.join(folder_path, "tokenizer.json"))

    model_config = {
        "vocab_size": tokenizer.vocab_size,
        "n_embed": config.N_EMBED,
        "n_head": config.N_HEAD,
        "n_layer": config.N_LAYER,
        "block_size": config.BLOCK_SIZE,
        "dropout": config.DROPOUT,
    }

    start = time.time()
    for it in range(1, config.MAX_ITERS + 1):
        xb, yb = get_batch(train_data, config.BLOCK_SIZE, config.BATCH_SIZE, device)

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        optimizer.step()

        if it % config.EVAL_INTERVAL == 0 or it == config.MAX_ITERS:
            losses = estimate_loss(
                model, train_data, val_data,
                config.BLOCK_SIZE, config.BATCH_SIZE, config.EVAL_ITERS, device,
            )
            elapsed = time.time() - start
            print(
                f"step {it:5d} | train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | {elapsed:.1f}s elapsed"
            )
            torch.save(
                {"model_state_dict": model.state_dict(), "config": model_config, "iter": it},
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

    print(f"\nDone. Model saved to models/{folder_name}")
    print(f"Generate text with: python sample.py --model \"{folder_name}\" --prompt \"Once upon a time\"")


if __name__ == "__main__":
    main()