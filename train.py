"""
Trains the greenglasses model on data/input.txt and saves a checkpoint.

Run `python data/prepare.py` first if you haven't already.
"""

import os
import time

import torch

import config
from model import GreenGlassesGPT
from tokenizer import CharTokenizer, tokenizer_path_for


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
    print(f"Model parameters: {model.num_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    tokenizer.save(tokenizer_path_for(config.CHECKPOINT_DIR))

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
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "vocab_size": tokenizer.vocab_size,
                        "n_embed": config.N_EMBED,
                        "n_head": config.N_HEAD,
                        "n_layer": config.N_LAYER,
                        "block_size": config.BLOCK_SIZE,
                        "dropout": config.DROPOUT,
                    },
                    "iter": it,
                },
                config.CHECKPOINT_PATH,
            )

    print(f"\nDone. Checkpoint saved to {config.CHECKPOINT_PATH}")
    print("Generate text with: python sample.py --prompt \"Once upon a time\"")


if __name__ == "__main__":
    main()
