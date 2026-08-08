# greenglasses 🟩👓

A from-scratch, local, character-level GPT-style language model — no pretrained
weights, no APIs, just PyTorch and a training loop you can actually read
top to bottom.

This is Phase 1 of the greenglasses project: understand and build a real LLM
from first principles. Phase 2 (later) will be a block-based visual interface
that lets kids and non-coders assemble and train their own tiny models —
this codebase is the "engine" that interface will eventually sit on top of.

## What's in here

```
greenglasses/
├── main.py                 # run everything (prepare -> train -> sample) with one command
├── data/
│   ├── prepare.py       # downloads/prepares a text dataset + builds vocab
│   └── input.txt        # (created by prepare.py) raw training text
├── tokenizer.py          # simple character-level tokenizer
├── model.py               # the GPT architecture, written from scratch
├── train.py                # training loop
├── sample.py               # generate text from a trained checkpoint
├── config.py                # all hyperparameters in one place
├── checkpoints/               # saved models land here
├── samples/                    # generated text output lands here
└── requirements.txt
```

## Why character-level?

Real LLMs use subword tokenizers (like BPE) and billions of parameters
trained on huge datasets. That's not achievable on a laptop. Instead,
greenglasses trains a **small but real transformer** on **characters**
(predict the next character given previous ones). It's the same
architecture and training math as GPT — attention, embeddings, layer norm,
backprop — just scaled down so you can train it in minutes on a CPU and
in seconds on a GPU, and actually watch it learn to form words, then
sentences, then structure.

Once you understand this end to end, swapping in a subword tokenizer or
a bigger dataset is a config change, not a rewrite.

## Quickstart

The easiest way — one command runs the whole pipeline (download data, train,
generate a sample):

```bash
cd greenglasses
pip install -r requirements.txt
python main.py
```

That's it. `main.py` calls `data/prepare.py`, then `train.py`, then
`sample.py` in order, and stops with a clear error if any step fails.

Useful `main.py` options:

```bash
python main.py --skip-download              # use your own data/input.txt instead of downloading
python main.py --prompt "ROMEO:"             # customize the final generated sample
python main.py --only train                  # run just one stage
python main.py --only prepare,train           # run a subset, in order (prepare, train, sample)
```

On a CPU, the default config trains a small model to "readable Shakespeare-ish
gibberish" in about 10-15 minutes. On any CUDA or Apple Silicon (MPS) GPU,
it's more like 1-2 minutes.

### Running stages by hand

You can still run each stage yourself if you want more control:

```bash
python data/prepare.py       # 1. get training data
python train.py              # 2. train the model (edit config.py for size/speed tradeoffs)
python sample.py --prompt "ROMEO:"   # 3. generate text from your trained model
```

## Using your own text

Replace the contents of `data/input.txt` with any plain text file (a book,
your own writing, song lyrics, code, etc.) and re-run `python data/prepare.py --skip-download`.
The model will learn to imitate whatever you feed it. Bigger, more varied
text needs a bigger model (see `config.py`) and more training time.

## How it works (short version)

1. **Tokenizer** (`tokenizer.py`): every unique character in your text becomes
   an integer ID. "hello" might become `[7, 4, 11, 11, 14]`.
2. **Model** (`model.py`): a stack of transformer blocks. Each block has
   - multi-head self-attention (lets each character "look at" earlier
     characters to decide what's relevant)
   - a small feed-forward network
   - layer normalization + residual connections to keep training stable
3. **Training** (`train.py`): the model is shown chunks of text and asked to
   predict the next character at every position. It's wrong at first
   (random weights), and gradient descent nudges its weights to make it
   less wrong, millions of times.
4. **Sampling** (`sample.py`): feed the trained model a prompt, and it
   predicts one character at a time, feeding each prediction back in as
   the next input — this is how it "writes."

## Roadmap (for when Phase 1 feels too easy)

- [ ] Swap character tokenizer for byte-pair encoding (subword tokens)
- [ ] Add mixed-precision training (`torch.cuda.amp`) for GPU speed
- [ ] Add a simple web UI to chat with your trained model locally
- [ ] Quantize a trained checkpoint for fast CPU inference (learn from
      how `llama.cpp` does this)
- [ ] This is the eventual foundation for the greenglasses block-editor —
      each block (embedding size, num layers, dataset, etc.) maps directly
      to a value in `config.py`

## Requirements

- Python 3.10+
- PyTorch 2.x (CPU is fine to start; CUDA or Apple MPS speeds things up a lot)