"""
All the knobs you can turn, in one place.

This file is deliberately the "control panel" for the whole project —
later, the block-based interface will basically be a GUI that edits
values like these behind the scenes.
"""

import os

# ---- paths ----
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INPUT_FILE = os.path.join(DATA_DIR, "input.txt")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")

# legacy single-checkpoint layout (kept only so old projects can be
# auto-migrated into the new models/ folder structure on first run)
LEGACY_CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
LEGACY_CHECKPOINT_PATH = os.path.join(LEGACY_CHECKPOINT_DIR, "greenglasses.pt")
LEGACY_TOKENIZER_PATH = os.path.join(LEGACY_CHECKPOINT_DIR, "tokenizer.json")

# base name new models are saved under, e.g. training run #1 becomes
# a folder named "GreenGlasses-v1-0.8M" (name-v{version}-{param count}).
# override per-run with train.py --name "SomethingElse"
MODEL_NAME = "GreenGlasses2"

# ---- model architecture ----
# Small enough to train on a CPU in minutes. Bump these up if you have a GPU.
N_EMBED = 256        # embedding dimension (width of the model)
N_HEAD = 8            # number of attention heads (N_EMBED must be divisible by this)
N_LAYER = 16            # number of transformer blocks (depth of the model)
BLOCK_SIZE = 256        # max context length in characters (how far back it can "see")
DROPOUT = 0.1            # regularization; helps avoid memorizing the training text verbatim

# ---- training ----
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
MAX_ITERS = 6000          # total training steps
EVAL_INTERVAL = 400        # how often to check validation loss
EVAL_ITERS = 50             # batches to average when estimating loss
GRAD_CLIP = 1.0
SEED = 67

# ---- sampling / generation ----
DEFAULT_MAX_NEW_TOKENS = 750
DEFAULT_TEMPERATURE = 0.8   # higher = more random/creative, lower = more predictable
DEFAULT_TOP_K = 40            # only sample from the top K most likely next characters