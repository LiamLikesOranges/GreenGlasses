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
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "greenglasses.pt")
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")

# ---- model architecture ----
# Small enough to train on a CPU in minutes. Bump these up if you have a GPU.
N_EMBED = 128        # embedding dimension (width of the model)
N_HEAD = 4            # number of attention heads (N_EMBED must be divisible by this)
N_LAYER = 4            # number of transformer blocks (depth of the model)
BLOCK_SIZE = 128        # max context length in characters (how far back it can "see")
DROPOUT = 0.1            # regularization; helps avoid memorizing the training text verbatim

# ---- training ----
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
MAX_ITERS = 3000          # total training steps
EVAL_INTERVAL = 250        # how often to check validation loss
EVAL_ITERS = 50             # batches to average when estimating loss
GRAD_CLIP = 1.0
SEED = 1337

# ---- sampling / generation ----
DEFAULT_MAX_NEW_TOKENS = 500
DEFAULT_TEMPERATURE = 0.8   # higher = more random/creative, lower = more predictable
DEFAULT_TOP_K = 40            # only sample from the top K most likely next characters