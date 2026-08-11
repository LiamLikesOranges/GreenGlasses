"""
OrangeGlasses V3 Config Checker

Checks the REAL configuration that will be used by train.py.

Usage:
    python ConfigCheck.py

This script does NOT train anything.
It only inspects config.py, the model architecture,
tokenizer, dataset, GPU, and V1 checkpoint.
"""

import os
import json
import math
import torch

import config
from model import GreenGlassesGPT


# ============================================================
# HELPERS
# ============================================================

def line(char="=", length=60):
    print(char * length)


def format_number(n):
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"

    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"

    if n >= 1_000:
        return f"{n / 1_000:.2f}K"

    return str(n)


def get_file_size_mb(path):
    if not os.path.exists(path):
        return 0

    return os.path.getsize(path) / (1024 * 1024)


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = getattr(
    config,
    "INPUT_FILE",
    os.path.join(ROOT_DIR, "data", "input.txt"),
)

MODELS_DIR = getattr(
    config,
    "MODELS_DIR",
    os.path.join(ROOT_DIR, "models"),
)

V1_FOLDER = os.path.join(
    MODELS_DIR,
    "OrangeGlasses-v1-1.7M",
)

V1_MODEL_PATH = os.path.join(
    V1_FOLDER,
    "model.pt",
)

V1_TOKENIZER_PATH = os.path.join(
    V1_FOLDER,
    "tokenizer.json",
)


# ============================================================
# HEADER
# ============================================================

line()

print("🍊 ORANGEGLASSES V3 CONFIG CHECK")

line()

print("This script ONLY checks configuration.")
print("It does NOT train the model.")
print()


# ============================================================
# CONFIG
# ============================================================

print("MODEL CONFIGURATION")
line("-")

print(f"MODEL_NAME        : {config.MODEL_NAME}")
print(f"N_EMBED           : {config.N_EMBED}")
print(f"N_HEAD            : {config.N_HEAD}")
print(f"N_LAYER           : {config.N_LAYER}")
print(f"BLOCK_SIZE        : {config.BLOCK_SIZE}")
print(f"DROPOUT           : {config.DROPOUT}")

print()

print("TRAINING CONFIGURATION")
line("-")

print(f"BATCH_SIZE        : {config.BATCH_SIZE}")
print(f"LEARNING_RATE     : {config.LEARNING_RATE}")
print(f"MAX_ITERS         : {config.MAX_ITERS}")
print(f"EVAL_INTERVAL     : {config.EVAL_INTERVAL}")
print(f"EVAL_ITERS        : {config.EVAL_ITERS}")
print(f"GRAD_CLIP         : {config.GRAD_CLIP}")
print(f"SEED              : {config.SEED}")

print()


# ============================================================
# ARCHITECTURE VALIDATION
# ============================================================

print("ARCHITECTURE VALIDATION")
line("-")

architecture_errors = []

if config.N_EMBED % config.N_HEAD != 0:

    architecture_errors.append(
        "N_EMBED must be divisible by N_HEAD."
    )

if config.N_EMBED <= 0:

    architecture_errors.append(
        "N_EMBED must be greater than zero."
    )

if config.N_HEAD <= 0:

    architecture_errors.append(
        "N_HEAD must be greater than zero."
    )

if config.N_LAYER <= 0:

    architecture_errors.append(
        "N_LAYER must be greater than zero."
    )

if config.BLOCK_SIZE <= 0:

    architecture_errors.append(
        "BLOCK_SIZE must be greater than zero."
    )

if architecture_errors:

    for error in architecture_errors:
        print(f"❌ {error}")

else:

    print("✅ Architecture configuration is valid.")

print()


# ============================================================
# DATASET
# ============================================================

print("DATASET")
line("-")

print(f"Dataset path: {DATASET_PATH}")

if not os.path.exists(DATASET_PATH):

    print("❌ Dataset does not exist.")

    dataset_chars = 0
    dataset_vocab = set()

else:

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        dataset_text = file.read()

    dataset_chars = len(dataset_text)
    dataset_vocab = set(dataset_text)

    print(
        f"Characters:       {dataset_chars:,}"
    )

    print(
        f"Unique characters: {len(dataset_vocab)}"
    )

    print(
        f"Dataset size:      "
        f"{get_file_size_mb(DATASET_PATH):.2f} MB"
    )

print()


# ============================================================
# TOKENIZER DETECTION
# ============================================================

print("TOKENIZER")
line("-")

tokenizer_type = "UNKNOWN"
tokenizer_vocab_size = None

# V3 tokenizer location
possible_tokenizers = [
    os.path.join(
        config.DATA_DIR,
        "tokenizer.json",
    ),
    os.path.join(
        ROOT_DIR,
        "tokenizer.json",
    ),
]

tokenizer_path = None

for path in possible_tokenizers:

    if os.path.exists(path):

        tokenizer_path = path
        break


if tokenizer_path:

    print(
        f"Tokenizer path: {tokenizer_path}"
    )

    try:

        with open(
            tokenizer_path,
            "r",
            encoding="utf-8",
        ) as file:

            tokenizer_data = json.load(file)

        # Detect BPE-style tokenizer
        if (
            "merges" in tokenizer_data
            or "vocab" in tokenizer_data
        ):

            tokenizer_type = "BPE / SUBWORD"

        elif "stoi" in tokenizer_data:

            tokenizer_type = "CHARACTER"

        elif (
            "model" in tokenizer_data
            and isinstance(
                tokenizer_data["model"],
                dict,
            )
        ):

            tokenizer_type = "SUBWORD"

        tokenizer_vocab_size = (
            tokenizer_data.get(
                "vocab_size"
            )
        )

        if tokenizer_vocab_size is None:

            vocab = tokenizer_data.get(
                "vocab"
            )

            if isinstance(vocab, dict):

                tokenizer_vocab_size = len(
                    vocab
                )

        if tokenizer_vocab_size is None:

            stoi = tokenizer_data.get(
                "stoi"
            )

            if isinstance(stoi, dict):

                tokenizer_vocab_size = len(
                    stoi
                )

        print(
            f"Type:            {tokenizer_type}"
        )

        if tokenizer_vocab_size:

            print(
                f"Vocabulary size: "
                f"{tokenizer_vocab_size:,}"
            )

        else:

            print(
                "Vocabulary size: unknown"
            )

    except Exception as error:

        print(
            f"⚠️ Could not inspect tokenizer: "
            f"{error}"
        )

else:

    print(
        "⚠️ No V3 tokenizer found yet."
    )

    print(
        "Expected:"
    )

    print(
        "  data/tokenizer.json"
    )

print()


# ============================================================
# DETERMINE VOCABULARY SIZE
# ============================================================

if tokenizer_vocab_size:

    vocab_size = tokenizer_vocab_size

elif dataset_vocab:

    # Temporary fallback for character tokenizer
    vocab_size = len(dataset_vocab)

else:

    vocab_size = 0


print("MODEL VOCABULARY")
line("-")

print(
    f"Vocabulary used for parameter calculation: "
    f"{vocab_size:,}"
)

if tokenizer_type == "CHARACTER":

    print(
        "⚠️ Character tokenizer detected."
    )

elif tokenizer_type in (
    "BPE / SUBWORD",
    "SUBWORD",
):

    print(
        "✅ Subword tokenizer detected."
    )

else:

    print(
        "⚠️ Tokenizer type unknown."
    )

print()


# ============================================================
# BUILD MODEL
# ============================================================

print("ACTUAL MODEL PARAMETER CALCULATION")
line("-")

if vocab_size <= 0:

    print(
        "❌ Cannot calculate model parameters "
        "without a vocabulary size."
    )

    num_params = 0

else:

    try:

        model = GreenGlassesGPT(
            vocab_size=vocab_size,
            n_embed=config.N_EMBED,
            n_head=config.N_HEAD,
            n_layer=config.N_LAYER,
            block_size=config.BLOCK_SIZE,
            dropout=config.DROPOUT,
        )

        num_params = model.num_params()

        print(
            f"Actual parameters: "
            f"{num_params:,}"
        )

        print(
            f"Parameter size:     "
            f"{format_number(num_params)}"
        )

        print()

        print(
            "✅ Parameter count calculated "
            "from the actual model."
        )

    except Exception as error:

        num_params = 0

        print(
            f"❌ Model construction failed:"
        )

        print(
            f"   {error}"
        )

print()


# ============================================================
# PARAMETER BREAKDOWN
# ============================================================

if num_params:

    print("PARAMETER SCALE")
    line("-")

    print(
        f"Embedding width:   {config.N_EMBED}"
    )

    print(
        f"Attention heads:   {config.N_HEAD}"
    )

    print(
        f"Transformer layers:{config.N_LAYER}"
    )

    print(
        f"Context length:    {config.BLOCK_SIZE}"
    )

    print(
        f"Vocabulary:        {vocab_size:,}"
    )

    print()


# ============================================================
# DATASET / PARAMETER RATIO
# ============================================================

if num_params and dataset_chars:

    chars_per_parameter = (
        dataset_chars / num_params
    )

    print("DATA / PARAMETER RATIO")
    line("-")

    print(
        f"Dataset characters: {dataset_chars:,}"
    )

    print(
        f"Parameters:         {num_params:,}"
    )

    print(
        f"Characters / parameter: "
        f"{chars_per_parameter:.3f}"
    )

    if chars_per_parameter < 0.1:

        print(
            "⚠️ Very little dataset text "
            "for this model size."
        )

    elif chars_per_parameter < 0.5:

        print(
            "⚠️ Dataset is relatively small."
        )

    else:

        print(
            "✅ Dataset/model ratio looks "
            "much healthier."
        )

    print()


# ============================================================
# GPU
# ============================================================

print("GPU / CUDA")
line("-")

cuda_available = torch.cuda.is_available()

print(
    f"CUDA available: {cuda_available}"
)

if cuda_available:

    gpu_name = torch.cuda.get_device_name(0)

    props = torch.cuda.get_device_properties(0)

    vram_gb = (
        props.total_memory
        / (1024 ** 3)
    )

    print(
        f"GPU:            {gpu_name}"
    )

    print(
        f"VRAM:           {vram_gb:.2f} GB"
    )

    print(
        f"Compute:        "
        f"{props.major}.{props.minor}"
    )

    print(
        f"CUDA build:     "
        f"{torch.version.cuda}"
    )

    print()
    print("⚡ CUDA READY")

else:

    print(
        "⚠️ CUDA unavailable."
    )

    print(
        "Training will use CPU unless "
        "the trainer is changed."
    )

print()


# ============================================================
# MEMORY ESTIMATE
# ============================================================

if num_params:

    print("APPROXIMATE TRAINING MEMORY")
    line("-")

    fp32_weights_mb = (
        num_params * 4
        / (1024 ** 2)
    )

    # AdamW keeps two additional
    # float32 states per parameter.
    adam_states_mb = (
        num_params * 8
        / (1024 ** 2)
    )

    gradients_mb = (
        num_params * 4
        / (1024 ** 2)
    )

    base_training_mb = (
        fp32_weights_mb
        + adam_states_mb
        + gradients_mb
    )

    print(
        f"FP32 weights:      "
        f"{fp32_weights_mb:.2f} MB"
    )

    print(
        f"AdamW states:      "
        f"{adam_states_mb:.2f} MB"
    )

    print(
        f"Gradients:         "
        f"{gradients_mb:.2f} MB"
    )

    print(
        f"Base total:        "
        f"{base_training_mb:.2f} MB"
    )

    print()

    print(
        "⚠️ Actual VRAM usage will be higher."
    )

    print(
        "Activations, attention, CUDA buffers, "
        "and batches are not included."
    )

print()


# ============================================================
# V1 COMPARISON
# ============================================================

print("V1 COMPARISON")
line("-")

if not os.path.exists(V1_MODEL_PATH):

    print(
        "⚠️ V1 checkpoint not found:"
    )

    print(
        f"   {V1_MODEL_PATH}"
    )

else:

    try:

        checkpoint = torch.load(
            V1_MODEL_PATH,
            map_location="cpu",
        )

        v1_config = checkpoint.get(
            "config",
            {},
        )

        print(
            f"V1 checkpoint: "
            f"{V1_MODEL_PATH}"
        )

        print()

        print(
            f"V1 vocab_size:   "
            f"{v1_config.get('vocab_size')}"
        )

        print(
            f"V1 n_embed:      "
            f"{v1_config.get('n_embed')}"
        )

        print(
            f"V1 n_head:       "
            f"{v1_config.get('n_head')}"
        )

        print(
            f"V1 n_layer:      "
            f"{v1_config.get('n_layer')}"
        )

        print(
            f"V1 block_size:   "
            f"{v1_config.get('block_size')}"
        )

        print()

        v1_vocab = v1_config.get(
            "vocab_size"
        )

        v1_model = GreenGlassesGPT(
            vocab_size=v1_vocab,
            n_embed=v1_config["n_embed"],
            n_head=v1_config["n_head"],
            n_layer=v1_config["n_layer"],
            block_size=v1_config["block_size"],
            dropout=v1_config.get(
                "dropout",
                0.1,
            ),
        )

        v1_params = v1_model.num_params()

        print(
            f"V1 parameters: "
            f"{v1_params:,}"
        )

        if num_params:

            print(
                f"V3 parameters: "
                f"{num_params:,}"
            )

            print()

            multiplier = (
                num_params / v1_params
            )

            print(
                f"V3 / V1 size: "
                f"{multiplier:.2f}×"
            )

        print()

        print(
            "⚠️ V3 is intended to train "
            "FROM SCRATCH."
        )

        print(
            "The new tokenizer means V1 "
            "weights are not reused."
        )

    except Exception as error:

        print(
            f"⚠️ Could not inspect V1:"
        )

        print(
            f"   {error}"
        )

print()


# ============================================================
# FINAL STATUS
# ============================================================

line()

print("FINAL STATUS")

line("-")

errors = []

warnings = []


if architecture_errors:

    errors.extend(
        architecture_errors
    )


if not os.path.exists(DATASET_PATH):

    errors.append(
        "Dataset does not exist."
    )


if vocab_size <= 0:

    errors.append(
        "No valid tokenizer vocabulary found."
    )


if tokenizer_type == "CHARACTER":

    warnings.append(
        "V3 is still using a character tokenizer."
    )


if tokenizer_type == "UNKNOWN":

    warnings.append(
        "Tokenizer type could not be determined."
    )


if dataset_chars and num_params:

    ratio = (
        dataset_chars / num_params
    )

    if ratio < 0.1:

        warnings.append(
            "Dataset is very small relative "
            "to model size."
        )


if errors:

    print()

    for error in errors:

        print(
            f"❌ BLOCKING ERROR: {error}"
        )

    print()

    print(
        "🚫 DO NOT START TRAINING."
    )

else:

    print()

    print(
        "✅ NO BLOCKING ERRORS"
    )

    if warnings:

        print()

        print("WARNINGS:")

        for warning in warnings:

            print(
                f"⚠️ {warning}"
            )

    else:

        print(
            "✅ No warnings."
        )

    print()

    print(
        "V3 TRAINING MODE:"
    )

    print(
        "FROM SCRATCH"
    )

    print()

    print(
        "Start training with:"
    )

    print()

    print(
        'python train.py --name "OrangeGlasses"'
    )

line()

print()
print("🍊 ORANGEGLASSES V3 CONFIG CHECK COMPLETE")
print()