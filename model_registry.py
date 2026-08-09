"""
model_registry.py — shared logic for how trained models are named, saved,
and discovered under models/.

Layout: every training run gets its own folder under models/, e.g.

    models/
      GreenGlasses-v1-0.8M/
        model.pt          the trained weights + architecture config
        tokenizer.json      the character tokenizer for this model
        info.json             human-readable metadata (see save_model_info)

Naming: "{name}-v{version}-{size}", e.g. "GreenGlasses-v1-0.8M".
  - name: a label you choose (train.py --name), default "GreenGlasses"
  - version: auto-incrementing per name (v1, v2, v3, ...) so retraining
    never overwrites a previous run — each one gets its own folder
  - size: parameter count, auto-formatted as K/M/B (827,000 -> "0.8M")

Windows can't have ":" in folder names, which is why this uses
"name-vN-size" with dashes instead of the "name:tag" style you might have
seen elsewhere (e.g. Ollama) — same idea, filesystem-safe spelling.
"""

import json
import os
import re
from datetime import datetime, timezone


def format_param_count(n):
    """827136 -> '0.8M', 1_500_000_000 -> '1.5B', 3200 -> '3.2K'."""
    if n is None:
        return "?"
    n = float(n)
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(int(n))


_FOLDER_RE = re.compile(r"^(?P<name>.+)-v(?P<version>\d+)-(?P<size>[\d.]+[KMB]?)$")


def parse_folder_name(folder_name):
    """Splits 'GreenGlasses-v1-0.8M' back into its parts, or None if it
    doesn't match the expected pattern (e.g. a folder a user renamed)."""
    match = _FOLDER_RE.match(folder_name)
    if not match:
        return None
    return {
        "name": match.group("name"),
        "version": int(match.group("version")),
        "size": match.group("size"),
    }


def next_version(models_dir, name):
    """Scans models_dir for existing '{name}-vN-...' folders and returns
    the next version number to use (1 if there are none yet)."""
    if not os.path.exists(models_dir):
        return 1
    highest = 0
    for folder_name in os.listdir(models_dir):
        parsed = parse_folder_name(folder_name)
        if parsed and parsed["name"] == name:
            highest = max(highest, parsed["version"])
    return highest + 1


def make_model_folder_name(name, version, num_params):
    size = format_param_count(num_params)
    return f"{name}-v{version}-{size}"


def save_model_info(folder_path, *, name, version, num_params, iteration,
                     max_iters, model_config, description=None,
                     dataset_chars=None, created_at=None):
    """Writes/updates info.json in a model's folder. Called both when a
    model folder is first created and again on every later checkpoint
    save, so 'iteration' and 'updated_at' stay current during training."""
    info_path = os.path.join(folder_path, "info.json")

    if created_at is None:
        # preserve the original creation time across repeated saves
        if os.path.exists(info_path):
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    created_at = json.load(f).get("created_at")
            except Exception:
                created_at = None
        if created_at is None:
            created_at = datetime.now(timezone.utc).isoformat()

    if description is None:
        if dataset_chars:
            description = (
                f"Character-level GPT trained on {dataset_chars:,} characters of text."
            )
        else:
            description = "Character-level GPT trained from scratch."

    info = {
        "name": name,
        "version": version,
        "folder": os.path.basename(folder_path),
        "size_tag": format_param_count(num_params),
        "num_params": num_params,
        "iteration": iteration,
        "max_iters": max_iters,
        "config": model_config,
        "description": description,
        "created_at": created_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    os.makedirs(folder_path, exist_ok=True)
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return info


def read_model_info(folder_path):
    info_path = os.path.join(folder_path, "info.json")
    if not os.path.exists(info_path):
        return None
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def list_model_folders(models_dir):
    """Returns folder names under models_dir that actually contain a
    trained model (a model.pt file), sorted by last-updated (newest
    first) when info.json is available, else alphabetically."""
    if not os.path.exists(models_dir):
        return []

    entries = []
    for folder_name in os.listdir(models_dir):
        folder_path = os.path.join(models_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        if not os.path.exists(os.path.join(folder_path, "model.pt")):
            continue
        info = read_model_info(folder_path)
        updated_at = info.get("updated_at") if info else None
        entries.append((updated_at or "", folder_name))

    entries.sort(reverse=True)
    return [name for _, name in entries]


def find_latest_model_folder(models_dir):
    folders = list_model_folders(models_dir)
    return folders[0] if folders else None


def migrate_legacy_checkpoint(legacy_checkpoint_path, legacy_tokenizer_path,
                               models_dir, default_name="GreenGlasses"):
    """One-time upgrade path: if an old-style single checkpoints/*.pt
    setup exists and models/ doesn't have anything yet, copy it over into
    the new per-model-folder layout so it doesn't just disappear.
    Returns the new folder name if a migration happened, else None."""
    import shutil
    import torch

    if not os.path.exists(legacy_checkpoint_path):
        return None
    if list_model_folders(models_dir):
        return None  # already have real models; don't touch anything

    try:
        checkpoint = torch.load(legacy_checkpoint_path, map_location="cpu")
    except Exception:
        return None

    model_config = checkpoint.get("config", {})
    num_params = None
    try:
        from model import GreenGlassesGPT
        m = GreenGlassesGPT(
            vocab_size=model_config["vocab_size"],
            n_embed=model_config["n_embed"],
            n_head=model_config["n_head"],
            n_layer=model_config["n_layer"],
            block_size=model_config["block_size"],
            dropout=model_config["dropout"],
        )
        num_params = m.num_params()
    except Exception:
        pass

    version = next_version(models_dir, default_name)
    folder_name = make_model_folder_name(default_name, version, num_params)
    folder_path = os.path.join(models_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    shutil.copy2(legacy_checkpoint_path, os.path.join(folder_path, "model.pt"))
    if os.path.exists(legacy_tokenizer_path):
        shutil.copy2(legacy_tokenizer_path, os.path.join(folder_path, "tokenizer.json"))

    save_model_info(
        folder_path,
        name=default_name,
        version=version,
        num_params=num_params,
        iteration=checkpoint.get("iter"),
        max_iters=None,
        model_config=model_config,
        description="Migrated automatically from an earlier version of greenglasses "
                     "that stored a single checkpoint instead of per-model folders.",
    )
    return folder_name