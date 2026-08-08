"""
A tokenizer converts text <-> numbers, since neural networks only work with
numbers. Real LLMs use subword tokenizers (byte-pair encoding) so common
chunks like "ing" or "the" become single tokens. Here, we keep it as simple
as possible: every unique character in the training text is its own token.

This means the model has to learn spelling from scratch, which is part of
why character-level models need to see a lot of text before they produce
recognizable words — but it also means there's no separate tokenizer to
train, and the vocabulary is tiny (usually under 100 characters).
"""

import json
import os


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str):
        """String -> list of integer token ids."""
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        """List of integer token ids -> string."""
        return "".join(self.itos[i] for i in ids)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        obj = cls.__new__(cls)
        obj.stoi = data["stoi"]
        obj.itos = {int(v): k for k, v in obj.stoi.items()}
        obj.vocab_size = len(obj.stoi)
        return obj

    @classmethod
    def from_file(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(text)


def tokenizer_path_for(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, "tokenizer.json")
