"""
OrangeGlasses V4 Byte-Level BPE Tokenizer

Supports:
    V3 byte-level BPE tokenizers (version 1)
    V4 byte-level BPE tokenizers (version 2)

V3:
    - 256 base byte tokens
    - learned BPE merges
    - no special tokens

V4:
    - 256 base byte tokens
    - learned BPE merges
    - special tokens:
        <PAD>
        <UNK>
        <BOS>
        <EOS>

Usage:
    python tokenizer.py --train data\input.txt --vocab-size 4096
"""

import argparse
import json
import os
from collections import Counter


# ============================================================
# CONSTANTS
# ============================================================

BASE_VOCAB_SIZE = 256

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

SPECIAL_TOKENS = [
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
]


# ============================================================
# BPE TOKENIZER
# ============================================================

class BPETokenizer:

    def __init__(
        self,
        vocab,
        merges,
        special_tokens=None,
    ):

        self.vocab = {
            int(k): bytes(v)
            for k, v in vocab.items()
        }

        self.merges = [
            (
                int(a),
                int(b),
            )
            for a, b in merges
        ]

        self.vocab_size = len(
            self.vocab
        )

        # ----------------------------------------------------
        # Reverse lookup
        # ----------------------------------------------------

        self.bytes_to_id = {
            value: token_id
            for token_id, value
            in self.vocab.items()
        }

        # ----------------------------------------------------
        # Merge ranks
        # ----------------------------------------------------

        self.merge_ranks = {
            pair: rank
            for rank, pair
            in enumerate(self.merges)
        }

        # ----------------------------------------------------
        # Special tokens
        # ----------------------------------------------------

        if special_tokens is None:
            special_tokens = {}

        self.special_tokens = {
            str(token): int(token_id)
            for token, token_id
            in special_tokens.items()
        }

        self.special_ids = {
            token_id: token
            for token, token_id
            in self.special_tokens.items()
        }

        self.pad_id = self.special_tokens.get(
            PAD_TOKEN
        )

        self.unk_id = self.special_tokens.get(
            UNK_TOKEN
        )

        self.bos_id = self.special_tokens.get(
            BOS_TOKEN
        )

        self.eos_id = self.special_tokens.get(
            EOS_TOKEN
        )

        # Longest first.
        self._special_strings = sorted(
            self.special_tokens.keys(),
            key=len,
            reverse=True,
        )

    # ========================================================
    # BPE TRAINING
    # ========================================================

    @classmethod
    def train(
        cls,
        text,
        vocab_size=4096,
        min_pair_frequency=2,
    ):

        if vocab_size <= (
            BASE_VOCAB_SIZE
            + len(SPECIAL_TOKENS)
        ):
            raise ValueError(
                "vocab_size must be larger than "
                "the 256 byte tokens plus the "
                "four special tokens."
            )

        raw = text.encode(
            "utf-8"
        )

        tokens = list(raw)

        learned_vocab_size = (
            vocab_size
            - len(SPECIAL_TOKENS)
        )

        vocab = {
            i: bytes([i])
            for i in range(
                BASE_VOCAB_SIZE
            )
        }

        merges = []

        print("=" * 60)
        print(
            "ORANGEGLASSES V4 BPE "
            "TOKENIZER TRAINING"
        )
        print("=" * 60)

        print(
            f"Training characters: {len(text):,}"
        )

        print(
            f"Training bytes:      {len(raw):,}"
        )

        print(
            f"Target total vocab:  {vocab_size:,}"
        )

        print(
            f"Learned BPE vocab:   "
            f"{learned_vocab_size:,}"
        )

        print(
            f"Base byte vocab:     "
            f"{BASE_VOCAB_SIZE}"
        )

        print()

        if not tokens:
            raise ValueError(
                "Training text is empty."
            )

        while len(vocab) < learned_vocab_size:

            pair_counts = Counter(
                zip(
                    tokens,
                    tokens[1:],
                )
            )

            if not pair_counts:
                break

            pair, frequency = (
                pair_counts.most_common(1)[0]
            )

            if frequency < min_pair_frequency:

                print(
                    "Stopping: highest pair "
                    "frequency fell below "
                    f"{min_pair_frequency}."
                )

                break

            new_id = len(vocab)

            left, right = pair

            vocab[new_id] = (
                vocab[left]
                + vocab[right]
            )

            merges.append(pair)

            new_tokens = []

            i = 0

            while i < len(tokens):

                if (
                    i + 1 < len(tokens)
                    and tokens[i] == left
                    and tokens[i + 1] == right
                ):

                    new_tokens.append(
                        new_id
                    )

                    i += 2

                else:

                    new_tokens.append(
                        tokens[i]
                    )

                    i += 1

            tokens = new_tokens

            if (
                len(vocab)
                <= BASE_VOCAB_SIZE + 10
                or len(vocab) % 100 == 0
            ):

                print(
                    f"Vocabulary: "
                    f"{len(vocab):5d}/"
                    f"{learned_vocab_size} | "
                    f"pair frequency: "
                    f"{frequency:6d} | "
                    f"tokens remaining: "
                    f"{len(tokens):,}",
                    flush=True,
                )

        # ----------------------------------------------------
        # V4 special tokens
        # ----------------------------------------------------

        special_tokens = {}

        for token in SPECIAL_TOKENS:

            special_tokens[token] = (
                len(vocab)
            )

        print()

        print(
            f"Final BPE vocabulary: "
            f"{len(vocab):,}"
        )

        for (
            token,
            token_id,
        ) in special_tokens.items():

            print(
                f"  {token}: {token_id}"
            )

        print(
            f"Total vocabulary: "
            f"{len(vocab) + len(special_tokens):,}"
        )

        print(
            f"Learned merges: "
            f"{len(merges):,}"
        )

        print("=" * 60)

        return cls(
            vocab,
            merges,
            special_tokens=special_tokens,
        )

    # ========================================================
    # BPE MERGING
    # ========================================================

    def _merge_tokens(
        self,
        tokens,
    ):

        if len(tokens) < 2:
            return tokens

        while True:

            best_pair = None
            best_rank = None

            for pair in zip(
                tokens,
                tokens[1:],
            ):

                rank = self.merge_ranks.get(
                    pair
                )

                if rank is not None:

                    if (
                        best_rank is None
                        or rank < best_rank
                    ):

                        best_rank = rank
                        best_pair = pair

            if best_pair is None:
                break

            left, right = best_pair

            merged_bytes = (
                self.vocab[left]
                + self.vocab[right]
            )

            merged_id = (
                self.bytes_to_id.get(
                    merged_bytes
                )
            )

            if merged_id is None:

                raise RuntimeError(
                    "BPE merge produced a "
                    "token that does not exist "
                    "in the vocabulary."
                )

            new_tokens = []

            i = 0

            while i < len(tokens):

                if (
                    i + 1 < len(tokens)
                    and tokens[i] == left
                    and tokens[i + 1] == right
                ):

                    new_tokens.append(
                        merged_id
                    )

                    i += 2

                else:

                    new_tokens.append(
                        tokens[i]
                    )

                    i += 1

            tokens = new_tokens

        return tokens

    # ========================================================
    # ENCODING
    # ========================================================

    def _encode_plain_text(
        self,
        text,
    ):

        raw = text.encode(
            "utf-8"
        )

        tokens = list(raw)

        return self._merge_tokens(
            tokens
        )

    def encode(
        self,
        text,
        add_bos=False,
        add_eos=False,
    ):

        output = []

        position = 0

        while position < len(text):

            found_special = None

            for special in (
                self._special_strings
            ):

                if text.startswith(
                    special,
                    position,
                ):

                    found_special = (
                        special
                    )

                    break

            if found_special is not None:

                output.append(
                    self.special_tokens[
                        found_special
                    ]
                )

                position += len(
                    found_special
                )

                continue

            next_special = len(text)

            for special in (
                self._special_strings
            ):

                index = text.find(
                    special,
                    position,
                )

                if index != -1:

                    next_special = min(
                        next_special,
                        index,
                    )

            chunk = text[
                position:next_special
            ]

            if chunk:

                output.extend(
                    self._encode_plain_text(
                        chunk
                    )
                )

            position = next_special

        if add_bos:

            if self.bos_id is None:

                raise RuntimeError(
                    "Tokenizer has no BOS token."
                )

            output.insert(
                0,
                self.bos_id,
            )

        if add_eos:

            if self.eos_id is None:

                raise RuntimeError(
                    "Tokenizer has no EOS token."
                )

            output.append(
                self.eos_id
            )

        return output

    # ========================================================
    # DECODING
    # ========================================================

    def decode(
        self,
        ids,
        skip_special_tokens=True,
    ):

        output = bytearray()

        for token_id in ids:

            token_id = int(
                token_id
            )

            # ------------------------------------------------
            # Special token
            # ------------------------------------------------

            if token_id in self.special_ids:

                if skip_special_tokens:
                    continue

                token = self.special_ids[
                    token_id
                ]

                output.extend(
                    token.encode(
                        "utf-8"
                    )
                )

                continue

            # ------------------------------------------------
            # Normal token
            # ------------------------------------------------

            if token_id not in self.vocab:

                raise ValueError(
                    f"Unknown token ID: "
                    f"{token_id}"
                )

            output.extend(
                self.vocab[token_id]
            )

        return bytes(
            output
        ).decode(
            "utf-8",
            errors="replace",
        )

    # ========================================================
    # SAVE
    # ========================================================

    def save(
        self,
        path,
    ):

        os.makedirs(
            os.path.dirname(
                os.path.abspath(path)
            ),
            exist_ok=True,
        )

        data = {
            "type": "byte_bpe",

            "version": 2,

            "vocab_size": (
                self.vocab_size
                + len(
                    self.special_tokens
                )
            ),

            "vocab": {
                str(token_id): list(
                    token_bytes
                )

                for (
                    token_id,
                    token_bytes
                )
                in self.vocab.items()
            },

            "merges": [
                [
                    left,
                    right,
                ]

                for (
                    left,
                    right
                )
                in self.merges
            ],

            "special_tokens":
                self.special_tokens,
        }

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )

        print(
            f"Tokenizer saved to: {path}"
        )

    # ========================================================
    # LOAD
    # ========================================================

    @classmethod
    def load(
        cls,
        path,
    ):

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(
                file
            )

        # ----------------------------------------------------
        # Check tokenizer type
        # ----------------------------------------------------

        if data.get("type") != "byte_bpe":

            raise ValueError(
                "Tokenizer file is not "
                "a byte-BPE tokenizer."
            )

        version = int(
            data.get(
                "version",
                1,
            )
        )

        # ----------------------------------------------------
        # V3 + V4 compatibility
        # ----------------------------------------------------

        if version not in (
            1,
            2,
        ):

            raise ValueError(
                "Unsupported byte-BPE "
                f"tokenizer version: {version}"
            )

        # ----------------------------------------------------
        # Load vocabulary
        # ----------------------------------------------------

        vocab = {
            int(token_id): bytes(
                token_bytes
            )

            for (
                token_id,
                token_bytes
            )
            in data["vocab"].items()
        }

        # ----------------------------------------------------
        # Load merges
        # ----------------------------------------------------

        merges = [
            (
                int(pair[0]),
                int(pair[1]),
            )

            for pair in data.get(
                "merges",
                [],
            )
        ]

        # ----------------------------------------------------
        # V3 has no special tokens.
        #
        # V4 stores them.
        # ----------------------------------------------------

        special_tokens = {
            str(token): int(token_id)

            for (
                token,
                token_id
            )
            in data.get(
                "special_tokens",
                {},
            ).items()
        }

        tokenizer = cls(
            vocab,
            merges,
            special_tokens=special_tokens,
        )

        print(
            f"Loaded byte-BPE tokenizer "
            f"version {version} "
            f"with "
            f"{len(vocab):,} tokens."
        )

        if version == 1:

            print(
                "Tokenizer format: "
                "OrangeGlasses V3"
            )

        elif version == 2:

            print(
                "Tokenizer format: "
                "OrangeGlasses V4"
            )

        return tokenizer

    # ========================================================
    # INFO
    # ========================================================

    def info(
        self,
    ):

        return {
            "type": "byte_bpe",

            "version": 2,

            "vocab_size": (
                self.vocab_size
                + len(
                    self.special_tokens
                )
            ),

            "bpe_vocab_size":
                self.vocab_size,

            "base_vocab_size":
                BASE_VOCAB_SIZE,

            "merge_count":
                len(self.merges),

            "special_tokens":
                self.special_tokens,

            "pad_id":
                self.pad_id,

            "unk_id":
                self.unk_id,

            "bos_id":
                self.bos_id,

            "eos_id":
                self.eos_id,
        }


# ============================================================
# COMPATIBILITY HELPER
# ============================================================

def tokenizer_path_for(
    checkpoint_dir,
):

    return os.path.join(
        checkpoint_dir,
        "tokenizer.json",
    )


# ============================================================
# TRAIN COMMAND
# ============================================================

def train_tokenizer(
    input_path,
    output_path,
    vocab_size,
):

    print("=" * 60)
    print(
        "ORANGEGLASSES V4 TOKENIZER"
    )
    print("=" * 60)

    print(
        f"Input:       {input_path}"
    )

    print(
        f"Output:      {output_path}"
    )

    print(
        f"Vocabulary:  {vocab_size:,}"
    )

    print()

    if not os.path.exists(
        input_path
    ):

        raise FileNotFoundError(
            f"Training file not found: "
            f"{input_path}"
        )

    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as file:

        text = file.read()

    if not text:

        raise ValueError(
            "Training text is empty."
        )

    tokenizer = BPETokenizer.train(
        text,
        vocab_size=vocab_size,
    )

    tokenizer.save(
        output_path
    )

    print()
    print(
        "Testing tokenizer..."
    )

    test_texts = [
        "Hello, world!",
        "What is the meaning of life?",
        "OrangeGlasses V4 🍊",
        "Unicode test: 🤣 Ω é ñ",
        "Hello<EOS>World",
    ]

    for test_text in test_texts:

        ids = tokenizer.encode(
            test_text
        )

        decoded = tokenizer.decode(
            ids,
            skip_special_tokens=False,
        )

        print()

        print(
            f"Input:   {test_text}"
        )

        print(
            f"Tokens:  {len(ids)}"
        )

        print(
            f"IDs:     {ids[:30]}"
        )

        print(
            f"Decoded: {decoded}"
        )

        if decoded != test_text:

            raise RuntimeError(
                "Tokenizer round-trip "
                "failed!\n"
                f"Original: {test_text!r}\n"
                f"Decoded:  {decoded!r}"
            )

    print()
    print("=" * 60)
    print(
        "TOKENIZER READY 🍊"
    )
    print("=" * 60)

    info = tokenizer.info()

    print()

    print(
        f"Type:       {info['type']}"
    )

    print(
        f"Version:    {info['version']}"
    )

    print(
        f"Vocabulary: "
        f"{info['vocab_size']:,}"
    )

    print(
        f"BPE vocab:  "
        f"{info['bpe_vocab_size']:,}"
    )

    print(
        f"Merges:     "
        f"{info['merge_count']:,}"
    )

    print()

    print(
        "Special tokens:"
    )

    for (
        token,
        token_id
    ) in info[
        "special_tokens"
    ].items():

        print(
            f"  {token:5s} -> "
            f"{token_id}"
        )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train the OrangeGlasses V4 "
            "byte-level BPE tokenizer."
        )
    )

    parser.add_argument(
        "--train",
        default=os.path.join(
            "data",
            "input.txt",
        ),
        help=(
            "Training text file "
            "(default: data/input.txt)"
        ),
    )

    parser.add_argument(
        "--output",
        default=os.path.join(
            "data",
            "tokenizer.json",
        ),
        help=(
            "Output tokenizer path "
            "(default: data/tokenizer.json)"
        ),
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=4096,
        help=(
            "Target total vocabulary size "
            "(default: 4096)"
        ),
    )

    args = parser.parse_args()

    train_tokenizer(
        input_path=args.train,
        output_path=args.output,
        vocab_size=args.vocab_size,
    )


if __name__ == "__main__":
    main()