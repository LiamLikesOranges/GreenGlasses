"""
Prepares training data for greenglasses.

By default, downloads the "Tiny Shakespeare" corpus (~1MB of plain text,
a classic small dataset for exactly this kind of experiment). If you
already have your own input.txt in this folder, pass --skip-download to
use it as-is instead.

If the download fails for any reason (no internet, firewall, corporate
proxy, etc.), this falls back to a small built-in sample text so the
pipeline always has *something* to train on rather than silently failing.
"""

import argparse
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(THIS_DIR, "input.txt")

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

# Used only if the download fails. Small on purpose — just enough to prove
# the pipeline works end to end. Swap in your own data/input.txt for
# anything real.
FALLBACK_TEXT = """ROMEO:
But, soft! what light through yonder window breaks?
It is the east, and Juliet is the sun.
Arise, fair sun, and kill the envious moon,
Who is already sick and pale with grief,
That thou her maid art far more fair than she.

JULIET:
O Romeo, Romeo! wherefore art thou Romeo?
Deny thy father and refuse thy name;
Or, if thou wilt not, be but sworn my love,
And I'll no longer be a Capulet.

ROMEO:
Shall I hear more, or shall I speak at this?

JULIET:
'Tis but thy name that is my enemy;
Thou art thyself, though not a Montague.
What's Montague? it is nor hand, nor foot,
Nor arm, nor face, nor any other part
Belonging to a man. O, be some other name!
What's in a name? that which we call a rose
By any other name would smell as sweet.
""" * 20  # repeated so there's enough text to form train/val splits


def download_default_corpus():
    import requests

    print(f"Downloading default corpus from {TINY_SHAKESPEARE_URL} ...")
    resp = requests.get(TINY_SHAKESPEARE_URL, timeout=15)
    resp.raise_for_status()
    if len(resp.text) < 1000:
        raise ValueError(f"Downloaded file looks too small ({len(resp.text)} chars) — treating as failed.")
    with open(INPUT_PATH, "w", encoding="utf-8") as f:
        f.write(resp.text)
    print(f"Saved {len(resp.text):,} characters to {INPUT_PATH}")


def write_fallback_corpus():
    print("Falling back to a small built-in sample text instead (no internet needed).")
    with open(INPUT_PATH, "w", encoding="utf-8") as f:
        f.write(FALLBACK_TEXT)
    print(f"Saved {len(FALLBACK_TEXT):,} characters to {INPUT_PATH}")
    print("(This is a tiny placeholder — swap in your own data/input.txt for real training.)")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for greenglasses")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Don't download anything; use the existing data/input.txt as-is.",
    )
    args = parser.parse_args()

    if args.skip_download:
        if not os.path.exists(INPUT_PATH):
            raise FileNotFoundError(
                f"--skip-download was passed but {INPUT_PATH} doesn't exist. "
                "Put your own text file there first."
            )
        print(f"Using existing file at {INPUT_PATH}")
    else:
        try:
            download_default_corpus()
        except Exception as e:
            print(f"\n[prepare] Download failed: {e}")
            write_fallback_corpus()

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Something went wrong — {INPUT_PATH} still doesn't exist after prepare step."
        )

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    vocab = sorted(set(text))
    print(f"\nDataset stats:")
    print(f"  file: {INPUT_PATH}")
    print(f"  total characters: {len(text):,}")
    print(f"  unique characters (vocab size): {len(vocab)}")
    print(f"  vocab: {''.join(vocab)!r}")
    print("\nReady. Next step: python train.py")


if __name__ == "__main__":
    main()