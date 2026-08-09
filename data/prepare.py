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
    """Downloads with a hard, unignorable timeout, streaming in chunks so
    real progress can be reported (prints DOWNLOAD_PROGRESS:<percent>
    lines that gui.py parses into an actual progress bar).

    Some networks (corporate proxies, firewalls, VPNs) can cause a
    connection to hang at the OS level in a way Python can't interrupt
    from the outside. To guarantee this function never hangs forever, the
    actual request runs in a daemon thread: we wait up to
    HARD_TIMEOUT_SECONDS for it, and if it's still stuck, we give up and
    move on — the daemon thread gets silently abandoned (and cleaned up
    whenever the process eventually exits) instead of blocking this
    script."""
    import threading
    import requests

    HARD_TIMEOUT_SECONDS = 20
    result = {}

    def _worker():
        try:
            # (connect_timeout, read_timeout) — separate so a slow-but-alive
            # connection can't quietly stall past a single combined timeout.
            resp = requests.get(TINY_SHAKESPEARE_URL, timeout=(5, 8), stream=True)
            resp.raise_for_status()

            total = resp.headers.get("Content-Length")
            total = int(total) if total is not None else None

            # Read raw (still-compressed) bytes rather than
            # resp.iter_content()'s auto-decompressed bytes. GitHub serves
            # this file gzip-compressed, so Content-Length reflects the
            # *compressed* size — tracking decompressed bytes against that
            # header would blow past 100% before the download is actually
            # done. decode_content=False keeps what we count in sync with
            # what Content-Length promised.
            chunks = []
            downloaded = 0
            last_reported_percent = -1

            for chunk in resp.raw.stream(8192, decode_content=False):
                if not chunk:
                    continue
                chunks.append(chunk)
                downloaded += len(chunk)

                if total:
                    percent = min(100.0, downloaded / total * 100)
                    # only print when the whole percent changes, so we
                    # don't flood the log with hundreds of lines
                    if int(percent) != last_reported_percent:
                        last_reported_percent = int(percent)
                        print(f"DOWNLOAD_PROGRESS:{percent:.1f}", flush=True)
                else:
                    # server didn't send a size — report bytes instead so
                    # there's still visible movement, just not a percent
                    print(f"DOWNLOAD_BYTES:{downloaded}", flush=True)

            raw_bytes = b"".join(chunks)
            encoding = (resp.headers.get("Content-Encoding") or "").lower()
            if encoding == "gzip":
                import gzip
                raw_bytes = gzip.decompress(raw_bytes)
            elif encoding == "deflate":
                import zlib
                raw_bytes = zlib.decompress(raw_bytes)
            # (anything else — identity/br/unsupported — is used as-is;
            # br is rare for plain-text GitHub responses)

            result["text"] = raw_bytes.decode("utf-8", errors="replace")
        except Exception as e:
            result["error"] = e

    print(f"Downloading default corpus from {TINY_SHAKESPEARE_URL} ...", flush=True)
    print(f"(will give up after {HARD_TIMEOUT_SECONDS}s and use a built-in sample instead)", flush=True)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=HARD_TIMEOUT_SECONDS)

    if thread.is_alive():
        # Still stuck. We can't force-kill a thread in Python, but because
        # it's a daemon thread it won't stop this script from continuing
        # or from exiting later — we just stop waiting on it here.
        raise TimeoutError(
            f"Download did not respond within {HARD_TIMEOUT_SECONDS}s "
            "(likely blocked by a firewall/proxy)."
        )

    if "error" in result:
        raise result["error"]

    text = result["text"]
    if len(text) < 1000:
        raise ValueError(f"Downloaded file looks too small ({len(text)} chars) — treating as failed.")

    with open(INPUT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print("DOWNLOAD_PROGRESS:100.0", flush=True)
    print(f"Saved {len(text):,} characters to {INPUT_PATH}", flush=True)


def write_fallback_corpus():
    print("Falling back to a small built-in sample text instead (no internet needed).", flush=True)
    with open(INPUT_PATH, "w", encoding="utf-8") as f:
        f.write(FALLBACK_TEXT)
    print(f"Saved {len(FALLBACK_TEXT):,} characters to {INPUT_PATH}", flush=True)
    print("(This is a tiny placeholder — swap in your own data/input.txt for real training.)", flush=True)


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
        print(f"Using existing file at {INPUT_PATH}", flush=True)
    else:
        try:
            download_default_corpus()
        except Exception as e:
            print(f"\n[prepare] Download failed: {e}", flush=True)
            write_fallback_corpus()

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Something went wrong — {INPUT_PATH} still doesn't exist after prepare step."
        )

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    vocab = sorted(set(text))
    print(f"\nDataset stats:", flush=True)
    print(f"  file: {INPUT_PATH}", flush=True)
    print(f"  total characters: {len(text):,}", flush=True)
    print(f"  unique characters (vocab size): {len(vocab)}", flush=True)
    print(f"  vocab: {''.join(vocab)!r}", flush=True)
    print("\nReady. Next step: python train.py", flush=True)


if __name__ == "__main__":
    main()