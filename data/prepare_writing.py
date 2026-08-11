import os
import random
from datasets import load_dataset

OUTPUT_PATH = os.path.join("data", "writing.txt")
MAX_CHARS = 8_000_000
SEED = 18

random.seed(SEED)

OUTPUT = []


def clean_text(text):
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\r", "")
    text = text.replace("\x00", "")

    lines = []

    for line in text.split("\n"):
        line = line.strip()

        if line:
            lines.append(line)

    return "\n".join(lines)


def add_text(text):
    text = clean_text(text)

    if len(text) < 30:
        return

    OUTPUT.append(
        text + "\n\n"
    )


def load_wikipedia():
    print("[1/4] Loading Wikipedia...")

    try:
        dataset = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split="train",
            streaming=True,
        )

        count = 0

        for item in dataset:
            text = item.get("text")

            if text:
                add_text(text)
                count += 1

            if sum(len(x) for x in OUTPUT) >= MAX_CHARS:
                break

            if count % 5000 == 0:
                print(
                    f"  Added {count:,} articles",
                    flush=True,
                )

        print(
            f"  Wikipedia articles: {count:,}"
        )

    except Exception as error:
        print(
            f"  Wikipedia failed: {error}"
        )


def load_writing_prompts():
    print("[2/4] Loading writing dataset...")

    try:
        dataset = load_dataset(
            "fancyzhx/yelp_polarity",
            split="train",
            streaming=True,
        )

        count = 0

        for item in dataset:
            text = item.get("text")

            if text:
                add_text(text)
                count += 1

            if sum(len(x) for x in OUTPUT) >= MAX_CHARS:
                break

            if count >= 50000:
                break

        print(
            f"  Added {count:,} long-form examples."
        )

    except Exception as error:
        print(
            f"  Writing dataset failed: {error}"
        )


def load_books():
    print("[3/4] Loading public-domain books...")

    try:
        dataset = load_dataset(
            "pg19",
            split="train",
            streaming=True,
        )

        count = 0

        for item in dataset:
            text = item.get("text")

            if text:
                add_text(text)
                count += 1

            if sum(len(x) for x in OUTPUT) >= MAX_CHARS:
                break

            if count >= 3000:
                break

        print(
            f"  Added {count:,} books."
        )

    except Exception as error:
        print(
            f"  PG19 failed: {error}"
        )


def load_news():
    print("[4/4] Loading news-style writing...")

    try:
        dataset = load_dataset(
            "ccdv/govreport-summarization",
            split="train",
            streaming=True,
        )

        count = 0

        for item in dataset:
            text = item.get("report")

            if text:
                add_text(text)
                count += 1

            if sum(len(x) for x in OUTPUT) >= MAX_CHARS:
                break

            if count >= 10000:
                break

        print(
            f"  Added {count:,} reports."
        )

    except Exception as error:
        print(
            f"  Reports failed: {error}"
        )


def main():
    print("=" * 60)
    print("ORANGEGLASSES V4 WRITING DATASET")
    print("=" * 60)

    load_wikipedia()
    load_writing_prompts()
    load_books()
    load_news()

    print()
    print("Combining writing datasets...")

    random.shuffle(OUTPUT)

    final_text = ""
    count = 0

    for text in OUTPUT:
        if len(final_text) + len(text) > MAX_CHARS:
            break

        final_text += text
        count += 1

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(final_text)

    print()
    print("=" * 60)
    print("WRITING DATASET READY")
    print("=" * 60)
    print(f"Output:     {OUTPUT_PATH}")
    print(f"Characters: {len(final_text):,}")
    print(f"Documents:  {count:,}")


if __name__ == "__main__":
    main()