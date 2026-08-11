import os
import random
from datasets import load_dataset

OUTPUT_PATH = os.path.join("data", "facts.txt")
MAX_CHARS = 10_000_000
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


def add_fact(text):
    text = clean_text(text)

    if len(text) < 40:
        return

    OUTPUT.append(
        text + "\n\n"
    )


def load_wikipedia():
    print("[1/5] Loading Wikipedia knowledge...")

    try:
        dataset = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split="train",
            streaming=True,
        )

        count = 0

        for item in dataset:
            title = item.get("title")
            text = item.get("text")

            if text:
                if title:
                    add_fact(
                        f"{title}\n{text}"
                    )
                else:
                    add_fact(text)

                count += 1

            if count % 5000 == 0:
                print(
                    f"  Added {count:,} articles",
                    flush=True,
                )

            if sum(len(x) for x in OUTPUT) >= MAX_CHARS:
                break

        print(
            f"  Wikipedia: {count:,} articles"
        )

    except Exception as error:
        print(
            f"  Wikipedia failed: {error}"
        )


def load_openwebmath():
    print("[2/5] Loading educational knowledge...")

    try:
        dataset = load_dataset(
            "open-web-math/open-web-math",
            split="train",
            streaming=True,
        )

        count = 0

        for item in dataset:
            text = item.get("text")

            if text:
                add_fact(text)
                count += 1

            if count >= 50000:
                break

            if sum(len(x) for x in OUTPUT) >= MAX_CHARS:
                break

        print(
            f"  Educational documents: {count:,}"
        )

    except Exception as error:
        print(
            f"  Educational dataset failed: {error}"
        )


def load_science():
    print("[3/5] Loading science knowledge...")

    try:
        dataset = load_dataset(
            "allenai/scitldr",
            split="train",
            streaming=True,
        )

        count = 0

        for item in dataset:
            source = item.get("source")

            if source:
                if isinstance(source, list):
                    source = " ".join(
                        str(x)
                        for x in source
                    )

                add_fact(source)
                count += 1

            if count >= 20000:
                break

            if sum(len(x) for x in OUTPUT) >= MAX_CHARS:
                break

        print(
            f"  Science documents: {count:,}"
        )

    except Exception as error:
        print(
            f"  Science dataset failed: {error}"
        )


def load_qa():
    print("[4/5] Loading factual Q&A...")

    try:
        dataset = load_dataset(
            "openbookqa",
            "main",
            split="train",
            streaming=True,
        )

        count = 0

        for item in dataset:
            question = item.get("question")
            choices = item.get("choices")
            answer = item.get("answerKey")

            if not question:
                continue

            text = f"Question: {question}"

            if choices:
                labels = choices.get("label", [])
                values = choices.get("text", [])

                for label, value in zip(
                    labels,
                    values,
                ):
                    text += (
                        f"\n{label}: {value}"
                    )

            if answer:
                text += (
                    f"\nAnswer: {answer}"
                )

            add_fact(text)
            count += 1

            if count >= 10000:
                break

        print(
            f"  Q&A examples: {count:,}"
        )

    except Exception as error:
        print(
            f"  Q&A failed: {error}"
        )


def load_general_knowledge():
    print("[5/5] Loading general knowledge...")

    try:
        dataset = load_dataset(
            "dair-ai/emotion",
            split="train",
            streaming=True,
        )

        count = 0

        for item in dataset:
            text = item.get("text")

            if text:
                add_fact(text)
                count += 1

            if count >= 20000:
                break

            if sum(len(x) for x in OUTPUT) >= MAX_CHARS:
                break

        print(
            f"  General examples: {count:,}"
        )

    except Exception as error:
        print(
            f"  General dataset failed: {error}"
        )


def main():
    print("=" * 60)
    print("ORANGEGLASSES V4 FACT DATASET")
    print("=" * 60)

    load_wikipedia()
    load_openwebmath()
    load_science()
    load_qa()
    load_general_knowledge()

    print()
    print("Combining knowledge datasets...")

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
    print("FACT DATASET READY")
    print("=" * 60)
    print(f"Output:     {OUTPUT_PATH}")
    print(f"Characters: {len(final_text):,}")
    print(f"Documents:  {count:,}")


if __name__ == "__main__":
    main()