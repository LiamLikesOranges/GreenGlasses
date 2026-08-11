import os
import random

from datasets import load_dataset


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

OUTPUT_PATH = os.path.join(
    DATA_DIR,
    "conversation2.txt"
)

TARGET_CHARS = 12000000
SEED = 18

random.seed(SEED)


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


def add_example(
    output,
    user_text,
    assistant_text
):
    user_text = clean_text(
        user_text
    )

    assistant_text = clean_text(
        assistant_text
    )

    if not user_text:
        return

    if not assistant_text:
        return

    output.append(
        "[LANGUAGE]\n"
        "User: "
        + user_text
        + "\n"
        "Assistant: "
        + assistant_text
        + "\n\n"
    )


def load_grammar_correction(
    output
):
    print(
        "[1/7] Loading grammar correction..."
    )

    try:
        dataset = load_dataset(
            "agentlans/grammar-correction"
        )
    except Exception as error:
        print(
            "Skipped grammar correction:"
        )
        print(error)
        return

    count = 0

    for split_name in dataset:
        split = dataset[
            split_name
        ]

        for item in split:
            bad = item.get(
                "input"
            )

            good = item.get(
                "output"
            )

            if not bad or not good:
                continue

            add_example(
                output,
                (
                    "Correct the grammar "
                    "in this sentence:\n"
                    + str(bad)
                ),
                str(good)
            )

            count += 1

    print(
        f"Added {count:,} grammar examples."
    )


def load_jfleg(
    output
):
    print(
        "[2/7] Loading JFLEG..."
    )

    try:
        dataset = load_dataset(
            "jhu-clsp/jfleg"
        )
    except Exception as error:
        print(
            "Skipped JFLEG:"
        )
        print(error)
        return

    count = 0

    for split_name in dataset:
        split = dataset[
            split_name
        ]

        for item in split:
            sentence = item.get(
                "sentence"
            )

            corrections = item.get(
                "corrections"
            )

            if not sentence:
                continue

            if not corrections:
                continue

            for correction in corrections:
                if not correction:
                    continue

                add_example(
                    output,
                    (
                        "Improve the grammar "
                        "and fluency of this sentence:\n"
                        + str(sentence)
                    ),
                    str(correction)
                )

                count += 1

    print(
        f"Added {count:,} JFLEG examples."
    )


def load_vocabulary(
    output
):
    print(
        "[3/7] Loading vocabulary..."
    )

    try:
        dataset = load_dataset(
            "MongoDB/english-words-definitions"
        )
    except Exception as error:
        print(
            "Skipped vocabulary:"
        )
        print(error)
        return

    count = 0

    for split_name in dataset:
        split = dataset[
            split_name
        ]

        for item in split:
            word = item.get(
                "word"
            )

            definition = item.get(
                "definition"
            )

            if not word or not definition:
                continue

            add_example(
                output,
                (
                    "What does the word "
                    + str(word)
                    + " mean?"
                ),
                str(definition)
            )

            count += 1

            if count >= 300000:
                break

        if count >= 300000:
            break

    print(
        f"Added {count:,} vocabulary examples."
    )


def load_paraphrases(
    output
):
    print(
        "[4/7] Loading PAWS..."
    )

    try:
        dataset = load_dataset(
            "google-research-datasets/paws",
            "labeled_final"
        )
    except Exception as error:
        print(
            "Skipped PAWS:"
        )
        print(error)
        return

    count = 0

    for split_name in dataset:
        split = dataset[
            split_name
        ]

        for item in split:
            sentence1 = item.get(
                "sentence1"
            )

            sentence2 = item.get(
                "sentence2"
            )

            label = item.get(
                "label"
            )

            if not sentence1 or not sentence2:
                continue

            if label == 1:
                add_example(
                    output,
                    (
                        "Rewrite this sentence "
                        "using different wording "
                        "while keeping the same meaning:\n"
                        + str(sentence1)
                    ),
                    str(sentence2)
                )

                count += 1

            if count >= 100000:
                break

        if count >= 100000:
            break

    print(
        f"Added {count:,} paraphrasing examples."
    )


def load_squad(
    output
):
    print(
        "[5/7] Loading SQuAD..."
    )

    try:
        dataset = load_dataset(
            "rajpurkar/squad"
        )
    except Exception as error:
        print(
            "Skipped SQuAD:"
        )
        print(error)
        return

    count = 0

    for split_name in dataset:
        split = dataset[
            split_name
        ]

        for item in split:
            question = item.get(
                "question"
            )

            context = item.get(
                "context"
            )

            answers = item.get(
                "answers"
            )

            if not question or not context:
                continue

            answer_text = ""

            if isinstance(
                answers,
                dict
            ):
                values = answers.get(
                    "text",
                    []
                )

                if values:
                    answer_text = str(
                        values[0]
                    )

            if not answer_text:
                continue

            add_example(
                output,
                (
                    str(question)
                    + "\n\nContext:\n"
                    + str(context)
                ),
                answer_text
            )

            count += 1

            if count >= 100000:
                break

        if count >= 100000:
            break

    print(
        f"Added {count:,} language understanding examples."
    )


def load_blended_skill_talk(
    output
):
    print(
        "[6/7] Loading Blended Skill Talk..."
    )

    try:
        dataset = load_dataset(
            "blended_skill_talk"
        )
    except Exception as error:
        print(
            "Skipped Blended Skill Talk:"
        )
        print(error)
        return

    count = 0

    for split_name in dataset:
        split = dataset[
            split_name
        ]

        for item in split:
            dialog = item.get(
                "dialog"
            )

            if not dialog:
                continue

            if not isinstance(
                dialog,
                list
            ):
                continue

            for i in range(
                len(dialog) - 1
            ):
                user_text = dialog[i]
                assistant_text = dialog[
                    i + 1
                ]

                if not user_text:
                    continue

                if not assistant_text:
                    continue

                add_example(
                    output,
                    str(user_text),
                    str(assistant_text)
                )

                count += 1

    print(
        f"Added {count:,} skill examples."
    )


def load_daily_dialog(
    output
):
    print(
        "[7/7] Loading DailyDialog..."
    )

    try:
        dataset = load_dataset(
            "roskoN/dailydialog"
        )
    except Exception as error:
        print(
            "Skipped DailyDialog:"
        )
        print(error)
        return

    count = 0

    for split_name in dataset:
        split = dataset[
            split_name
        ]

        for item in split:
            dialogue = item.get(
                "dialog"
            )

            if not dialogue:
                continue

            if not isinstance(
                dialogue,
                list
            ):
                continue

            for i in range(
                len(dialogue) - 1
            ):
                user_text = dialogue[i]
                assistant_text = dialogue[
                    i + 1
                ]

                if not user_text:
                    continue

                if not assistant_text:
                    continue

                add_example(
                    output,
                    str(user_text),
                    str(assistant_text)
                )

                count += 1

    print(
        f"Added {count:,} dialogue examples."
    )


def build_dataset():
    os.makedirs(
        DATA_DIR,
        exist_ok=True
    )

    output = []

    load_grammar_correction(
        output
    )

    load_jfleg(
        output
    )

    load_vocabulary(
        output
    )

    load_paraphrases(
        output
    )

    load_squad(
        output
    )

    load_blended_skill_talk(
        output
    )

    load_daily_dialog(
        output
    )

    print()
    print(
        "Shuffling language examples..."
    )

    random.shuffle(
        output
    )

    final_blocks = []
    total_chars = 0

    for block in output:
        block_size = len(block)

        if (
            total_chars
            + block_size
            > TARGET_CHARS
        ):
            break

        final_blocks.append(
            block
        )

        total_chars += block_size

    text = "".join(
        final_blocks
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(
            text
        )

    print()
    print(
        "=" * 60
    )

    print(
        "CONVERSATION 2 DATASET READY"
    )

    print(
        "=" * 60
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    print(
        f"Characters: {len(text):,}"
    )

    print(
        f"Examples: {len(final_blocks):,}"
    )


def main():
    build_dataset()


if __name__ == "__main__":
    main()