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
    "science.txt"
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
    question,
    answer
):
    question = clean_text(
        question
    )

    answer = clean_text(
        answer
    )

    if not question or not answer:
        return

    output.append(
        "[SCIENCE]\n"
        "User: "
        + question
        + "\n"
        "Assistant: "
        + answer
        + "\n\n"
    )


def load_scienceqa(output):
    print(
        "[1/7] Loading ScienceQA..."
    )

    try:
        dataset = load_dataset(
            "derek-thomas/ScienceQA"
        )
    except Exception as error:
        print(
            "Skipped ScienceQA:"
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

            choices = item.get(
                "choices"
            )

            answer = item.get(
                "answer"
            )

            lecture = item.get(
                "lecture"
            )

            if not question:
                continue

            question_text = str(
                question
            )

            if choices:
                question_text += (
                    "\nChoices:\n"
                    + "\n".join(
                        str(choice)
                        for choice in choices
                    )
                )

            answer_parts = []

            if lecture:
                answer_parts.append(
                    str(lecture)
                )

            if isinstance(
                answer,
                int
            ) and choices:
                if (
                    0 <= answer
                    < len(choices)
                ):
                    answer_parts.append(
                        "Answer: "
                        + str(
                            choices[answer]
                        )
                    )
            elif answer is not None:
                answer_parts.append(
                    "Answer: "
                    + str(answer)
                )

            if not answer_parts:
                continue

            add_example(
                output,
                question_text,
                "\n".join(
                    answer_parts
                )
            )

            count += 1

    print(
        f"Added {count:,} ScienceQA examples."
    )


def load_arc_science(output):
    print(
        "[2/7] Loading ARC science..."
    )

    try:
        dataset = load_dataset(
            "allenai/ai2_arc",
            "ARC-Challenge"
        )
    except Exception as error:
        print(
            "Skipped ARC:"
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

            choices = item.get(
                "choices"
            )

            answer = item.get(
                "answerKey"
            )

            if not question or not choices:
                continue

            labels = choices.get(
                "label",
                []
            )

            texts = choices.get(
                "text",
                []
            )

            choice_lines = []

            for label, text in zip(
                labels,
                texts
            ):
                choice_lines.append(
                    str(label)
                    + ": "
                    + str(text)
                )

            question_text = (
                str(question)
                + "\nChoices:\n"
                + "\n".join(
                    choice_lines
                )
            )

            answer_text = (
                "The correct answer is "
                + str(answer)
                + "."
            )

            add_example(
                output,
                question_text,
                answer_text
            )

            count += 1

    print(
        f"Added {count:,} ARC examples."
    )


def load_openbookqa(output):
    print(
        "[3/7] Loading OpenBookQA..."
    )

    try:
        dataset = load_dataset(
            "allenai/openbookqa",
            "main"
        )
    except Exception as error:
        print(
            "Skipped OpenBookQA:"
        )
        print(error)
        return

    count = 0

    for split_name in dataset:
        split = dataset[
            split_name
        ]

        for item in split:
            question_data = item.get(
                "question_stem"
            )

            choices = item.get(
                "choices"
            )

            answer = item.get(
                "answerKey"
            )

            if not question_data:
                continue

            question_text = str(
                question_data
            )

            if choices:
                labels = choices.get(
                    "label",
                    []
                )

                texts = choices.get(
                    "text",
                    []
                )

                choice_lines = []

                for label, text in zip(
                    labels,
                    texts
                ):
                    choice_lines.append(
                        str(label)
                        + ": "
                        + str(text)
                    )

                question_text += (
                    "\nChoices:\n"
                    + "\n".join(
                        choice_lines
                    )
                )

            answer_text = (
                "The correct answer is "
                + str(answer)
                + "."
            )

            add_example(
                output,
                question_text,
                answer_text
            )

            count += 1

    print(
        f"Added {count:,} OpenBookQA examples."
    )


def load_science_explanations(output):
    print(
        "[4/7] Loading science explanations..."
    )

    try:
        dataset = load_dataset(
            "camel-ai/physics"
        )
    except Exception as error:
        print(
            "Skipped physics explanations:"
        )
        print(error)
        return

    count = 0

    for split_name in dataset:
        split = dataset[
            split_name
        ]

        for item in split:
            question = (
                item.get("question")
                or item.get("instruction")
                or item.get("prompt")
            )

            answer = (
                item.get("answer")
                or item.get("response")
                or item.get("solution")
            )

            if not question or not answer:
                continue

            add_example(
                output,
                (
                    "Explain and solve this "
                    "physics question:\n"
                    + str(question)
                ),
                str(answer)
            )

            count += 1

            if count >= 100000:
                break

        if count >= 100000:
            break

    print(
        f"Added {count:,} physics examples."
    )


def load_medical_science(output):
    print(
        "[5/7] Loading biology and science QA..."
    )

    try:
        dataset = load_dataset(
            "medmcqa"
        )
    except Exception as error:
        print(
            "Skipped biology/science QA:"
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

            answer = item.get(
                "cop"
            )

            explanation = item.get(
                "exp"
            )

            if not question:
                continue

            answer_parts = []

            if explanation:
                answer_parts.append(
                    str(explanation)
                )

            if answer is not None:
                answer_parts.append(
                    "Correct option: "
                    + str(answer)
                )

            if not answer_parts:
                continue

            add_example(
                output,
                (
                    "Answer this biology "
                    "or medical science "
                    "question:\n"
                    + str(question)
                ),
                "\n".join(
                    answer_parts
                )
            )

            count += 1

            if count >= 100000:
                break

        if count >= 100000:
            break

    print(
        f"Added {count:,} biology examples."
    )


def load_worldtree(output):
    print(
        "[6/7] Loading WorldTree..."
    )

    try:
        dataset = load_dataset(
            "allenai/worldtree"
        )
    except Exception as error:
        print(
            "Skipped WorldTree:"
        )
        print(error)
        return

    count = 0

    for split_name in dataset:
        split = dataset[
            split_name
        ]

        for item in split:
            question = (
                item.get("question")
            )

            if not question:
                continue

            answer = (
                item.get("answer")
                or item.get("answer_text")
                or item.get("explanation")
            )

            if not answer:
                continue

            add_example(
                output,
                (
                    "Explain the scientific "
                    "reasoning behind this "
                    "question:\n"
                    + str(question)
                ),
                str(answer)
            )

            count += 1

            if count >= 100000:
                break

        if count >= 100000:
            break

    print(
        f"Added {count:,} WorldTree examples."
    )


def load_wikipedia_science(output):
    print(
        "[7/7] Loading Wikipedia science..."
    )

    try:
        dataset = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split="train"
        )
    except Exception as error:
        print(
            "Skipped Wikipedia:"
        )
        print(error)
        return

    science_terms = [
        "physics",
        "chemistry",
        "biology",
        "astronomy",
        "geology",
        "mathematics",
        "ecology",
        "genetics",
        "cell",
        "planet",
        "atom",
        "molecule",
        "evolution",
        "gravity",
        "energy",
        "electricity",
        "magnetism",
        "thermodynamics",
        "quantum",
        "scientific"
    ]

    count = 0

    for item in dataset:
        title = item.get(
            "title"
        )

        text = item.get(
            "text"
        )

        if not title or not text:
            continue

        title_lower = str(
            title
        ).lower()

        if not any(
            term in title_lower
            for term in science_terms
        ):
            continue

        text = clean_text(
            text
        )

        if len(text) < 200:
            continue

        answer = text[:5000]

        question = (
            "Explain the scientific topic "
            "of "
            + str(title)
            + " in a clear and useful way."
        )

        add_example(
            output,
            question,
            answer
        )

        count += 1

        if count >= 50000:
            break

    print(
        f"Added {count:,} Wikipedia science examples."
    )


def build_dataset():
    os.makedirs(
        DATA_DIR,
        exist_ok=True
    )

    output = []

    load_scienceqa(
        output
    )

    load_arc_science(
        output
    )

    load_openbookqa(
        output
    )

    load_science_explanations(
        output
    )

    load_medical_science(
        output
    )

    load_worldtree(
        output
    )

    load_wikipedia_science(
        output
    )

    print()
    print(
        "Shuffling science examples..."
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
    print("=" * 60) 
    print("SCIENCE DATASET READY")
    print("=" * 60)

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