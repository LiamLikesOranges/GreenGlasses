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
    "math.txt"
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


def add_example(output, question, answer):
    question = clean_text(question)
    answer = clean_text(answer)

    if not question or not answer:
        return

    output.append(
        "[MATH]\n"
        "User: "
        + question
        + "\n"
        "Assistant: "
        + answer
        + "\n\n"
    )


def load_gsm8k(output):
    print("[1/6] Loading GSM8K...")

    try:
        dataset = load_dataset(
            "openai/gsm8k",
            "main"
        )
    except Exception as error:
        print("Skipped GSM8K:")
        print(error)
        return

    count = 0

    for split_name in dataset:
        split = dataset[split_name]

        for item in split:
            question = item.get("question")
            answer = item.get("answer")

            if not question or not answer:
                continue

            add_example(
                output,
                (
                    "Solve this math problem "
                    "step by step:\n"
                    + str(question)
                ),
                str(answer)
            )

            count += 1

    print(
        f"Added {count:,} GSM8K examples."
    )


def load_math_qa(output):
    print("[2/6] Loading MathQA...")

    try:
        dataset = load_dataset(
            "allenai/math_qa"
        )
    except Exception as error:
        print("Skipped MathQA:")
        print(error)
        return

    count = 0

    for split_name in dataset:
        split = dataset[split_name]

        for item in split:
            problem = item.get("Problem")
            options = item.get("options")
            correct = item.get("correct")
            rationale = item.get("Rationale")

            if not problem:
                continue

            question = str(problem)

            if options:
                question += (
                    "\nChoices:\n"
                    + str(options)
                )

            answer_parts = []

            if rationale:
                answer_parts.append(
                    str(rationale)
                )

            if correct:
                answer_parts.append(
                    "Correct answer: "
                    + str(correct)
                )

            if not answer_parts:
                continue

            answer = "\n".join(
                answer_parts
            )

            add_example(
                output,
                question,
                answer
            )

            count += 1

    print(
        f"Added {count:,} MathQA examples."
    )


def load_aqua(output):
    print("[3/6] Loading AQUA-RAT...")

    try:
        dataset = load_dataset(
            "deepmind/aqua_rat"
        )
    except Exception as error:
        print("Skipped AQUA-RAT:")
        print(error)
        return

    count = 0

    for split_name in dataset:
        split = dataset[split_name]

        for item in split:
            question = item.get("question")
            options = item.get("options")
            rationale = item.get("rationale")
            correct = item.get("correct")

            if not question:
                continue

            question_text = str(question)

            if options:
                question_text += (
                    "\nChoices:\n"
                    + "\n".join(
                        str(option)
                        for option in options
                    )
                )

            answer_parts = []

            if rationale:
                answer_parts.append(
                    str(rationale)
                )

            if correct:
                answer_parts.append(
                    "Correct answer: "
                    + str(correct)
                )

            if not answer_parts:
                continue

            add_example(
                output,
                question_text,
                "\n".join(answer_parts)
            )

            count += 1

    print(
        f"Added {count:,} AQUA-RAT examples."
    )


def load_svamp(output):
    print("[4/6] Loading SVAMP...")

    try:
        dataset = load_dataset(
            "ChilleD/SVAMP"
        )
    except Exception as error:
        print("Skipped SVAMP:")
        print(error)
        return

    count = 0

    for split_name in dataset:
        split = dataset[split_name]

        for item in split:
            body = item.get("Body")
            question = item.get("Question")
            answer = item.get("Answer")
            equation = item.get("Equation")

            if not question:
                continue

            question_text = ""

            if body:
                question_text += (
                    str(body)
                    + "\n"
                )

            question_text += str(question)

            answer_parts = []

            if equation:
                answer_parts.append(
                    "Equation: "
                    + str(equation)
                )

            if answer is not None:
                answer_parts.append(
                    "Answer: "
                    + str(answer)
                )

            if not answer_parts:
                continue

            add_example(
                output,
                question_text,
                "\n".join(answer_parts)
            )

            count += 1

    print(
        f"Added {count:,} SVAMP examples."
    )


def load_math_word_problems(output):
    print("[5/6] Loading ASDiv...")

    try:
        dataset = load_dataset(
            "EleutherAI/asdiv"
        )
    except Exception as error:
        print("Skipped ASDiv:")
        print(error)
        return

    count = 0

    for split_name in dataset:
        split = dataset[split_name]

        for item in split:
            question = item.get("body")
            solution = item.get("solution")
            answer = item.get("answer")

            if not question:
                continue

            answer_parts = []

            if solution:
                answer_parts.append(
                    str(solution)
                )

            if answer:
                answer_parts.append(
                    "Answer: "
                    + str(answer)
                )

            if not answer_parts:
                continue

            add_example(
                output,
                (
                    "Solve this problem "
                    "and explain the reasoning:\n"
                    + str(question)
                ),
                "\n".join(answer_parts)
            )

            count += 1

    print(
        f"Added {count:,} ASDiv examples."
    )


def load_algebra(output):
    print("[6/6] Loading algebra data...")

    try:
        dataset = load_dataset(
            "math_dataset",
            "algebra__linear_1d",
            trust_remote_code=True
        )
    except Exception as error:
        print("Skipped algebra dataset:")
        print(error)
        return

    count = 0

    for split_name in dataset:
        split = dataset[split_name]

        for item in split:
            question = item.get("question")
            answer = item.get("answer")

            if not question or not answer:
                continue

            add_example(
                output,
                (
                    "Solve this algebra problem:\n"
                    + str(question)
                ),
                str(answer)
            )

            count += 1

            if count >= 200000:
                break

        if count >= 200000:
            break

    print(
        f"Added {count:,} algebra examples."
    )


def build_dataset():
    os.makedirs(
        DATA_DIR,
        exist_ok=True
    )

    output = []

    load_gsm8k(output)
    load_math_qa(output)
    load_aqua(output)
    load_svamp(output)
    load_math_word_problems(output)
    load_algebra(output)

    print()
    print(
        "Shuffling math examples..."
    )

    random.shuffle(output)

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

        final_blocks.append(block)
        total_chars += block_size

    text = "".join(
        final_blocks
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(text)

    print()
    print("=" * 60)
    print("MATH DATASET READY")
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