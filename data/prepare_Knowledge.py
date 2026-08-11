import os
import random

from datasets import load_dataset


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "knowledge.txt")

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

    block = (
        "[KNOWLEDGE]\n"
        "User: " + question + "\n"
        "Assistant: " + answer + "\n\n"
    )

    output.append(block)


def load_dolly(output):
    print("[1/8] Loading Dolly")

    try:
        dataset = load_dataset(
            "databricks/databricks-dolly-15k",
            split="train"
        )
    except Exception as error:
        print("Skipped Dolly:", error)
        return

    count = 0

    for item in dataset:
        instruction = item.get("instruction")
        context = item.get("context")
        response = item.get("response")

        if not instruction or not response:
            continue

        question = instruction

        if context:
            question += (
                "\n\nContext:\n"
                + str(context)
            )

        add_example(
            output,
            question,
            response
        )

        count += 1

    print(
        f"Added {count:,} examples."
    )


def load_openassistant(output):
    print("[2/8] Loading OpenAssistant")

    try:
        dataset = load_dataset(
            "OpenAssistant/oasst1"
        )
    except Exception as error:
        print("Skipped OpenAssistant:", error)
        return

    count = 0

    for split_name in dataset:
        split = dataset[split_name]

        for item in split:
            if item.get("lang") != "en":
                continue

            role = item.get("role")
            text = item.get("text")

            if role != "prompter":
                continue

            if not text:
                continue

            add_example(
                output,
                text,
                "Please provide a complete and accurate answer."
            )

            count += 1

    print(
        f"Added {count:,} examples."
    )


def load_scienceqa(output):
    print("[3/8] Loading ScienceQA")

    try:
        dataset = load_dataset(
            "derek-thomas/ScienceQA",
            split="train"
        )
    except Exception as error:
        print("Skipped ScienceQA:", error)
        return

    count = 0

    for item in dataset:
        question = item.get("question")
        choices = item.get("choices")
        answer = item.get("answer")

        if not question:
            continue

        if choices:
            question_text = (
                str(question)
                + "\nChoices: "
                + ", ".join(
                    str(choice)
                    for choice in choices
                )
            )
        else:
            question_text = str(question)

        if isinstance(answer, int) and choices:
            if 0 <= answer < len(choices):
                answer_text = (
                    f"The correct answer is "
                    f"{choices[answer]}."
                )
            else:
                answer_text = str(answer)
        else:
            answer_text = str(answer)

        add_example(
            output,
            question_text,
            answer_text
        )

        count += 1

    print(
        f"Added {count:,} examples."
    )


def load_arc(output):
    print("[4/8] Loading ARC")

    try:
        dataset = load_dataset(
            "allenai/ai2_arc",
            "ARC-Challenge",
            split="train"
        )
    except Exception as error:
        print("Skipped ARC:", error)
        return

    count = 0

    for item in dataset:
        question = item.get("question")
        choices = item.get("choices")
        answer = item.get("answerKey")

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
                f"{label}: {text}"
            )

        question_text = (
            str(question)
            + "\n"
            + "\n".join(choice_lines)
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
        f"Added {count:,} examples."
    )


def load_math(output):
    print("[5/8] Loading GSM8K")

    try:
        dataset = load_dataset(
            "openai/gsm8k",
            "main",
            split="train"
        )
    except Exception as error:
        print("Skipped GSM8K:", error)
        return

    count = 0

    for item in dataset:
        question = item.get("question")
        answer = item.get("answer")

        if not question or not answer:
            continue

        add_example(
            output,
            question,
            answer
        )

        count += 1

    print(
        f"Added {count:,} examples."
    )


def load_general_qa(output):
    print("[6/8] Loading TriviaQA")

    try:
        dataset = load_dataset(
            "trivia_qa",
            "rc",
            split="train"
        )
    except Exception as error:
        print("Skipped TriviaQA:", error)
        return

    count = 0

    for item in dataset:
        question = item.get("question")
        answer_data = item.get("answer")

        if not question or not answer_data:
            continue

        if isinstance(
            answer_data,
            dict
        ):
            answer = answer_data.get(
                "value"
            )
        else:
            answer = answer_data

        if not answer:
            continue

        add_example(
            output,
            question,
            str(answer)
        )

        count += 1

        if count >= 100000:
            break

    print(
        f"Added {count:,} examples."
    )


def load_wikipedia(output):
    print("[7/8] Loading Wikipedia")

    try:
        dataset = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split="train"
        )
    except Exception as error:
        print("Skipped Wikipedia:", error)
        return

    count = 0

    for item in dataset:
        title = item.get("title")
        text = item.get("text")

        if not title or not text:
            continue

        text = clean_text(text)

        if len(text) < 100:
            continue

        answer = text[:4000]

        question = (
            "Explain what "
            + title
            + " is and provide useful factual information about it."
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
        f"Added {count:,} examples."
    )


def load_web_questions(output):
    print("[8/8] Loading Web Questions")

    try:
        dataset = load_dataset(
            "stanfordnlp/web_questions",
            split="train"
        )
    except Exception as error:
        print("Skipped Web Questions:", error)
        return

    count = 0

    for item in dataset:
        question = item.get("question")
        answers = item.get("answers")

        if not question or not answers:
            continue

        if isinstance(
            answers,
            list
        ):
            answer = ", ".join(
                str(value)
                for value in answers
            )
        else:
            answer = str(answers)

        add_example(
            output,
            question,
            answer
        )

        count += 1

    print(
        f"Added {count:,} examples."
    )


def build():
    os.makedirs(
        DATA_DIR,
        exist_ok=True
    )

    output = []

    load_dolly(output)
    load_openassistant(output)
    load_scienceqa(output)
    load_arc(output)
    load_math(output)
    load_general_qa(output)
    load_wikipedia(output)
    load_web_questions(output)

    print()
    print("Shuffling knowledge examples...")

    random.shuffle(output)

    final_blocks = []
    total = 0

    for block in output:
        size = len(block)

        if total + size > TARGET_CHARS:
            break

        final_blocks.append(block)
        total += size

    text = "".join(final_blocks)

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(text)

    print()
    print("=" * 60)
    print("KNOWLEDGE DATASET READY")
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
    build()


if __name__ == "__main__":
    main()