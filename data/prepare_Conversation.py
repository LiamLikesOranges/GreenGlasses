import os
import random

from datasets import load_dataset


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "conversation.txt")

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


def add_conversation(output, conversation):
    cleaned = []

    for speaker, text in conversation:
        text = clean_text(text)

        if not text:
            continue

        cleaned.append(
            f"{speaker}: {text}"
        )

    if len(cleaned) < 2:
        return

    output.append(
        "[CONVERSATION]\n"
        + "\n".join(cleaned)
        + "\n\n"
    )


def load_openassistant(output):
    print("[1/8] OpenAssistant")

    dataset = load_dataset(
        "OpenAssistant/oasst1"
    )

    count = 0

    for split_name in dataset:
        split = dataset[split_name]

        for item in split:
            if item.get("lang") != "en":
                continue

            text = item.get("text")

            if not text:
                continue

            role = item.get(
                "role",
                "assistant"
            )

            speaker = (
                "User"
                if role == "prompter"
                else "Assistant"
            )

            add_conversation(
                output,
                [
                    (
                        speaker,
                        text
                    ),
                    (
                        "Assistant",
                        "Understood."
                    )
                ]
            )

            count += 1

    print(
        f"Added {count:,} messages."
    )


def load_empathetic(output):
    print("[2/8] EmpatheticDialogues")

    dataset = load_dataset(
        "facebook/empathetic_dialogues"
    )

    count = 0

    for split_name in dataset:
        split = dataset[split_name]

        conversations = {}

        for item in split:
            conv_id = item.get(
                "conv_id"
            )

            text = item.get(
                "utterance"
            )

            if not text:
                continue

            speaker_id = item.get(
                "speaker_idx"
            )

            conversations.setdefault(
                conv_id,
                []
            ).append(
                (
                    f"Person{speaker_id}",
                    text
                )
            )

        for conversation in conversations.values():
            add_conversation(
                output,
                conversation
            )

            count += 1

    print(
        f"Added {count:,} conversations."
    )


def load_personachat(output):
    print("[3/8] PersonaChat")

    try:
        dataset = load_dataset(
            "bavard/personachat_truecased"
        )
    except Exception as error:
        print(
            f"Skipped PersonaChat: {error}"
        )
        return

    count = 0

    for split_name in dataset:
        split = dataset[split_name]

        for item in split:
            history = item.get(
                "history"
            )

            candidates = item.get(
                "candidates"
            )

            if not history:
                continue

            conversation = []

            for i, message in enumerate(history):
                speaker = (
                    "User"
                    if i % 2 == 0
                    else "Assistant"
                )

                conversation.append(
                    (
                        speaker,
                        message
                    )
                )

            if candidates:
                if isinstance(
                    candidates,
                    list
                ):
                    response = candidates[-1]
                else:
                    response = candidates

                if response:
                    conversation.append(
                        (
                            "Assistant",
                            response
                        )
                    )

            add_conversation(
                output,
                conversation
            )

            count += 1

    print(
        f"Added {count:,} conversations."
    )


def load_daily_dialog(output):
    print("[4/8] DailyDialog")

    try:
        dataset = load_dataset(
            "roskoN/dailydialog"
        )
    except Exception as error:
        print(
            f"Skipped DailyDialog: {error}"
        )
        return

    count = 0

    for split_name in dataset:
        split = dataset[split_name]

        for item in split:
            dialogue = item.get(
                "dialog"
            )

            if not dialogue:
                continue

            conversation = []

            for i, text in enumerate(dialogue):
                speaker = (
                    "User"
                    if i % 2 == 0
                    else "Assistant"
                )

                conversation.append(
                    (
                        speaker,
                        text
                    )
                )

            add_conversation(
                output,
                conversation
            )

            count += 1

    print(
        f"Added {count:,} conversations."
    )


def load_ultrachat(output):
    print("[5/8] UltraChat")

    try:
        dataset = load_dataset(
            "HuggingFaceH4/ultrachat_200k",
            split="train_sft"
        )
    except Exception as error:
        print(
            f"Skipped UltraChat: {error}"
        )
        return

    count = 0

    for item in dataset:
        messages = item.get(
            "messages"
        )

        if not messages:
            continue

        conversation = []

        for message in messages:
            role = message.get(
                "role"
            )

            content = message.get(
                "content"
            )

            if not content:
                continue

            if role == "user":
                speaker = "User"
            elif role == "assistant":
                speaker = "Assistant"
            else:
                continue

            conversation.append(
                (
                    speaker,
                    content
                )
            )

        add_conversation(
            output,
            conversation
        )

        count += 1

        if count >= 100000:
            break

    print(
        f"Added {count:,} conversations."
    )


def load_soda(output):
    print("[6/8] SODA")

    try:
        dataset = load_dataset(
            "allenai/soda",
            split="train"
        )
    except Exception as error:
        print(
            f"Skipped SODA: {error}"
        )
        return

    count = 0

    for item in dataset:
        dialogue = item.get(
            "dialogue"
        )

        if not dialogue:
            continue

        conversation = []

        for i, text in enumerate(dialogue):
            speaker = (
                "User"
                if i % 2 == 0
                else "Assistant"
            )

            conversation.append(
                (
                    speaker,
                    text
                )
            )

        add_conversation(
            output,
            conversation
        )

        count += 1

        if count >= 100000:
            break

    print(
        f"Added {count:,} conversations."
    )


def load_dolly(output):
    print("[7/8] Dolly")

    try:
        dataset = load_dataset(
            "databricks/databricks-dolly-15k",
            split="train"
        )
    except Exception as error:
        print(
            f"Skipped Dolly: {error}"
        )
        return

    count = 0

    for item in dataset:
        instruction = item.get(
            "instruction"
        )

        context = item.get(
            "context"
        )

        response = item.get(
            "response"
        )

        if not instruction or not response:
            continue

        user_text = instruction

        if context:
            user_text += (
                "\n\nContext:\n"
                + str(context)
            )

        add_conversation(
            output,
            [
                (
                    "User",
                    user_text
                ),
                (
                    "Assistant",
                    response
                )
            ]
        )

        count += 1

    print(
        f"Added {count:,} conversations."
    )


def load_oasst2(output):
    print("[8/8] OpenAssistant additional")

    try:
        dataset = load_dataset(
            "OpenAssistant/oasst2",
            split="train"
        )
    except Exception as error:
        print(
            f"Skipped OASST2: {error}"
        )
        return

    count = 0

    for item in dataset:
        if item.get("lang") != "en":
            continue

        text = item.get("text")

        if not text:
            continue

        role = item.get(
            "role",
            "assistant"
        )

        speaker = (
            "User"
            if role == "prompter"
            else "Assistant"
        )

        add_conversation(
            output,
            [
                (
                    speaker,
                    text
                )
            ]
        )

        count += 1

        if count >= 100000:
            break

    print(
        f"Added {count:,} messages."
    )


def build():
    os.makedirs(
        DATA_DIR,
        exist_ok=True
    )

    output = []

    load_openassistant(output)
    load_empathetic(output)
    load_personachat(output)
    load_daily_dialog(output)
    load_ultrachat(output)
    load_soda(output)
    load_dolly(output)
    load_oasst2(output)

    print()
    print("Combining conversations...")

    random.shuffle(output)

    final_text = []
    total = 0

    for block in output:
        size = len(block)

        if total + size > TARGET_CHARS:
            break

        final_text.append(block)
        total += size

    text = "".join(final_text)

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(text)

    print()
    print("=" * 60)
    print("CONVERSATION DATASET READY")
    print("=" * 60)
    print(
        f"Output: {OUTPUT_PATH}"
    )
    print(
        f"Characters: {len(text):,}"
    )
    print(
        f"Conversations: {len(final_text):,}"
    )


def main():
    build()


if __name__ == "__main__":
    main()