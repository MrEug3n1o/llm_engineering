"""Interactive Spanish tutor backed by a local Ollama model."""

import csv
import io

from openai import OpenAI


MODEL = "llama3.2"
OLLAMA_URL = "http://localhost:11434/v1"

SYSTEM_PROMPT = """You are a friendly Spanish tutor.
Help the learner practice Spanish, explain mistakes clearly, and keep answers practical.
Whenever you write a Spanish noun for learning, include its definite article:
el, la, los, or las. Use examples such as la manzana and el plátano.
For flashcards, preserve articles in the Spanish column.
"""

FLASHCARD_PROMPT = """Create import-ready CSV flashcards for Flashcards.com.
Return only CSV card rows, with no header row, followed by no more than 50 cards.
Quote both fields with double quotes. Put Spanish on the front and English on the back.
Always include the definite article with Spanish nouns: el, la, los, or las.
Create only cards relevant to the user's request; do not invent names or unrelated words.
Do not include numbering, explanations, Markdown, or code fences.

User request:
"""


def normalize_csv(text: str) -> str:
    """Repair common local-model formatting mistakes before printing CSV."""
    cleaned = text.strip().replace("```csv", "").replace("```", "").strip()
    rows = list(csv.reader(io.StringIO(cleaned)))
    if rows and all(len(row) == 1 for row in rows):
        values = [row[0].strip().strip('"') for row in rows[1:] if row[0].strip()]
        rows = [["Spanish", "English"]]
        rows.extend(values[i : i + 2] for i in range(0, len(values) - 1, 2))
    if rows and [cell.lower() for cell in rows[0][:2]] == ["spanish", "english"]:
        rows = rows[1:]
    rows = [row[:2] for row in rows if len(row) >= 2]
    rows = [row for row in rows if [cell.lower() for cell in row] != ["spanish", "english"]]
    if len(rows) > 50:
        raise ValueError("The model returned more than 50 cards.")
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(rows)
    return output.getvalue().strip()


def main() -> None:
    client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Spanish tutor ready. Type a message, /cards <request>, /reset, or /quit.")
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not user_input:
            continue
        if user_input == "/quit":
            print("Bye!")
            break
        if user_input == "/reset":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("Conversation reset.")
            continue

        if user_input.startswith("/cards"):
            request = user_input.removeprefix("/cards").strip()
            if not request:
                request = input("Card request: ").strip()
            prompt = FLASHCARD_PROMPT + request
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": prompt}],
            )
            print("\nCopy this CSV into Flashcards.com:\n")
            print(normalize_csv(response.choices[0].message.content))
            continue

        messages.append({"role": "user", "content": user_input})
        response = client.chat.completions.create(model=MODEL, messages=messages)
        reply = response.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": reply})
        print(f"\nTutor: {reply}")


if __name__ == "__main__":
    main()
