"""Week 3 exercise: a synthetic data generator, continuing the Spanish tutor project.

Describe a dataset (what kind, topic, CEFR level, how many rows) and an open-source model
generates it in batches. Every row is validated and de-duplicated, then you can preview it
and download it as CSV or JSONL - flashcards import straight into Flashcards.com.

Dataset types:
- flashcards:   spanish, english, part_of_speech, example   (nouns always carry el/la/los/las)
- corrections:  incorrect, correct, explanation               (typical learner mistakes)
- quiz:         question, option_a..option_d, answer, explanation
- dialogues:    scenario, speaker, spanish, english           (one row per line of dialogue)

Run it with:  python week3/spanish_data_generator.py   (needs `ollama serve`, or an API key in .env)
"""

import csv
import json
import os
import re
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd
import requests
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError, field_validator, model_validator

load_dotenv(override=True)

OLLAMA_URL = "http://localhost:11434/v1"
BATCH_SIZE = 10
MAX_ROWS = 200
MAX_ATTEMPTS_PER_BATCH = 3
LEVELS = ["A1", "A2", "B1", "B2", "C1"]
ARTICLES = ("el ", "la ", "los ", "las ")


class Flashcard(BaseModel):
    spanish: str
    english: str
    part_of_speech: str
    example: str

    @model_validator(mode="after")
    def noun_has_article(self):
        if self.part_of_speech.lower().startswith("noun") and not self.spanish.lower().startswith(ARTICLES):
            raise ValueError(f"noun without article: {self.spanish!r}")
        return self


class Correction(BaseModel):
    incorrect: str
    correct: str
    explanation: str

    @model_validator(mode="after")
    def actually_corrected(self):
        if self.incorrect.strip().lower() == self.correct.strip().lower():
            raise ValueError("the incorrect and correct sentences are identical")
        return self


class QuizQuestion(BaseModel):
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    answer: str
    explanation: str

    @field_validator("answer")
    @classmethod
    def answer_is_a_letter(cls, value):
        letter = value.strip().lower()[:1]
        if letter not in "abcd" or not letter:
            raise ValueError(f"answer must be a, b, c or d, got {value!r}")
        return letter


class DialogueLine(BaseModel):
    scenario: str
    speaker: str
    spanish: str
    english: str


DATASETS = {
    "flashcards": {
        "schema": Flashcard,
        "key": "spanish",
        "instructions": (
            "Each item is a vocabulary flashcard with keys spanish, english, part_of_speech "
            "(noun, verb, adjective, adverb or phrase) and example (a short Spanish sentence using "
            "the word). Every noun MUST include its definite article: 'la manzana', never 'manzana'."
        ),
    },
    "corrections": {
        "schema": Correction,
        "key": "incorrect",
        "instructions": (
            "Each item is a sentence with one typical English-speaker mistake, with keys incorrect, "
            "correct and explanation (one short English sentence naming the rule). Vary the mistakes: "
            "gender agreement, ser/estar, por/para, verb tense, false friends, word order."
        ),
    },
    "quiz": {
        "schema": QuizQuestion,
        "key": "question",
        "instructions": (
            "Each item is a multiple-choice question with keys question, option_a, option_b, "
            "option_c, option_d, answer (the single letter a, b, c or d) and explanation. "
            "Exactly one option is correct; spread the correct letter across a, b, c and d."
        ),
    },
    "dialogues": {
        "schema": DialogueLine,
        "key": None,  # dialogue lines legitimately repeat ("Gracias"), so only drop exact duplicate rows
        "instructions": (
            "Each item is one line of a short dialogue, with keys scenario (a few words, the same for "
            "every line of one dialogue), speaker (a first name), spanish and english. Write complete "
            "dialogues of 6 to 10 lines, in order, each in a different situation within the topic."
        ),
    },
}

SYSTEM_PROMPT = """You generate synthetic training data for Spanish learners whose native language is English.
The Spanish is natural, correct, and pitched exactly at the requested CEFR level.
You reply with a single JSON object {"items": [...]} and nothing else."""


def ollama_models():
    """Models on a local Ollama server; an empty list if it is not running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        response.raise_for_status()
        return [model["name"] for model in response.json().get("models", [])]
    except Exception:
        return []


def available_models():
    """Map a friendly label to (client, model): local open-source models first, then APIs."""
    models = {}
    ollama = OpenAI(base_url=OLLAMA_URL, api_key="ollama")
    for name in ollama_models():
        models[f"Ollama: {name}"] = (ollama, name)
    if os.getenv("OPENAI_API_KEY"):
        models["GPT-4.1-mini"] = (OpenAI(), "gpt-4.1-mini")
    return models


def batch_prompt(kind, topic, level, count, avoid):
    prompt = (
        f"Generate {count} {kind} items about the topic: {topic}. CEFR level: {level}.\n"
        f"{DATASETS[kind]['instructions']}\n"
        f'Reply as JSON: {{"items": [ ... {count} objects ... ]}}'
    )
    if avoid:
        # Showing what already exists is the simplest way to stop a small model repeating itself
        prompt += "\nDo not repeat any of these, which already exist: " + "; ".join(avoid[-40:])
    return prompt


def parse_items(text):
    """Pull the list of items out of a model reply, tolerating code fences and bare lists."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("items") or next((v for v in data.values() if isinstance(v, list)), [])
    return data if isinstance(data, list) else []


def generate(kind, topic, level, rows, model_label, temperature, progress=gr.Progress()):
    """Generate `rows` validated, de-duplicated rows. Returns (dataframe, log)."""
    models = available_models()
    if model_label not in models:
        raise gr.Error("No model available - start Ollama (`ollama serve`) or add an API key to .env.")
    client, model = models[model_label]
    spec = DATASETS[kind]
    rows = max(1, min(int(rows), MAX_ROWS))

    records, seen, log = [], set(), []
    attempts = 0
    max_attempts = MAX_ATTEMPTS_PER_BATCH * -(-rows // BATCH_SIZE)
    while len(records) < rows and attempts < max_attempts:
        attempts += 1
        progress(len(records) / rows, desc=f"{len(records)}/{rows} rows")
        want = min(BATCH_SIZE, rows - len(records))
        avoid = [r[spec["key"]] for r in records] if spec["key"] else []
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": batch_prompt(kind, topic, level, want, avoid)},
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
            )
            items = parse_items(response.choices[0].message.content)
        except Exception as error:
            log.append(f"batch {attempts}: request or JSON failed ({error.__class__.__name__})")
            continue

        kept = rejected = 0
        for item in items:
            try:
                record = spec["schema"].model_validate(item).model_dump()
            except ValidationError:
                rejected += 1
                continue
            fingerprint = (record[spec["key"]] if spec["key"] else json.dumps(record)).strip().lower()
            if fingerprint in seen:
                rejected += 1
                continue
            seen.add(fingerprint)
            records.append(record)
            kept += 1
            if len(records) >= rows:
                break
        log.append(f"batch {attempts}: kept {kept}, rejected {rejected}")

    if not records:
        raise gr.Error("The model produced no valid rows. Try a larger model or a lower temperature.")
    summary = f"{len(records)} of {rows} rows from {model} in {attempts} calls."
    return pd.DataFrame(records), summary + "\n" + "\n".join(log)


def export(df, kind, topic, fmt):
    """Write the preview table to a temp file for download."""
    if df is None or len(df) == 0:
        raise gr.Error("Generate some data first.")
    slug = re.sub(r"[^a-z0-9]+", "_", f"{kind}_{topic}".lower()).strip("_")[:50]
    path = Path(tempfile.mkdtemp()) / f"{slug}.{fmt}"
    if fmt == "jsonl":
        df.to_json(path, orient="records", lines=True, force_ascii=False)
    elif kind == "flashcards":
        # Flashcards.com import: front and back only, both fields quoted, no header
        df[["spanish", "english"]].to_csv(path, index=False, header=False, quoting=csv.QUOTE_ALL)
    else:
        df.to_csv(path, index=False, quoting=csv.QUOTE_ALL)
    return str(path)


def build_ui():
    models = list(available_models())
    preferred = next((m for m in models if "qwen2.5:14b" in m), models[0] if models else None)

    with gr.Blocks(title="Spanish synthetic data generator") as ui:
        gr.Markdown("## 🇪🇸 Spanish synthetic data generator\nDescribe a dataset; an open-source model builds it.")
        with gr.Row():
            kind = gr.Dropdown(list(DATASETS), value="flashcards", label="Dataset type")
            level = gr.Dropdown(LEVELS, value="A2", label="CEFR level")
            rows = gr.Slider(5, MAX_ROWS, value=20, step=5, label="Rows")
        topic = gr.Textbox(value="ordering food in a restaurant", label="Topic")
        with gr.Row():
            model = gr.Dropdown(models, value=preferred, label="Model")
            temperature = gr.Slider(0.0, 1.5, value=0.8, step=0.1, label="Temperature (higher = more varied)")
        go = gr.Button("Generate", variant="primary")
        table = gr.Dataframe(label="Preview", wrap=True)
        log = gr.Textbox(label="Log", lines=4)
        with gr.Row():
            fmt = gr.Radio(["csv", "jsonl"], value="csv", label="Format")
            download = gr.Button("Export")
        file = gr.File(label="Download")

        go.click(generate, [kind, topic, level, rows, model, temperature], [table, log])
        download.click(export, [table, kind, topic, fmt], file)
    return ui


if __name__ == "__main__":
    build_ui().launch(inbrowser=True)
