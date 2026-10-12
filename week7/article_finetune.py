"""
Week 7 exercise for the Spanish tutor: fine-tune a model that knows which article a noun takes.

Every flashcard the tutor makes must put the definite article on a noun ("la carpeta", "el éxito"),
and that is exactly where small local models slip: el problema, el día, el agua, la mano, la foto.
This module turns the week 7 recipe from prices to articles:

  data      AnCora (Universal Dependencies, CC BY 4.0): 500k words of hand-annotated Spanish news.
            Every "el"/"la" directly in front of a singular noun is a vote; a noun's label is its majority.
  baseline  the classroom rule (-a -> la, otherwise el), the base Llama 3.2 3B, and the tutor's
            own Ollama models (llama3.2, qwen2.5:14b)
  model     Llama 3.2 3B in 4 bits with LoRA adapters, trained with local_finetune.py (QLoRA on MLX)

    uv run week7/article_finetune.py train
    uv run week7/article_finetune.py check week2/decks/False_Friends__Falsos_Amigos_.csv
"""

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import mlx.core as mx
import requests

import local_finetune as lf

DATA_DIR = lf.WEEK7 / "data"
ANCORA_DIR = DATA_DIR / "ancora"
ANCORA_URL = "https://raw.githubusercontent.com/UniversalDependencies/UD_Spanish-AnCora/master/es_ancora-ud-{}.conllu"
DATASET_FILE = DATA_DIR / "articles.json"

MIN_VOTES = 2  # a noun needs at least this many article sightings
MIN_AGREEMENT = 0.9  # and this share for the winning article; the rest are ambiguous (el/la capital)
ARTICLES = ("el", "la")
QUESTION = "Which definite article does this Spanish noun take?"
PREFIX = "Article:"

CONFIG = lf.Config(
    "articles",
    epochs=2,
    batch_size=16,
    max_seq_length=64,
    lora_r=16,
    lora_alpha=32,
    learning_rate=2e-4,
    warmup_ratio=0.05,
)

OLLAMA_CHAT = "http://localhost:11434/api/chat"


# ---------- data ----------


def download_ancora() -> list[Path]:
    ANCORA_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for split in ("train", "dev", "test"):
        path = ANCORA_DIR / f"es_ancora-ud-{split}.conllu"
        if not path.exists():
            response = requests.get(ANCORA_URL.format(split), timeout=300)
            response.raise_for_status()
            path.write_bytes(response.content)
        paths.append(path)
    return paths


def sentences(path: Path):
    """Yield each sentence as a list of word rows (id, form, lemma, upos, feats, head); multiword tokens skipped."""
    sentence = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if sentence:
                    yield sentence
                sentence = []
            elif not line.startswith("#"):
                cols = line.split("\t")
                if cols[0].isdigit():
                    sentence.append((int(cols[0]), cols[1], cols[2], cols[3], cols[5], cols[6]))
    if sentence:
        yield sentence


def article_votes(paths: list[Path]) -> dict[str, Counter]:
    """Count el/la directly in front of each singular common noun that the article modifies."""
    votes = defaultdict(Counter)
    for path in paths:
        for sentence in sentences(path):
            for (i, form, lemma, upos, feats, head), nxt in zip(sentence, sentence[1:]):
                article = form.lower()
                if lemma != "el" or upos != "DET" or article not in ARTICLES:
                    continue
                n_id, n_form, _, n_upos, n_feats, _ = nxt
                noun = n_form.lower()
                if n_upos == "NOUN" and head == str(n_id) and "Number=Sing" in n_feats and noun.isalpha():
                    votes[noun][article] += 1
    return votes


def split_of(noun: str) -> str:
    """A stable 80/10/10 split by noun, so no test noun is ever seen in training."""
    bucket = int(hashlib.md5(noun.encode()).hexdigest(), 16) % 10
    return "test" if bucket == 0 else "val" if bucket == 1 else "train"


def build_dataset() -> dict:
    votes = article_votes(download_ancora())
    data = {"train": [], "val": [], "test": []}
    ambiguous = {}
    for noun, counts in sorted(votes.items()):
        total = sum(counts.values())
        article, top = counts.most_common(1)[0]
        if total < MIN_VOTES:
            continue
        if top / total < MIN_AGREEMENT:
            ambiguous[noun] = dict(counts)
            continue
        data[split_of(noun)].append({"noun": noun, "article": article, "votes": total})
    data["ambiguous"] = ambiguous
    DATASET_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=0))
    return data


def load_data() -> dict:
    if not DATASET_FILE.exists():
        return build_dataset()
    return json.loads(DATASET_FILE.read_text())


def to_prompt(noun: str) -> str:
    return f"{QUESTION}\n\nNoun: {noun}\n\n{PREFIX}"


def to_datapoint(row: dict) -> dict:
    # " el" with its leading space is a single Llama token, just like the price after "Price is $"
    return {"prompt": to_prompt(row["noun"]), "completion": " " + row["article"]}


# ---------- predictors: noun -> "el" / "la" ----------


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def classroom_rule(noun: str) -> str:
    """What a beginner learns first: nouns in -a are feminine, everything else masculine."""
    return "la" if noun.endswith("a") else "el"


def better_rule(noun: str) -> str:
    """The rule plus the endings textbooks teach next (-ión, -dad, -umbre ... and -ema, -aje, -or)."""
    plain = strip_accents(noun)
    if re.search(r"(ion|dad|tad|tud|umbre|ie|is)$", plain):
        return "la"
    if re.search(r"(ema|grama|aje|or)$", plain):
        return "el"
    return classroom_rule(noun)


def make_llama_classifier(model, tokenizer):
    """Compare the model's probability of ' el' and ' la' after 'Article:' - no free-form generation."""
    ids = [tokenizer.encode(" " + a, add_special_tokens=False)[0] for a in ARTICLES]

    def predict(noun: str) -> str:
        tokens = mx.array(tokenizer.encode(to_prompt(noun)))[None]
        logits = model(tokens)[0, -1, mx.array(ids)]
        return ARTICLES[mx.argmax(logits).item()]

    return predict


def make_ollama_classifier(model: str):
    """Ask one of the tutor's chat models the way the tutor uses them: write the card's front side."""

    def predict(noun: str) -> str:
        messages = [
            {"role": "system", "content": "You are a Spanish teacher writing flashcards."},
            {
                "role": "user",
                "content": f'Write the Spanish noun "{noun}" with its singular definite article (el or la), '
                'as in "el libro" or "la casa". Reply with only those two words.',
            },
        ]
        response = requests.post(
            OLLAMA_CHAT,
            json={"model": model, "messages": messages, "stream": False, "options": {"temperature": 0}},
            timeout=120,
        )
        response.raise_for_status()
        words = re.findall(r"\b(el|la)\b", response.json()["message"]["content"].lower())
        return words[0] if words else "?"

    return predict


def score(predict, rows: list[dict]) -> dict:
    """Accuracy overall and on the irregular nouns that break even the textbook rules - the ones worth a flashcard."""
    guesses = [predict(row["noun"]) for row in rows]
    right = [g == row["article"] for g, row in zip(guesses, rows)]
    hard = [ok for ok, row in zip(right, rows) if better_rule(row["noun"]) != row["article"]]
    mistakes = [f"{g} {row['noun']}" for g, row, ok in zip(guesses, rows, right) if not ok]
    return {
        "accuracy": sum(right) / len(right),
        "hard_accuracy": sum(hard) / len(hard),
        "hard_count": len(hard),
        "mistakes": mistakes,
    }


# ---------- training and the tutor hook ----------


def train() -> Path:
    data = load_data()
    prompts = {split: [to_datapoint(row) for row in data[split]] for split in ("train", "val")}
    return lf.run_training(CONFIG, prompts)


def check_deck(path: Path, predict) -> list[str]:
    """Flag flashcards whose article disagrees with the fine-tuned model."""
    problems = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            match = re.fullmatch(r"(el|la) (\w+)", row[0].strip().lower())
            if match and predict(match[2]) != match[1]:
                problems.append(f"{row[0]!r}: the model expects {predict(match[2])} {match[2]}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["data", "train", "check"])
    parser.add_argument("deck", nargs="?", type=Path)
    args = parser.parse_args()

    if args.action == "data":
        data = build_dataset()
        print({split: len(rows) for split, rows in data.items()})
    elif args.action == "train":
        train()
    else:
        model, tokenizer = lf.load_finetuned(CONFIG.name)
        problems = check_deck(args.deck, make_llama_classifier(model, tokenizer))
        print("\n".join(problems) or "Every article in the deck matches the model.")


if __name__ == "__main__":
    main()
