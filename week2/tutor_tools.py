"""Deterministic tools the Spanish tutor can call: verb conjugation and flashcard decks."""

import csv
import io
from pathlib import Path

DECK_DIR = Path(__file__).parent / "decks"
MAX_CARDS = 50

_latest_deck = None


def latest_deck():
    """The CSV written by the most recent successful save_flashcards call, or None."""
    return _latest_deck

PRONOUNS = ["yo", "tú", "él/ella/usted", "nosotros", "vosotros", "ellos/ellas/ustedes"]

REGULAR_ENDINGS = {
    "present": {
        "ar": ["o", "as", "a", "amos", "áis", "an"],
        "er": ["o", "es", "e", "emos", "éis", "en"],
        "ir": ["o", "es", "e", "imos", "ís", "en"],
    },
    "preterite": {
        "ar": ["é", "aste", "ó", "amos", "asteis", "aron"],
        "er": ["í", "iste", "ió", "imos", "isteis", "ieron"],
        "ir": ["í", "iste", "ió", "imos", "isteis", "ieron"],
    },
    "imperfect": {
        "ar": ["aba", "abas", "aba", "ábamos", "abais", "aban"],
        "er": ["ía", "ías", "ía", "íamos", "íais", "ían"],
        "ir": ["ía", "ías", "ía", "íamos", "íais", "ían"],
    },
}

# Future and conditional add their endings to the whole infinitive, not to the stem
FUTURE_ENDINGS = ["é", "ás", "á", "emos", "éis", "án"]
CONDITIONAL_ENDINGS = ["ía", "ías", "ía", "íamos", "íais", "ían"]

# The handful of verbs that break the rules above and that a beginner meets first
IRREGULAR_STEMS = {
    "tener": "tendr", "hacer": "har", "poder": "podr", "poner": "pondr",
    "salir": "saldr", "venir": "vendr", "decir": "dir", "haber": "habr",
    "saber": "sabr", "querer": "querr",
}

IRREGULAR_FORMS = {
    "ser": {
        "present": ["soy", "eres", "es", "somos", "sois", "son"],
        "preterite": ["fui", "fuiste", "fue", "fuimos", "fuisteis", "fueron"],
        "imperfect": ["era", "eras", "era", "éramos", "erais", "eran"],
    },
    "estar": {
        "present": ["estoy", "estás", "está", "estamos", "estáis", "están"],
        "preterite": ["estuve", "estuviste", "estuvo", "estuvimos", "estuvisteis", "estuvieron"],
    },
    "ir": {
        "present": ["voy", "vas", "va", "vamos", "vais", "van"],
        "preterite": ["fui", "fuiste", "fue", "fuimos", "fuisteis", "fueron"],
        "imperfect": ["iba", "ibas", "iba", "íbamos", "ibais", "iban"],
    },
    "tener": {
        "present": ["tengo", "tienes", "tiene", "tenemos", "tenéis", "tienen"],
        "preterite": ["tuve", "tuviste", "tuvo", "tuvimos", "tuvisteis", "tuvieron"],
    },
    "hacer": {
        "present": ["hago", "haces", "hace", "hacemos", "hacéis", "hacen"],
        "preterite": ["hice", "hiciste", "hizo", "hicimos", "hicisteis", "hicieron"],
    },
    "decir": {
        "present": ["digo", "dices", "dice", "decimos", "decís", "dicen"],
        "preterite": ["dije", "dijiste", "dijo", "dijimos", "dijisteis", "dijeron"],
    },
    "poder": {
        "present": ["puedo", "puedes", "puede", "podemos", "podéis", "pueden"],
        "preterite": ["pude", "pudiste", "pudo", "pudimos", "pudisteis", "pudieron"],
    },
    "querer": {
        "present": ["quiero", "quieres", "quiere", "queremos", "queréis", "quieren"],
        "preterite": ["quise", "quisiste", "quiso", "quisimos", "quisisteis", "quisieron"],
    },
    "ver": {
        "present": ["veo", "ves", "ve", "vemos", "veis", "ven"],
        "preterite": ["vi", "viste", "vio", "vimos", "visteis", "vieron"],
        "imperfect": ["veía", "veías", "veía", "veíamos", "veíais", "veían"],
    },
    "dar": {
        "present": ["doy", "das", "da", "damos", "dais", "dan"],
        "preterite": ["di", "diste", "dio", "dimos", "disteis", "dieron"],
    },
}

TENSES = ["present", "preterite", "imperfect", "future", "conditional"]


def conjugate_verb(infinitive, tense="present"):
    """Conjugate a Spanish verb. Returns the six forms as a readable table."""
    print(f"TOOL CALLED: conjugate_verb({infinitive!r}, {tense!r})", flush=True)
    verb = infinitive.strip().lower()
    tense = tense.strip().lower()

    if tense not in TENSES:
        return f"I can only conjugate these tenses: {', '.join(TENSES)}."
    if len(verb) < 3 or verb[-2:] not in ("ar", "er", "ir"):
        return f"'{infinitive}' does not look like a Spanish infinitive ending in -ar, -er or -ir."

    forms = IRREGULAR_FORMS.get(verb, {}).get(tense)
    note = "irregular forms"

    if not forms:
        stem, ending_type = verb[:-2], verb[-2:]
        if tense in ("future", "conditional"):
            endings = FUTURE_ENDINGS if tense == "future" else CONDITIONAL_ENDINGS
            irregular_stem = IRREGULAR_STEMS.get(verb)
            forms = [(irregular_stem or verb) + e for e in endings]
            note = "irregular stem" if irregular_stem else "regular pattern"
        else:
            forms = [stem + e for e in REGULAR_ENDINGS[tense][ending_type]]
            note = "regular pattern"

    table = "\n".join(f"{pronoun}: {form}" for pronoun, form in zip(PRONOUNS, forms))
    result = f"{verb} - {tense} ({note})\n{table}"
    if note == "regular pattern" and tense == "present":
        result += "\nThis tool applies the regular pattern; stem-changing verbs such as pensar are not in its list."
    return result


def save_flashcards(deck_name, cards):
    """Write cards to a CSV that can be imported into Flashcards.com. Spanish front, English back."""
    print(f"TOOL CALLED: save_flashcards({deck_name!r}, {len(cards)} cards)", flush=True)
    global _latest_deck
    _latest_deck = None

    if not cards:
        return "No cards were provided, so nothing was saved."
    if len(cards) > MAX_CARDS:
        return f"That is {len(cards)} cards; please send at most {MAX_CARDS} in one deck."

    rows = []
    for card in cards:
        spanish = str(card.get("spanish", "")).strip()
        english = str(card.get("english", "")).strip()
        if spanish and english:
            rows.append([spanish, english])
    if not rows:
        return "Every card was missing a Spanish or an English side, so nothing was saved."

    safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in deck_name).strip() or "deck"
    DECK_DIR.mkdir(exist_ok=True)
    path = DECK_DIR / f"{safe_name.replace(' ', '_')}.csv"

    buffer = io.StringIO(newline="")
    csv.writer(buffer, quoting=csv.QUOTE_ALL, lineterminator="\n").writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8")

    _latest_deck = path

    return f"Saved {len(rows)} cards to {path.name}. The deck is ready to download and import."


# The JSON schemas that tell the LLM what it is allowed to call

CONJUGATE_SCHEMA = {
    "name": "conjugate_verb",
    "description": (
        "Conjugate a Spanish verb in a given tense. Always call this before showing a "
        "conjugation table, so the forms are correct rather than remembered."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "infinitive": {"type": "string", "description": "The infinitive, for example hablar or tener"},
            "tense": {
                "type": "string",
                "enum": TENSES,
                "description": "Which tense to conjugate; defaults to present",
            },
        },
        "required": ["infinitive"],
        "additionalProperties": False,
    },
}

FLASHCARD_SCHEMA = {
    "name": "save_flashcards",
    "description": (
        "Save a deck of Spanish/English flashcards as a CSV the learner can import into "
        f"Flashcards.com. Call this whenever the learner asks for flashcards or a deck. "
        f"At most {MAX_CARDS} cards, and Spanish nouns must carry their definite article."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "deck_name": {"type": "string", "description": "Short name for the deck, e.g. 'kitchen verbs'"},
            "cards": {
                "type": "array",
                "description": "The cards, Spanish on the front and English on the back",
                "items": {
                    "type": "object",
                    "properties": {
                        "spanish": {"type": "string", "description": "Spanish side, with el/la/los/las for nouns"},
                        "english": {"type": "string", "description": "English side"},
                    },
                    "required": ["spanish", "english"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["deck_name", "cards"],
        "additionalProperties": False,
    },
}

TOOLS = [
    {"type": "function", "function": CONJUGATE_SCHEMA},
    {"type": "function", "function": FLASHCARD_SCHEMA},
]

TOOL_FUNCTIONS = {"conjugate_verb": conjugate_verb, "save_flashcards": save_flashcards}
