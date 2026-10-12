"""Week 5 exercise: the Spanish tutor becomes an expert knowledge worker.

Weeks 1-4 built a tutor that explains Spanish and saves Flashcards.com decks. Here it answers from a knowledge base
instead of from memory: the grammar and vocabulary notes in `spanish-kb/`, plus any decks the week 2 tutor saved to
`week2/decks/` (so "which kitchen words have I studied?" has a real answer). Every answer shows the notes it used,
and the notes behind the latest answer can be turned into a new deck with one click.

The pipeline is the one the week builds up to, on local models (see local_rag.py):
- chunks are the notes' `##` sections, each prefixed with its note's title - a deterministic version of day 5's
  "headline + summary" pre-processing that costs no LLM calls
- BGE embeddings in Chroma, query rewriting so follow-ups like "and with estar?" find the right note, and a
  cross-encoder reranker
- an evaluation set (`spanish_tests.jsonl`) scored with day 4's retrieval metrics and LLM judge

Run the app:          python week5/spanish_tutor_rag.py
Run the evaluation:   python week5/spanish_tutor_rag.py --eval
"""

import csv
import sys
from pathlib import Path

import gradio as gr
from pydantic import BaseModel, Field

from local_rag import (
    DB_PATH, RAG, Result, answer_report, build_store, fetch_documents, format_context, llm, load_tests,
    ollama_running, retrieval_report,
)

HERE = Path(__file__).parent
KNOWLEDGE_BASE = HERE / "spanish-kb"
DECK_DIR = HERE.parent / "week2" / "decks"
TESTS = HERE / "spanish_tests.jsonl"
COLLECTION = "spanish-tutor"

sys.path.insert(0, str(HERE.parent / "week2"))
from tutor_tools import MAX_CARDS, latest_deck, save_flashcards  # noqa: E402

SYSTEM_PROMPT = """You are an expert Spanish tutor for an English-speaking learner.
Explain clearly and briefly, and keep every answer practical: a learner should be able to use what you say in a
conversation the same day.

Rules you always follow:
- Base your explanation on the tutor's notes below. Mention the note you used, e.g. (see: Por vs para).
- If the notes don't cover the question, say "My notes don't cover this" and then give your best short answer.
- Whenever you write a Spanish noun, include its definite article: el, la, los or las.
- Give one or two Spanish example sentences with their English translation.
- Answer in English unless the learner writes in Spanish or asks for Spanish.

The tutor's notes that matched the learner's question:
{context}
"""


# ---------------------------------------------------------------------------------------------------------------
# Knowledge base


def note_sections(document):
    """Split a note at its `##` headings; each section keeps the note's title so it can be found on its own."""
    text = document["text"].strip()
    title = text.splitlines()[0].lstrip("# ").strip()
    body = text.split("\n", 1)[1] if "\n" in text else ""
    intro, *sections = body.split("\n## ")
    pieces = [f"{title}\n\n{intro.strip()}"] if intro.strip() else []
    for section in sections:
        heading, _, content = section.partition("\n")
        pieces.append(f"{title} - {heading.strip()}\n\n{content.strip()}")
    metadata = {"source": document["source"], "type": document["type"], "title": title}
    return [Result(page_content=piece, metadata=metadata) for piece in pieces]


def deck_documents(deck_dir=DECK_DIR):
    """The learner's saved decks as chunks of up to 25 cards each."""
    chunks = []
    for path in sorted(Path(deck_dir).glob("*.csv")):
        with open(path, encoding="utf-8", newline="") as f:
            cards = [row for row in csv.reader(f) if len(row) >= 2]
        name = path.stem.replace("_", " ")
        for start in range(0, len(cards), 25):
            lines = "\n".join(f"- {spanish} = {english}" for spanish, english, *_ in cards[start:start + 25])
            chunks.append(Result(
                page_content=f"Flashcard deck the learner has studied: {name}\n\n{lines}",
                metadata={"source": f"week2/decks/{path.name}", "type": "decks", "title": f"Deck: {name}"},
            ))
    return chunks


def tutor_chunks():
    chunks = [chunk for document in fetch_documents(KNOWLEDGE_BASE) for chunk in note_sections(document)]
    return chunks + deck_documents()


def tutor_store(rebuild=False):
    return build_store(COLLECTION, tutor_chunks(), encoder="bge", path=DB_PATH, rebuild=rebuild)


def tutor_rag(store=None, rewrite=True, rerank="cross-encoder"):
    return RAG(store or tutor_store(), rewrite=rewrite, rerank=rerank, retrieval_k=10, final_k=5,
               system_prompt=SYSTEM_PROMPT)


# ---------------------------------------------------------------------------------------------------------------
# Flashcards from the notes


class Card(BaseModel):
    spanish: str = Field(description="The Spanish side: a word with its definite article, or a short example sentence")
    english: str = Field(description="The English translation")


class Deck(BaseModel):
    name: str = Field(description="A short name for the deck, e.g. 'Por vs para'")
    cards: list[Card]


def deck_from_notes(chunks, count=12):
    """Write flashcards from the retrieved notes and save them with the week 2 tutor's CSV tool."""
    notes = "\n\n".join(chunk.page_content for chunk in chunks)
    prompt = f"""Make {count} flashcards that drill what these Spanish tutor notes teach.
Spanish on the front, English on the back. Prefer the notes' own example sentences and word pairs.
Every Spanish noun on its own must carry its definite article (la manzana, el problema).
Name the deck after the topic of the notes.

{notes}"""
    deck = llm([{"role": "user", "content": prompt}], schema=Deck, temperature=0.3)
    cards = [card.model_dump() for card in deck.cards[:MAX_CARDS]]
    return save_flashcards(deck.name, cards), latest_deck()


# ---------------------------------------------------------------------------------------------------------------
# App


WELCOME = (
    "¡Hola! Ask me about Spanish grammar - *ser or estar?*, *why el agua?*, *por or para for a deadline?* - "
    "and I'll answer from my notes. Then **Make a deck** turns the notes behind my answer into flashcards."
)


def build_ui():
    rag = tutor_rag()

    def put_message_in_chatbot(message, history):
        return "", history + [{"role": "user", "content": message}]

    def chat(history):
        answer, chunks = rag.answer_question(history[-1]["content"], history[:-1])
        history = history + [{"role": "assistant", "content": answer}]
        query_note = f"*Searched for:* {rag.last_query}\n\n" if rag.last_query else ""
        return history, query_note + format_context(chunks), chunks

    def make_deck(chunks):
        if not chunks:
            return "Ask a question first - the deck is made from the notes behind the answer.", None
        message, path = deck_from_notes(chunks)
        return message, str(path) if path else None

    with gr.Blocks(title="Spanish Tutor - with notes", theme=gr.themes.Soft()) as ui:
        gr.Markdown("# 🇪🇸 Spanish Tutor, now with notes\nAnswers come from the tutor's grammar notes and your saved decks.")
        last_chunks = gr.State([])
        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(value=[{"role": "assistant", "content": WELCOME}], type="messages", height=560)
                message = gr.Textbox(placeholder="e.g. When do I use the subjunctive?", show_label=False)
                with gr.Row():
                    deck_button = gr.Button("Make a deck from these notes")
                    clear = gr.Button("New conversation")
                deck_status = gr.Markdown()
                deck_file = gr.File(label="Flashcards.com CSV")
            with gr.Column(scale=1):
                context = gr.Markdown("*The notes behind each answer will appear here*", container=True, height=700)

        message.submit(put_message_in_chatbot, [message, chatbot], [message, chatbot]).then(
            chat, chatbot, [chatbot, context, last_chunks]
        )
        deck_button.click(make_deck, last_chunks, [deck_status, deck_file])
        clear.click(lambda: ([{"role": "assistant", "content": WELCOME}], "", [], "", None), None,
                    [chatbot, context, last_chunks, deck_status, deck_file])
    return ui


def launch(**kwargs):
    """Start the app. Decks are saved in week2/decks, so Gradio has to be allowed to serve files from there."""
    DECK_DIR.mkdir(parents=True, exist_ok=True)
    return build_ui().launch(allowed_paths=[str(DECK_DIR)], **kwargs)


# ---------------------------------------------------------------------------------------------------------------
# Evaluation


CONFIGS = {
    "plain vector search": dict(rewrite=False, rerank=None),
    "+ query rewrite": dict(rewrite=True, rerank=None),
    "+ rewrite + cross-encoder rerank": dict(rewrite=True, rerank="cross-encoder"),
}


def compare_retrieval(tests=None):
    """Retrieval metrics for each configuration, averaged over the tutor's test questions."""
    import pandas as pd

    tests = tests or load_tests(TESTS)
    store = tutor_store()
    rows = []
    for label, config in CONFIGS.items():
        report = retrieval_report(tutor_rag(store, **config), tests, k=5)
        rows.append({"pipeline": label, **report[["mrr", "ndcg", "coverage"]].mean().round(3).to_dict()})
    return pd.DataFrame(rows)


def evaluate_answers(tests=None, **config):
    return answer_report(tutor_rag(**config), tests or load_tests(TESTS))


if __name__ == "__main__":
    if not ollama_running():
        sys.exit("Start Ollama first: ollama serve")
    if "--eval" in sys.argv:
        print(compare_retrieval().to_string(index=False))
        answers = evaluate_answers()
        print(answers.groupby("category")[["accuracy", "completeness", "relevance"]].mean().round(2))
        print(answers[["accuracy", "completeness", "relevance"]].mean().round(2))
    else:
        launch(inbrowser=True)
