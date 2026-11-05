"""Week 6 exercise: "El Precio Justo" - the Spanish tutor plays The Price is Right against this week's model.

Weeks 1-5 built a tutor that explains Spanish, saves Flashcards.com decks and answers from its notes. Week 6 built
a model that prices a product from its description. Here they meet in a game for practising shopping vocabulary
and, above all, numbers - the thing learners stumble over at a real market stall:

1. A product from the test set (never seen in training) is rewritten in simple Spanish at the learner's level by
   qwen2.5:14b, with a handful of key words pulled out as flashcards.
2. The learner guesses the price - ideally in words ("ciento veinte dólares"), digits are accepted too.
3. The guess is read back deterministically (numeros.py), so the learner sees how their number is written or
   said; then the real price is revealed in words, next to the fine-tuned pricer's guess. Closest wins the round.
4. Every round's words, and the price itself as a number card, can be saved as a deck with the week 2 tutor's
   save_flashcards (Spanish front, English back, articles on nouns).

The opponent is the LoRA adapter trained in local_labs.ipynb (`python local_pricer.py` makes it). Without it the
game falls back to day 3's bag-of-words linear regression, which trains in seconds.

Run the game:   python week6/precio_justo.py
"""

import random
import sys
import threading
from pathlib import Path

import gradio as gr
from pydantic import BaseModel, Field

from local_pricer import ADAPTER_DIR, LoraPricer, chat, load_items, ollama_running
from numeros import NotANumber, numero_a_palabras, palabras_a_numero, precio_en_palabras

HERE = Path(__file__).parent
DECK_DIR = HERE.parent / "week2" / "decks"

sys.path.insert(0, str(HERE.parent / "week2"))
from tutor_tools import MAX_CARDS, save_flashcards  # noqa: E402

LEVELS = {
    "A2": "short, simple sentences in the present tense, using only very common words",
    "B1": "clear everyday Spanish; the present and the preterite; common words",
    "B2": "natural Spanish as a shop assistant would write it",
}


class Palabra(BaseModel):
    spanish: str = Field(description="A Spanish word or short phrase from the text; nouns with el/la/los/las")
    english: str = Field(description="Its English meaning")


class Producto(BaseModel):
    titulo: str = Field(description="The product name in Spanish, short; keep the brand")
    descripcion: str = Field(description="Two or three Spanish sentences describing the product, with no prices")
    vocabulario: list[Palabra] = Field(description="4 to 6 useful words from the Spanish text for flashcards")


PROMPT = """You are a Spanish tutor making a reading exercise for an English-speaking learner at CEFR level {level}.
Rewrite the product below in Spanish: {style}.
Never mention or hint at a price, a discount or a currency.
Then list 4 to 6 useful words that appear in your Spanish text. Every noun gets its definite article (el, la, los,
las), e.g. "la batería" -> "the battery". Prefer words for shopping and everyday objects over brand names.

Product:
{summary}"""


# ---------------------------------------------------------------------------------------------------------------
# The opponent


class Opponent:
    """The pricer the learner plays against, loaded in the background so the page opens straight away."""

    def __init__(self, train):
        self.name = "loading..."
        self.predict = None
        self.ready = threading.Event()
        threading.Thread(target=self._load, args=(train,), daemon=True).start()

    def _load(self, train):
        try:
            if (ADAPTER_DIR / "adapter_config.json").exists():
                pricer = LoraPricer()
                pricer.load()
                self.predict, self.name = pricer.predict, "Qwen2.5-0.5B + LoRA (fine-tuned)"
            else:
                self.predict, self.name = bag_of_words_pricer(train), "bag-of-words linear regression"
        finally:
            self.ready.set()

    def guess(self, item):
        self.ready.wait()
        return max(0.0, float(self.predict(item)))


def bag_of_words_pricer(train):
    """Day 3's natural-language linear regression: a fallback that needs no GPU and no adapter."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LinearRegression

    vectorizer = CountVectorizer(max_features=2000, stop_words="english")
    regressor = LinearRegression().fit(vectorizer.fit_transform([i.summary for i in train]), [i.price for i in train])
    return lambda item: regressor.predict(vectorizer.transform([item.summary]))[0]


# ---------------------------------------------------------------------------------------------------------------
# One round


def present(item, level):
    """The product as a Spanish reading card."""
    reply = chat(
        [{"role": "user", "content": PROMPT.format(level=level, style=LEVELS[level], summary=item.summary)}],
        schema=Producto,
        temperature=0.3,
    )
    reply.vocabulario = reply.vocabulario[:6]
    return reply


def judge(guess_text, item, model_guess):
    """Grade a guess. Returns (markdown feedback, learner won?, flashcards from this round)."""
    guess, in_words = palabras_a_numero(guess_text)
    price = item.price
    learner_error, model_error = abs(guess - price), abs(model_guess - price)

    lines = []
    said = precio_en_palabras(guess)
    if not in_words:
        lines.append(f"Tu respuesta, **{guess_text.strip()}**, se dice: *{said}*.")
    elif _normalise(guess_text) != _normalise(said) and _normalise(guess_text) != _normalise(numero_a_palabras(round(guess))):
        lines.append(f"Te entendí **{guess:,.2f} $** - se escribe: *{said}*.")
    else:
        lines.append(f"¡Bien escrito! *{said}*")

    lines.append(f"\n### Cuesta **{price:,.2f} $**\n*{precio_en_palabras(price)}*")
    lines.append(f"\n| | precio | error |\n|---|---:|---:|")
    lines.append(f"| Tú | {guess:,.2f} $ | {learner_error:,.2f} $ |")
    lines.append(f"| El modelo | {model_guess:,.2f} $ | {model_error:,.2f} $ |")
    won = learner_error <= model_error
    lines.append("\n**¡Ganaste esta ronda!** 🎉" if won else "\n**Esta vez gana el modelo.** 🤖")

    dollars = round(price)
    cards = [{"spanish": f"{numero_a_palabras(dollars, before_noun=True)} dólares ({dollars} $)",
              "english": f"{dollars} dollars"}]  # fmt: skip
    return "\n".join(lines), won, cards


def _normalise(text):
    drop = {"dólares", "dolares", "dólar", "dolar", "centavos", "centavo"}
    return " ".join(w for w in text.lower().split() if w not in drop)


# ---------------------------------------------------------------------------------------------------------------
# The app


def build_ui():
    train, _, test = load_items()
    opponent = Opponent(train)
    rng = random.Random()

    def new_round(level, state):
        if not ollama_running():
            return state, "**Ollama is not running.** Start it with `ollama serve` and try again.", "", gr.update()
        item = test[rng.randrange(len(test))]
        card = present(item, level)
        state = {**state, "item": item.model_dump(), "card": card.model_dump(), "answered": False}
        words = "\n".join(f"- **{w.spanish}** - {w.english}" for w in card.vocabulario)
        text = f"## {card.titulo}\n\n{card.descripcion}\n\n<details><summary>Vocabulario</summary>\n\n{words}\n</details>"
        return state, text, "", gr.update(value="")

    def guess(guess_text, state):
        if not state.get("item"):
            return state, "Pulsa **Nuevo producto** primero.", score_text(state)
        if state.get("answered"):
            return state, "Ya adivinaste este producto - pulsa **Nuevo producto**.", score_text(state)
        from pricer.items import Item

        item = Item.model_validate(state["item"])
        try:
            feedback, won, cards = judge(guess_text, item, opponent.guess(item))
        except NotANumber as error:
            return state, f"No entiendo **{error}** como número. Prueba con *ciento veinte* o *120*.", score_text(state)
        vocabulary = [{"spanish": w["spanish"], "english": w["english"]} for w in state["card"]["vocabulario"]]
        state = {
            **state,
            "answered": True,
            "rounds": state["rounds"] + 1,
            "wins": state["wins"] + won,
            "cards": state["cards"] + cards + vocabulary,
        }
        feedback += f"\n\n<details><summary>The original (English)</summary>\n\n{item.summary}\n</details>"
        return state, feedback, score_text(state)

    def score_text(state):
        return (
            f"**Tú {state['wins']} - {state['rounds'] - state['wins']} el modelo** · "
            f"{len(state['cards'])} tarjetas · rival: {opponent.name}"
        )

    def save(state):
        cards = list({c["spanish"]: c for c in state["cards"]}.values())[:MAX_CARDS]
        if not cards:
            return "Play a round first - there are no cards yet.", None
        message = save_flashcards("El precio justo", cards)
        return message, str(DECK_DIR / "El_precio_justo.csv")

    with gr.Blocks(title="El Precio Justo", theme=gr.themes.Soft()) as ui:
        gr.Markdown(
            "# 🛒 El Precio Justo\nRead the product in Spanish and guess its price - in words if you can. "
            "You play against this week's pricing model."
        )
        state = gr.State({"item": None, "card": None, "answered": False, "rounds": 0, "wins": 0, "cards": []})
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    level = gr.Radio(list(LEVELS), value="A2", label="Level", scale=2)
                    new = gr.Button("Nuevo producto", variant="primary", scale=1)
                product = gr.Markdown("*Pulsa **Nuevo producto** para empezar.*", container=True, min_height=220)
                with gr.Row():
                    answer = gr.Textbox(placeholder="p. ej. ciento veinte dólares", show_label=False, scale=3)
                    submit = gr.Button("Adivinar", scale=1)
                result = gr.Markdown()
            with gr.Column(scale=2):
                score = gr.Markdown("**Tú 0 - 0 el modelo**")
                save_button = gr.Button("Guardar tarjetas")
                save_status = gr.Markdown()
                deck_file = gr.File(label="Flashcards.com CSV")

        new.click(new_round, [level, state], [state, product, result, answer])
        submit.click(guess, [answer, state], [state, result, score])
        answer.submit(guess, [answer, state], [state, result, score])
        save_button.click(save, [state], [save_status, deck_file])
    return ui


if __name__ == "__main__":
    build_ui().launch(inbrowser=True, allowed_paths=[str(DECK_DIR)])
