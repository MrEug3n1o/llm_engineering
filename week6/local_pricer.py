"""Week 6's "Price is Right" project on a Mac with no API keys.

The course notebooks reach for paid services at every step: Groq batches rewrite the products on day 2,
GPT / Claude / Gemini / Grok guess prices on day 4, and day 5 fine-tunes gpt-4.1-nano through OpenAI. The
datasets themselves are public on the Hugging Face Hub, so only the model calls need replacing:

- Groq batch rewriting -> the same system prompt sent to a local Ollama model (concurrent requests, no batch API)
- frontier models -> llama3.2 and qwen2.5:14b on Ollama, called through the native /api/chat
- OpenAI fine-tuning -> a LoRA adapter on Qwen2.5-0.5B-Instruct, trained with peft on the Mac's GPU (MPS)

`evaluate` wraps the course's `pricer.evaluator.Tester` so the error numbers are computed exactly as in the
videos, but draws one static matplotlib chart instead of two interactive plotly ones (which bloat a saved
notebook by megabytes) and remembers every score for the closing results chart.
"""

import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
from sklearn.metrics import mean_squared_error, r2_score
from tqdm.auto import tqdm

from pricer.evaluator import Tester
from pricer.items import PREFIX, QUESTION, Item

HERE = Path(__file__).parent
MODELS_DIR = HERE / "models"

OLLAMA_API = "http://localhost:11434/api"
REQUEST_TIMEOUT = 600

# The day 2 prompt from pricer/batch.py (importing that module builds a Groq client, which fails without a key)
REWRITE_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = MODELS_DIR / "qwen-pricer"

# Every score from evaluate(), in order: title -> average absolute error in dollars
RESULTS = {}

# Ed's numbers from week6/results.ipynb (full dataset, frontier models through their APIs), for comparison
COURSE_RESULTS = {
    "Constant": 106.18,
    "Linear Regression": 101.56,
    "NLP + LR": 76.81,
    "Random Forest": 72.28,
    "XGBoost": 68.23,
    "Human (Ed)": 87.62,
    "Neural Network": 63.97,
    "GPT 4.1 Nano": 62.51,
    "Grok 4.1 Fast": 57.62,
    "Gemini 3 Pro": 50.54,
    "Claude 4.5 Sonnet": 47.10,
    "GPT 5.1": 44.74,
    "GPT 4.1 Nano (Fine-tuned)": 75.91,
    "Deep Neural Network": 46.49,
}


# ---------------------------------------------------------------------------------------------------------------
# Data


def load_items(lite=True):
    """Ed's curated and rewritten dataset: (train, val, test) lists of Items. Public, so no HF token is needed."""
    return Item.from_hub("ed-donner/items_lite" if lite else "ed-donner/items_full")


# ---------------------------------------------------------------------------------------------------------------
# Ollama


def ollama_running():
    try:
        return requests.get(f"{OLLAMA_API}/tags", timeout=2).ok
    except requests.RequestException:
        return False


def chat(messages, model="qwen2.5:14b", temperature=0, num_predict=None, schema=None):
    """One reply from a local Ollama model. A pydantic `schema` constrains the reply and parses it."""
    options = {"temperature": temperature}
    if num_predict:
        options["num_predict"] = num_predict
    payload = {"model": model, "messages": messages, "stream": False, "options": options}
    if schema is not None:
        payload["format"] = schema.model_json_schema()
    response = requests.post(f"{OLLAMA_API}/chat", json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    content = response.json()["message"]["content"]
    return schema.model_validate_json(content) if schema is not None else content


def unload(model):
    """Free the memory an Ollama model holds (it otherwise stays loaded for 5 minutes) before GPU training."""
    requests.post(f"{OLLAMA_API}/generate", json={"model": model, "keep_alive": 0}, timeout=60)


def rewrite(text, model="llama3.2"):
    """Day 2: turn a scraped product page into the five-line summary the rest of the week trains on."""
    return chat([{"role": "system", "content": REWRITE_PROMPT}, {"role": "user", "content": text}], model=model)


def rewrite_many(items, model="llama3.2", workers=4):
    """Rewrite `item.full` into `item.summary` for every item. Returns seconds per item.

    This stands in for the Groq batch API: requests run concurrently, and Ollama works through them as fast as
    the Mac allows (in parallel if the server was started with OLLAMA_NUM_PARALLEL > 1).
    """
    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        summaries = list(tqdm(pool.map(lambda item: rewrite(item.full, model), items), total=len(items)))
    for item, summary in zip(items, summaries):
        item.summary = summary
    return (time.time() - start) / max(len(items), 1)


def price_messages(item):
    """The day 4 prompt, word for word."""
    message = f"Estimate the price of this product. Respond with the price, no explanation\n\n{item.summary}"
    return [{"role": "user", "content": message}]


def llm_pricer(model):
    """A predictor for evaluate() that asks an Ollama model for the price, named after the model."""

    def predict(item):
        return chat(price_messages(item), model=model, num_predict=20)

    predict.__name__ = re.sub(r"[^0-9a-zA-Z]+", "_", model)
    return predict


# ---------------------------------------------------------------------------------------------------------------
# Evaluation


class StaticTester(Tester):
    """The course's Tester with a single matplotlib chart, and the score kept in RESULTS."""

    def report(self):
        average_error = sum(self.errors) / self.size
        mse = mean_squared_error(self.truths, self.guesses)
        r2 = r2_score(self.truths, self.guesses) * 100
        errors = np.array(self.errors)
        ci = 1.96 * errors.std() / math.sqrt(len(errors))
        RESULTS[self.title] = average_error

        colors = {"green": "#2a9d4b", "orange": "#e59a1a", "red": "#d43c3c"}
        fig, (scatter, trend) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.2, 1]})
        top = max(max(self.truths), max(self.guesses))
        scatter.scatter(self.truths, self.guesses, s=14, c=[colors[c] for c in self.colors])
        scatter.plot([0, top], [0, top], "--", color="deepskyblue", linewidth=1.5)
        scatter.set(xlim=(0, top), ylim=(0, top), xlabel="Actual price ($)", ylabel="Predicted price ($)")
        scatter.set_title(f"{self.title}: error ${average_error:,.2f}  MSE {mse:,.0f}  r² {r2:.1f}%")

        running = np.cumsum(errors) / np.arange(1, len(errors) + 1)
        trend.plot(np.arange(1, len(errors) + 1), running, color="firebrick", linewidth=2)
        trend.set(xlabel="Datapoints", ylabel="Average absolute error ($)", ylim=(0, None))
        trend.set_title(f"Error ${average_error:,.2f} ± ${ci:,.2f} (95% CI)")
        plt.tight_layout()
        plt.show()

    def run(self):
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for title, guess, truth, error, color in pool.map(self.run_datapoint, range(self.size)):
                self.titles.append(title)
                self.guesses.append(guess)
                self.truths.append(truth)
                self.errors.append(error)
                self.colors.append(color)
        self.report()
        return sum(self.errors) / self.size


def evaluate(predictor, data, size=200, workers=5, title=None):
    """Score `predictor` on the first `size` items of `data`, as the course does. Returns the average error."""
    return StaticTester(predictor, data, title=title, size=size, workers=workers).run()


def results_chart(results=None, reference=COURSE_RESULTS):
    """Side-by-side bars: the scores from this session against Ed's from the videos."""
    results = results or RESULTS
    fig, (mine, his) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for axis, scores, color, title in (
        (mine, results, "#4c72b0", "This Mac (lite dataset, local models)"),
        (his, reference, "#999999", "Course videos (full dataset, frontier APIs)"),
    ):
        names = list(scores)
        bars = axis.bar(names, [scores[n] for n in names], color=color)
        axis.bar_label(bars, fmt="%.0f", fontsize=8)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=60)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")
    mine.set_ylabel("Average absolute error on 200 test items ($)")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------------------------------------------
# Day 5: fine-tuning with LoRA instead of the OpenAI fine-tuning API


def device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def pricer_prompt(item):
    """The prompt the fine-tuned model sees: question, summary, then "Price is $" for it to complete."""
    return f"{QUESTION}\n\n{item.summary}\n\n{PREFIX}"


def pricer_completion(item):
    """What it learns to write after the prompt. The ".00" tells it where the number ends."""
    return f"{round(item.price)}.00"


class LoraPricer:
    """Qwen2.5-0.5B-Instruct with a LoRA adapter, trained to complete "Price is $" with the right number.

    Only the last few positions of each sequence produce logits: computing them for every position against a
    152k-token vocabulary is what makes a naive training step take a minute and a half on MPS instead of 1.5s.
    That works because sequences are left-padded, so the price is always at the very end.
    """

    MAX_LENGTH = 192
    LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def __init__(self, base_model=BASE_MODEL):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.base_model = base_model
        self.device = device()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32).to(self.device)
        self.model.eval()
        self.tuned = False

    # -- the untrained model, for a before/after comparison -------------------------------------------------

    def zero_shot(self, item):
        """Ask the base instruct model through its chat template, like the day 4 frontier calls."""
        text = self.tokenizer.apply_chat_template(price_messages(item), tokenize=False, add_generation_prompt=True)
        return self._generate([text], max_new_tokens=12)[0]

    # -- training --------------------------------------------------------------------------------------------

    def attach_adapter(self, rank=16, alpha=32, dropout=0.05):
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=self.LORA_TARGETS, task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, config)
        self.tuned = True
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"LoRA adapter: {trainable:,} trainable parameters of {total:,} ({trainable / total:.2%})")

    def _batch(self, items):
        """Tokenize prompt+completion, left-padded, with the completion length of each row."""
        texts = [pricer_prompt(item) + pricer_completion(item) for item in items]
        batch = self.tokenizer(
            texts, return_tensors="pt", padding="max_length", max_length=self.MAX_LENGTH, truncation=True
        ).to(self.device)
        lengths = [len(self.tokenizer(pricer_completion(item)).input_ids) for item in items]
        return batch, lengths

    def train(self, items, epochs=1, batch_size=16, learning_rate=2e-4, warmup=50, log_every=50):
        """Supervised fine-tuning, with the loss on the completion tokens only. Returns the loss history."""
        torch = self.torch
        if not self.tuned:
            self.attach_adapter()
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.01)
        steps = epochs * math.ceil(len(items) / batch_size)

        def schedule(step):  # linear warmup, then cosine decay to zero
            if step < warmup:
                return (step + 1) / warmup
            return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(steps - warmup, 1)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        keep = max(len(self.tokenizer(f"{round(item.price)}.00").input_ids) for item in items)
        rng = np.random.default_rng(42)
        history, recent = [], []
        self.model.train()
        progress = tqdm(total=steps, desc="LoRA training")
        for epoch in range(epochs):
            order = rng.permutation(len(items))
            for start in range(0, len(items), batch_size):
                batch, lengths = self._batch([items[i] for i in order[start : start + batch_size]])
                output = self.model(**batch, logits_to_keep=keep + 1)
                logits = output.logits[:, :-1].float()  # position t predicts token t+1
                targets = batch.input_ids[:, -keep:].clone()
                for row, length in enumerate(lengths):
                    targets[row, : keep - length] = -100  # ignore the prompt tokens that crept into the window
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                recent.append(loss.item())
                progress.update(1)
                if len(recent) == log_every:
                    history.append(sum(recent) / len(recent))
                    progress.set_postfix(loss=f"{history[-1]:.3f}")
                    recent = []
        progress.close()
        self.model.eval()
        return history

    def save(self, path=ADAPTER_DIR):
        self.model.save_pretrained(path)

    def load(self, path=ADAPTER_DIR):
        from peft import PeftModel

        self.model = PeftModel.from_pretrained(self.model, path).to(self.device)
        self.model.eval()
        self.tuned = True

    # -- inference -------------------------------------------------------------------------------------------

    def _generate(self, texts, max_new_tokens):
        with self.torch.no_grad():
            batch = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
            output = self.model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.batch_decode(output[:, batch.input_ids.shape[1] :], skip_special_tokens=True)

    def predict(self, item, beams=8):
        """The fine-tuned model's price for one item, as a float."""
        return self.predict_many([item], beams=beams)[0]

    def predict_many(self, items, beams=8, batch_size=8):
        """Prices for many items.

        Greedy decoding (beams=1) picks the likeliest digit one position at a time, which is not the likeliest -
        let alone the best - number. With beams > 1 the answer is the average of the top candidate prices,
        weighted by their probabilities: on the 200 test items that cut the error from $69 to $62.
        """
        torch = self.torch
        prices = []
        for start in range(0, len(items), batch_size):
            batch_items = items[start : start + batch_size]
            if beams == 1:
                replies = self._generate([pricer_prompt(i) for i in batch_items], max_new_tokens=5)
                prices.extend(_leading_number(reply) or 0.0 for reply in replies)
                continue
            batch = self.tokenizer([pricer_prompt(i) for i in batch_items], return_tensors="pt", padding=True)
            with torch.no_grad():
                output = self.model.generate(
                    **batch.to(self.device),
                    max_new_tokens=5,  # "123.0" - every candidate has the same length, so scores compare fairly
                    num_beams=beams,
                    num_return_sequences=beams,
                    do_sample=False,
                    length_penalty=0,  # scores are plain sums of log-probabilities
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            texts = self.tokenizer.batch_decode(output.sequences[:, batch.input_ids.shape[1] :], skip_special_tokens=True)
            scores = output.sequences_scores.view(len(batch_items), beams).float().cpu().numpy()
            for row in range(len(batch_items)):
                candidates = [(_leading_number(texts[row * beams + k]), scores[row, k]) for k in range(beams)]
                candidates = [(value, score) for value, score in candidates if value is not None]
                if not candidates:
                    prices.append(0.0)
                    continue
                values, logprobs = np.array(candidates).T
                weights = np.exp(logprobs - logprobs.max())
                prices.append(float((weights * values).sum() / weights.sum()))
        return prices


def _leading_number(text):
    match = re.match(r"\s*(\d+)", text)
    return float(match.group(1)) if match else None

if __name__ == "__main__":
    # Train the day 5 adapter outside the notebook (about half an hour for the 20,000 lite items on an M4 Pro):
    #   uv run python local_pricer.py [number of training items]
    import json
    import sys

    count = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
    train, val, test = load_items()
    pricer = LoraPricer()
    history = pricer.train(train[:count])
    pricer.save()
    (ADAPTER_DIR / "history.json").write_text(json.dumps({"items": count, "loss_every_50_steps": history}))
    predictions = pricer.predict_many(test[:200])
    error = sum(abs(p - t.price) for p, t in zip(predictions, test[:200])) / 200
    print(f"Saved the adapter to {ADAPTER_DIR}; average error on 200 test items: ${error:.2f}")
