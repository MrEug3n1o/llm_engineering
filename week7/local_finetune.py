"""
Local stand-in for the week 7 Colab notebooks: QLoRA fine-tuning on Apple Silicon.

The course fine-tunes Llama 3.2 3B on a CUDA GPU with bitsandbytes (4-bit base model),
PEFT (LoRA adapters) and TRL (SFTTrainer). bitsandbytes needs CUDA, so here the same
recipe runs on MLX instead: the base model is quantized to 4 bits and frozen, and small
LoRA matrices on top of it are trained in 16/32-bit. That combination is exactly QLoRA.

    uv run week7/local_finetune.py train lite          # ~2 hours on an M4 Pro
    uv run week7/local_finetune.py eval lite --size 200

meta-llama/Llama-3.2-3B is gated on Hugging Face and the .env has no HF token, so the
weights come from unsloth/Llama-3.2-3B, an ungated byte-for-byte mirror.
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# The hub's xet transfer stalled for good halfway through a 5 GB file on this Mac; plain HTTP resumes fine.
# Set before huggingface_hub is imported, which reads it once.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn
import mlx.optimizers as optim
from datasets import load_dataset
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten
from mlx_lm import convert, load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.datasets import CacheDataset
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.utils import linear_to_lora_layers, load_adapters

WEEK7 = Path(__file__).parent
BASE_HF = "unsloth/Llama-3.2-3B"
BF16_DIR = WEEK7 / "models" / "llama-3.2-3b-bf16"
MODEL_DIR = WEEK7 / "models" / "llama-3.2-3b-4bit"
ADAPTERS_DIR = WEEK7 / "adapters"
DATA_USER = "ed-donner"

ATTENTION_LAYERS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
MLP_LAYERS = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


@dataclass
class Config:
    """Hyperparameters, named after their counterparts in the course's Colab notebook."""

    name: str
    lite: bool = True
    epochs: float = 1
    batch_size: int = 32  # per optimizer step, as in the course
    micro_batch_size: int = 16  # what fits on the GPU at once; gradients accumulate up to batch_size
    max_seq_length: int = 128
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    target_modules: list[str] = field(default_factory=lambda: list(ATTENTION_LAYERS))
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.01
    weight_decay: float = 0.001
    train_size: int | None = None  # use only the first N training items
    seed: int = 42


CONFIGS = {
    # The course's LITE_MODE settings: r=32 on the attention layers, one epoch of 20k items
    "lite": Config("lite"),
    # Week 7 homework: try to beat the lite score. Same data and learning rate, 5x the adapter capacity:
    # r=64 on the MLP layers too, as in the course's full run. (2e-4 made the loss jump to 8.5 by step 40.)
    "lite-mlp": Config(
        "lite-mlp",
        lora_r=64,
        lora_alpha=128,
        target_modules=ATTENTION_LAYERS + MLP_LAYERS,
        micro_batch_size=8,  # at 16 the peak reached 19 GB and the run died at step 240
    ),
    # A 2-minute smoke test to check memory and speed before committing to a long run
    "smoke": Config("smoke", train_size=640),
}


# ---------- model ----------


def ensure_bf16_model() -> Path:
    """
    Download the original bf16 weights into models/ (6.4 GB).
    """
    index = BF16_DIR / "model.safetensors.index.json"
    shards = set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else {"missing"}
    if not all((BF16_DIR / shard).exists() for shard in shards):
        snapshot_download(BASE_HF, local_dir=BF16_DIR, allow_patterns=["*.json", "*.safetensors"])
    return BF16_DIR


def ensure_base_model() -> Path:
    """Quantize the base model to 4 bits once (1.8 GB)."""
    if not (MODEL_DIR / "config.json").exists():
        convert(str(ensure_bf16_model()), mlx_path=str(MODEL_DIR), quantize=True, q_bits=4, q_group_size=64)
    return MODEL_DIR


def load_base():
    return load(str(ensure_base_model()))


def load_finetuned(name: str):
    model, tokenizer = load_base()
    load_adapters(model, str(ADAPTERS_DIR / name))
    model.eval()
    return model, tokenizer


def parameter_bytes(model: nn.Module) -> int:
    return sum(p.nbytes for _, p in tree_flatten(model.parameters()))


def trainable_parameters(model: nn.Module) -> int:
    return sum(p.size for _, p in tree_flatten(model.trainable_parameters()))


def add_lora(model: nn.Module, config: Config) -> None:
    """Freeze the base model and wrap the target layers with LoRA adapters (PEFT's get_peft_model)."""
    model.freeze()
    linear_to_lora_layers(model, -1, lora_parameters(config))


def lora_parameters(config: Config) -> dict:
    # MLX multiplies the adapter output by `scale` directly; PEFT uses lora_alpha / r
    return {
        "rank": config.lora_r,
        "scale": config.lora_alpha / config.lora_r,
        "dropout": config.lora_dropout,
        "keys": config.target_modules,
    }


# ---------- data ----------


def load_prompts(lite: bool = True) -> dict[str, list[dict]]:
    """The prompt/completion dataset that week 7 day 2 pushes to the hub."""
    name = f"{DATA_USER}/items_prompts_lite" if lite else f"{DATA_USER}/items_prompts_full"
    ds = load_dataset(name)
    return {split: ds[split].to_list() for split in ds}


COMPLETION_WINDOW = 8  # longest completion in tokens, end of text included: "999", ".", "00", <eos>


class PromptCompletionDataset:
    """
    Plain prompt + completion text, no chat template, loss on the completion only.
    This mirrors TRL's SFTTrainer with a prompt/completion dataset on a base model.
    """

    def __init__(self, rows: list[dict], tokenizer):
        self.rows = rows
        self.tokenizer = tokenizer

    def process(self, row: dict) -> tuple[list[int], int]:
        prompt = self.tokenizer.encode(row["prompt"])  # adds <|begin_of_text|>
        completion = self.tokenizer.encode(row["completion"], add_special_tokens=False)
        if len(completion) + 1 > COMPLETION_WINDOW:
            raise ValueError(f"Completion longer than {COMPLETION_WINDOW} tokens: {row['completion']!r}")
        # The offset makes the trainer's loss mask skip the prompt tokens
        return prompt + completion + [self.tokenizer.eos_token_id], len(prompt)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]

    def __len__(self) -> int:
        return len(self.rows)


# ---------- training ----------


class JsonLog(TrainingCallback):
    def __init__(self, path: Path):
        self.path = path
        self.train, self.val = [], []

    def on_train_loss_report(self, info):
        self.train.append(info)
        self.flush()

    def on_val_loss_report(self, info):
        self.val.append(info)
        self.flush()

    def flush(self):
        self.path.write_text(json.dumps({"train": self.train, "val": self.val}, indent=1))


def completion_loss(model, batch: mx.array, lengths: mx.array) -> tuple[mx.array, mx.array]:
    """
    Cross-entropy on the completion tokens only - and the vocabulary projection only for them.

    mlx-lm's default loss projects every position onto the 128k-token vocabulary and masks most of it
    away afterwards: for 32 prompts of ~130 tokens that's 0.5 billion logits per step, which ran the
    GPU out of memory here. The price is 4 tokens, so only those positions go through the output layer.
    """
    offsets, ends = lengths[:, 0:1], lengths[:, 1:2]
    hidden = model.model(batch[:, :-1])
    # hidden[p] predicts token p+1; the completion is tokens offset .. end-1
    positions = offsets - 1 + mx.arange(COMPLETION_WINDOW)[None]
    mask = positions + 1 < ends
    positions = mx.minimum(positions, hidden.shape[1] - 1)
    hidden = mx.take_along_axis(hidden, positions[..., None], axis=1)
    if model.args.tie_word_embeddings:
        logits = model.model.embed_tokens.as_linear(hidden)
    else:
        logits = model.lm_head(hidden)
    targets = mx.take_along_axis(batch, positions + 1, axis=1)
    ce = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mask.sum()
    return ce.astype(mx.float32).sum() / ntoks, ntoks


def schedule(config: Config, iters: int):
    """Linear warmup then cosine decay, as with lr_scheduler_type='cosine' in TRL."""
    warmup = max(1, round(iters * config.warmup_ratio))
    return optim.join_schedules(
        [
            optim.linear_schedule(0.0, config.learning_rate, warmup),
            optim.cosine_decay(config.learning_rate, iters - warmup),
        ],
        [warmup],
    )


def run_training(config: Config, data: dict[str, list[dict]] | None = None) -> Path:
    """Train LoRA adapters into adapters/<name>; `data` defaults to the price prompts."""
    out = ADAPTERS_DIR / config.name
    out.mkdir(parents=True, exist_ok=True)
    mx.random.seed(config.seed)

    model, tokenizer = load_base()
    add_lora(model, config)
    print(f"Trainable parameters: {trainable_parameters(model):,}")

    data = data or load_prompts(config.lite)
    rows = data["train"][: config.train_size] if config.train_size else data["train"]
    train_set = CacheDataset(PromptCompletionDataset(rows, tokenizer))
    val_set = CacheDataset(PromptCompletionDataset(data["val"], tokenizer))
    accumulation = config.batch_size // config.micro_batch_size
    updates = math.ceil(len(rows) * config.epochs / config.batch_size)

    # Everything load_adapters needs to rebuild the same LoRA layers later, plus the run's settings
    adapter_config = {"fine_tune_type": "lora", "num_layers": -1, "lora_parameters": lora_parameters(config)}
    (out / "adapter_config.json").write_text(json.dumps(adapter_config | {"run": asdict(config)}, indent=1))

    # The schedule advances once per optimizer update; mlx-lm's iterations count micro-batches
    optimizer = optim.AdamW(learning_rate=schedule(config, updates), weight_decay=config.weight_decay)
    args = TrainingArgs(
        batch_size=config.micro_batch_size,
        grad_accumulation_steps=accumulation,
        iters=updates * accumulation,
        val_batches=16,
        steps_per_report=20,
        steps_per_eval=200,
        steps_per_save=200,
        max_seq_length=config.max_seq_length,
        adapter_file=str(out / "adapters.safetensors"),
    )
    start = time.time()
    log = JsonLog(out / "log.json")
    train(model, optimizer, train_set, val_set, args=args, loss=completion_loss, training_callback=log)
    summary = {"steps": updates, "minutes": (time.time() - start) / 60, "peak_memory_gb": mx.get_peak_memory() / 1e9}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"Trained {updates} steps in {summary['minutes']:.1f} min, peak memory {summary['peak_memory_gb']:.1f} GB")
    return out


def is_trained(name: str) -> bool:
    """Adapters are checkpointed during training, so their file alone doesn't mean the run finished."""
    return (ADAPTERS_DIR / name / "summary.json").exists()


# ---------- inference ----------


def price_token_values(tokenizer) -> tuple[mx.array, mx.array]:
    """Llama 3's tokenizer has one token for every number 0-999: the whole dollar part of most prices."""
    ids, values = [], []
    for n in range(1000):
        token_ids = tokenizer.encode(str(n), add_special_tokens=False)
        if len(token_ids) == 1:
            ids.append(token_ids[0])
            values.append(n)
    return mx.array(ids), mx.array(values, dtype=mx.float32)


def make_predictor(model, tokenizer, max_tokens: int = 6):
    """Greedy decoding of the completion, as in the day 5 notebook: returns e.g. '123.00'."""
    eos = tokenizer.eos_token_id

    def predict(datapoint: dict) -> str:
        tokens = mx.array(tokenizer.encode(datapoint["prompt"]))[None]
        cache = make_prompt_cache(model)
        out = []
        for _ in range(max_tokens):
            logits = model(tokens, cache=cache)[:, -1, :]
            token = mx.argmax(logits, axis=-1)
            next_id = token.item()
            if next_id == eos:
                break
            out.append(next_id)
            tokens = token[None]
        return tokenizer.decode(out)

    return predict


def make_weighted_predictor(model, tokenizer, top_k: int = 5):
    """
    Look at the probability of every price token after 'Price is $' and blend the top few.
    One forward pass, no decoding, and a hedge when the model is torn between two prices.
    """
    ids, values = price_token_values(tokenizer)

    def predict(datapoint: dict) -> float:
        tokens = mx.array(tokenizer.encode(datapoint["prompt"]))[None]
        logits = model(tokens)[0, -1, ids]
        top = mx.argsort(-logits)[:top_k]
        weights = mx.softmax(logits[top])
        return (weights * values[top]).sum().item()

    return predict


def average_error(predict, data: list[dict], size: int = 200) -> float:
    """The headline number from util.Tester, without the charts."""
    from util import Tester

    errors = [abs(Tester.post_process(predict(d)) - float(d["completion"])) for d in data[:size]]
    return sum(errors) / len(errors)


# ---------- CLI ----------


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["train", "eval", "prepare"])
    parser.add_argument("config", nargs="?", default="lite", choices=list(CONFIGS) + ["base"])
    parser.add_argument("--size", type=int, default=200)
    args = parser.parse_args()

    if args.action == "prepare":
        print(ensure_base_model())
    elif args.action == "train":
        run_training(CONFIGS[args.config])
    else:
        model, tokenizer = load_base() if args.config == "base" else load_finetuned(args.config)
        error = average_error(make_predictor(model, tokenizer), load_prompts(True)["test"], args.size)
        print(f"{args.config}: average error ${error:,.2f} over {args.size} test items")


if __name__ == "__main__":
    main()
