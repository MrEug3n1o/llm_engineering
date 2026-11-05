"""Model plumbing shared by the week 4 tools: which models are available, and one way to call them.

Local models run on Ollama. Its OpenAI-compatible endpoint always uses the server's default context
window (4096 tokens on this Mac) and quietly shifts out older tokens once a conversation outgrows it -
which a fix-the-compiler-error loop, or a prompt containing a whole module, soon does. Ollama's native
/api/chat accepts a bigger `num_ctx`, so local models go through that instead. API models (only offered
when their key is in .env) use the OpenAI client as in the course.
"""

import os

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

OLLAMA_API = "http://localhost:11434/api"
NUM_CTX = 16384
REQUEST_TIMEOUT = 900

# The frontier models from the course: label -> (key in .env, base URL, model)
API_MODELS = {
    "GPT-5": ("OPENAI_API_KEY", None, "gpt-5"),
    "Claude Sonnet 4.5": ("ANTHROPIC_API_KEY", "https://api.anthropic.com/v1/", "claude-sonnet-4-5-20250929"),
    "Gemini 2.5 Pro": ("GOOGLE_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-2.5-pro"),
    "Grok 4": ("GROK_API_KEY", "https://api.x.ai/v1", "grok-4"),
    "GPT-OSS 120B (Groq)": ("GROQ_API_KEY", "https://api.groq.com/openai/v1", "openai/gpt-oss-120b"),
    "Qwen3 Coder 30B (OpenRouter)": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1", "qwen/qwen3-coder-30b-a3b-instruct"),
}

OLLAMA_PREFIX = "Ollama: "


def ollama_models():
    """Models on a local Ollama server; an empty list if it is not running."""
    try:
        response = requests.get(f"{OLLAMA_API}/tags", timeout=2)
        response.raise_for_status()
        return [model["name"] for model in response.json().get("models", [])]
    except Exception:
        return []


def available_models():
    """Labels of every usable model: local ones first (coding models at the top), then APIs with a key."""
    local = sorted((n for n in ollama_models() if "embed" not in n), key=lambda n: ("coder" not in n, n))
    remote = [label for label, (key_name, _, _) in API_MODELS.items() if os.getenv(key_name)]
    return [OLLAMA_PREFIX + name for name in local] + remote


def preferred_model(labels):
    """The strongest local coder that is available, else whatever comes first."""
    for wanted in ("qwen2.5-coder", "qwen3-coder", "gpt-oss", "deepseek-coder", "qwen2.5:14b"):
        for label in labels:
            if wanted in label:
                return label
    return labels[0] if labels else None


def chat(label, messages, temperature=None):
    """Send the messages to the model with this label and return the reply text."""
    if label.startswith(OLLAMA_PREFIX):
        options = {"num_ctx": NUM_CTX}
        if temperature is not None:
            options["temperature"] = temperature
        response = requests.post(
            f"{OLLAMA_API}/chat",
            json={"model": label.removeprefix(OLLAMA_PREFIX), "messages": messages, "stream": False, "options": options},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    if label not in API_MODELS:
        raise ValueError(f"Unknown model {label!r}; choose one of {available_models()}")
    key_name, base_url, model = API_MODELS[label]
    client = OpenAI(api_key=os.getenv(key_name), base_url=base_url)
    extra = {"reasoning_effort": "high"} if model == "gpt-5" else {}
    if temperature is not None:
        extra["temperature"] = temperature
    response = client.chat.completions.create(model=model, messages=messages, **extra)
    return response.choices[0].message.content
