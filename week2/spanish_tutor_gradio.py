"""Week 2 exercise: the week 1 Spanish tutor as a full Gradio prototype.

Streaming replies, a model switcher across every provider you have configured,
two real tools (verb conjugation and flashcard decks), and optional voice in/out.

Run it with:  python week2/spanish_tutor_gradio.py
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import requests
from dotenv import load_dotenv
from openai import OpenAI

from tutor_tools import TOOL_FUNCTIONS, TOOLS, latest_deck

load_dotenv(override=True)

OLLAMA_URL = "http://localhost:11434/v1"
WHISPER_MODEL = "openai/whisper-base"  # swap for whisper-small if you want more accuracy
ANTHROPIC_URL = "https://api.anthropic.com/v1/"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

SYSTEM_PROMPT = """You are an expert Spanish tutor for an English-speaking learner.
Explain mistakes clearly and briefly, and keep every answer practical: a learner should be
able to use what you say in a conversation the same day.

Rules you always follow:
- Whenever you write a Spanish noun, include its definite article: el, la, los or las.
  Write "la manzana" and "el plátano", never "manzana" on its own.
- Mark the stressed syllable of a new word when the pronunciation is not obvious.
- When you are asked for a conjugation, call the conjugate_verb tool and report what it
  returns. Never write a conjugation table from memory.
- When the learner asks for flashcards or a deck, call the save_flashcards tool so they get
  a CSV they can import. Write the cards yourself, then pass them to the tool.
- Answer in English unless the learner writes in Spanish or asks for Spanish.
Respond in markdown, without code blocks."""

WELCOME = (
    "¡Hola! I'm your Spanish tutor. Ask me anything about Spanish - or try "
    "*\"conjugate tener in the future\"* or *\"make me a deck of 10 kitchen words\"*."
)


def ollama_models():
    """Ask a local Ollama server which models it has; an empty list if it is not running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        response.raise_for_status()
        return [model["name"] for model in response.json().get("models", [])]
    except Exception:
        return []


def available_models():
    """Map a friendly label to (client, model) for every provider that is configured."""
    models = {}

    if os.getenv("OPENAI_API_KEY"):
        client = OpenAI()
        models["GPT-4.1-mini"] = (client, "gpt-4.1-mini")
    if os.getenv("ANTHROPIC_API_KEY"):
        client = OpenAI(api_key=os.getenv("ANTHROPIC_API_KEY"), base_url=ANTHROPIC_URL)
        models["Claude Haiku 4.5"] = (client, "claude-haiku-4-5")
    if os.getenv("GOOGLE_API_KEY"):
        client = OpenAI(api_key=os.getenv("GOOGLE_API_KEY"), base_url=GEMINI_URL)
        models["Gemini 2.5 Flash"] = (client, "gemini-2.5-flash")

    ollama = OpenAI(base_url=OLLAMA_URL, api_key="ollama")
    for name in ollama_models():
        models[f"Ollama: {name}"] = (ollama, name)

    return models


MODELS = available_models()


def collect_tool_calls(chunks_of_tool_calls, accumulated):
    """Rebuild whole tool calls from the fragments that arrive in a stream."""
    for delta in chunks_of_tool_calls:
        call = accumulated.setdefault(
            delta.index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
        )
        if delta.id:
            call["id"] = delta.id
        if delta.function and delta.function.name:
            call["function"]["name"] = delta.function.name
        if delta.function and delta.function.arguments:
            call["function"]["arguments"] += delta.function.arguments
    return accumulated


def run_tool_calls(tool_calls, decks):
    """Run each tool the model asked for and build the tool messages that answer it."""
    results = []
    for call in tool_calls:
        name = call["function"]["name"]
        function = TOOL_FUNCTIONS.get(name)
        if function is None:
            content = f"There is no tool called {name}."
        else:
            try:
                arguments = json.loads(call["function"]["arguments"] or "{}")
                content = function(**arguments)
                if name == "save_flashcards" and latest_deck():
                    decks.append(latest_deck())
            except Exception as error:
                content = f"The {name} tool failed: {type(error).__name__}: {error}"
        results.append({"role": "tool", "content": content, "tool_call_id": call["id"]})
    return results


def stream_reply(messages, client, model, decks):
    """Stream the assistant's reply, pausing to run any tools it asks for along the way."""
    while True:
        stream = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS, stream=True
        )
        reply = ""
        tool_calls = {}
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                reply += delta.content
                yield reply
            if delta.tool_calls:
                collect_tool_calls(delta.tool_calls, tool_calls)

        if not tool_calls:
            return

        calls = [tool_calls[index] for index in sorted(tool_calls)]
        # Some providers reject a null content, so only include it when the model said something
        asking_for_tools = {"role": "assistant", "tool_calls": calls}
        if reply:
            asking_for_tools["content"] = reply
        messages.append(asking_for_tools)
        messages.extend(run_tool_calls(calls, decks))


_local_ears = None


def local_ears():
    """Load Whisper locally, once. The first call downloads the model, so it takes a while."""
    global _local_ears
    if _local_ears is None:
        gr.Info(f"Loading {WHISPER_MODEL} locally - the first time also downloads it.")
        from transformers import pipeline  # imported here so the app starts fast without it

        _local_ears = pipeline("automatic-speech-recognition", model=WHISPER_MODEL)
    return _local_ears


def ffmpeg_exe():
    """ffmpeg: this machine's own if it has one, otherwise the copy in the imageio-ffmpeg wheel."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg  # a bundled static ffmpeg, so nothing has to be installed system-wide

    return imageio_ffmpeg.get_ffmpeg_exe()


def read_recording(audio_path):
    """Decode a recording of any format into the mono 16kHz float array Whisper expects.

    A browser records webm/opus rather than wav, so this cannot be a simple wav read - ffmpeg
    decodes straight to raw 16-bit PCM on its stdout, which numpy reads without a temp file.
    """
    decode = [
        ffmpeg_exe(), "-nostdin", "-loglevel", "error",
        "-i", str(audio_path),
        "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", "-",
    ]
    finished = subprocess.run(decode, capture_output=True, timeout=120)
    if finished.returncode != 0:
        raise RuntimeError(finished.stderr.decode(errors="replace").strip() or "ffmpeg failed")
    return np.frombuffer(finished.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe(audio_path):
    """Turn a recording into text: OpenAI when there is a key, otherwise Whisper on this machine."""
    if audio_path is None:
        return ""

    if os.getenv("OPENAI_API_KEY"):
        with open(audio_path, "rb") as audio_file:
            transcription = OpenAI().audio.transcriptions.create(
                model="gpt-4o-mini-transcribe", file=audio_file
            )
        return transcription.text

    try:
        audio = read_recording(audio_path)
        return local_ears()({"raw": audio, "sampling_rate": 16_000})["text"].strip()
    except Exception as error:
        raise gr.Error(f"Could not transcribe that recording: {type(error).__name__}: {error}")


def spoken_text(text, limit=700):
    """Markdown reads badly out loud, so strip it back to the words."""
    plain = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    plain = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", plain)
    plain = re.sub(r"[*_`#>|-]+", " ", plain)
    plain = re.sub(r"\s+", " ", plain).strip()
    return plain[:limit]


_spanish_voice = None


def spanish_voice():
    """The name of an installed Spanish voice for macOS 'say', or None."""
    global _spanish_voice
    if _spanish_voice is None:
        try:
            voices = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=10).stdout
            match = re.search(r"^(\S+(?: \(Spanish[^)]*\))?)\s+es_", voices, re.MULTILINE)
            _spanish_voice = match.group(1).strip() if match else ""
        except Exception:
            _spanish_voice = ""
    return _spanish_voice or None


def speak_locally(text):
    """Fall back to the voice built into macOS, in Spanish if a Spanish voice is installed."""
    if not shutil.which("say"):
        gr.Warning("Speaking replies needs an OPENAI_API_KEY, or macOS with the 'say' command.")
        return None
    path = Path(tempfile.gettempdir()) / "spanish_tutor_reply.wav"
    command = ["say", "--data-format=LEI16@22050", "-o", str(path)]
    voice = spanish_voice()
    if voice:
        command += ["-v", voice]
    subprocess.run(command + [text], check=True, timeout=120)
    return str(path)


def speak(text):
    """Read the reply out loud, with OpenAI if there is a key and with this machine's voice if not."""
    words = spoken_text(text)
    if not words:
        return None
    try:
        if not os.getenv("OPENAI_API_KEY"):
            return speak_locally(words)
        response = OpenAI().audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="coral",
            input=words,
            instructions="Speak warmly, like a patient language teacher. Pronounce Spanish words as a native speaker would.",
        )
        return response.content
    except Exception as error:
        gr.Warning(f"Could not generate audio: {type(error).__name__}")
        return None


def chat(message, history, model_label, voice_on):
    """The Gradio callback: stream a reply, then optionally speak it and offer any new deck."""
    if not MODELS:
        raise gr.Error(
            "No models available. Add an API key to .env, or start a local model with `ollama serve`."
        )
    if not message.strip():
        return

    client, model = MODELS[model_label]
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [
        {"role": "user", "content": message}
    ]
    chat_history = history + [{"role": "user", "content": message}]
    decks = []

    yield "", chat_history + [{"role": "assistant", "content": ""}], None, gr.update()

    reply = ""
    for reply in stream_reply(messages, client, model, decks):
        yield "", chat_history + [{"role": "assistant", "content": reply}], None, gr.update()

    chat_history += [{"role": "assistant", "content": reply}]
    deck_update = gr.update(value=str(decks[-1]), visible=True) if decks else gr.update()
    yield "", chat_history, speak(reply) if voice_on else None, deck_update


def reset():
    """Back to a fresh conversation: the welcome message, no audio, no deck."""
    return [{"role": "assistant", "content": WELCOME}], None, gr.update(visible=False)


def build_ui():
    labels = list(MODELS)
    with gr.Blocks(title="Spanish Tutor", theme=gr.themes.Soft()) as ui:
        gr.Markdown("# 🇪🇸 Spanish Tutor\nStreaming, switchable models, real tools, and a voice.")
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=460,
                    type="messages",
                    value=[{"role": "assistant", "content": WELCOME}],
                    label="Tutor",
                )
            with gr.Column(scale=1):
                model_selector = gr.Dropdown(
                    labels, label="Model", value=labels[0] if labels else None
                )
                voice_on = gr.Checkbox(label="Speak the replies", value=False)
                audio_output = gr.Audio(label="Tutor's voice", autoplay=True)
                deck_file = gr.File(label="Your flashcard deck", visible=False)
                clear = gr.Button("Clear conversation")
        with gr.Row():
            message = gr.Textbox(
                label="Your message", placeholder="¿Cómo se dice...?", submit_btn=True, scale=4
            )
            microphone = gr.Audio(
                sources=["microphone", "upload"], type="filepath", label="Or speak", scale=1
            )

        inputs = [message, chatbot, model_selector, voice_on]
        outputs = [message, chatbot, audio_output, deck_file]
        message.submit(chat, inputs, outputs)
        # Recording and uploading a clip both transcribe into the box and then send it
        for speaking in (microphone.stop_recording, microphone.upload):
            speaking(transcribe, microphone, message).then(lambda: None, None, microphone).then(
                chat, inputs, outputs
            )
        fresh = [chatbot, audio_output, deck_file]
        clear.click(reset, None, fresh)
        chatbot.clear(reset, None, fresh)  # the chatbot's own bin icon, so both routes agree
    return ui


if __name__ == "__main__":
    if not MODELS:
        print("No models available: add an API key to .env, or run `ollama serve` for a local model.")
    else:
        print(f"Models available: {', '.join(MODELS)}")
        build_ui().launch(inbrowser=True)
