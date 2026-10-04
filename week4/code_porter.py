"""Week 4: port Python to high-performance C++ or Rust with open-source models, then prove it works.

The course labs (days 3-5) send the Python to a model once, write whatever comes back to main.cpp
and hope it compiles. Frontier models usually manage that; small local models usually don't - on
this Mac both qwen2.5:14b and llama3.2 failed to compile the pi example on their first try
(a missing #include <iomanip>, a Python-style 200_000_000 literal). So this module adds the step a
human would take next: hand the compiler error, or the wrong output, back to the model and ask for
a fix. A port only counts once it compiles and prints the same result as the Python.

Used by week4/local_labs.ipynb (days 3-5 on local models) and runnable as a Gradio app:

    python week4/code_porter.py      (needs `ollama serve`, or API keys in .env)
"""

import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from llm import available_models, chat, preferred_model

BUILD_DIR = Path(__file__).parent / "build"
RUN_TIMEOUT = 300  # seconds; generated code with an accidental infinite loop shouldn't hang the app
MAX_ATTEMPTS = 4
TEMPERATURE = 0.2  # low, so a leaderboard run is close to repeatable


# ------------------------------------------------------------------ toolchains


def rustc_path():
    """rustc on the PATH, or in ~/.cargo/bin where rustup installs it (often missing from the PATH)."""
    found = shutil.which("rustc")
    cargo_rustc = Path.home() / ".cargo" / "bin" / "rustc"
    return found or (str(cargo_rustc) if cargo_rustc.exists() else None)


def cpp_compiler():
    return shutil.which("clang++") or shutil.which("g++")


def available_languages():
    languages = []
    if cpp_compiler():
        languages.append("C++")
    if rustc_path():
        languages.append("Rust")
    return languages


def source_file(language):
    return BUILD_DIR / ("main.rs" if language == "Rust" else "main.cpp")


def executable():
    return BUILD_DIR / ("main.exe" if platform.system() == "Windows" else "main")


def compile_command(language):
    """Flags for the fastest runtime: the commands the course got from GPT-5, adapted per platform."""
    exe, src = str(executable()), str(source_file(language))
    if language == "Rust":
        rustc = rustc_path()
        if not rustc:
            raise RuntimeError("No Rust toolchain: install it from https://rustup.rs to port to Rust.")
        return [rustc, src, "-C", "opt-level=3", "-C", "target-cpu=native", "-C", "codegen-units=1",
                "-C", "lto=fat", "-C", "panic=abort", "-o", exe]
    compiler = cpp_compiler()
    if not compiler:
        raise RuntimeError("No C++ compiler: on a Mac run `xcode-select --install`.")
    # No -Ofast / -ffast-math, unlike the course: they let the compiler reorder floating-point sums,
    # which changed the 12th decimal of the pi result, and the port has to match the Python exactly.
    # Apple Silicon wants -mcpu=native, x86 wants -march=native.
    arch = "-mcpu=native" if platform.machine().lower() in ("arm64", "aarch64") else "-march=native"
    return [compiler, "-std=c++17", "-O3", arch, "-flto=thin", "-DNDEBUG", src, "-o", exe]


# ------------------------------------------------------------------ running code


def run_python(code):
    """Run the Python in a fresh interpreter, so a crash or a stray global can't touch the app."""
    try:
        result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, timeout=RUN_TIMEOUT)
    except subprocess.TimeoutExpired:
        return f"Error: Python took longer than {RUN_TIMEOUT} seconds"
    return result.stdout if result.returncode == 0 else f"Error:\n{result.stderr}"


def compile_code(code, language):
    """Write and compile the code. Returns the compiler's error text, or None if it compiled."""
    BUILD_DIR.mkdir(exist_ok=True)
    source_file(language).write_text(code, encoding="utf-8")
    executable().unlink(missing_ok=True)
    result = subprocess.run(compile_command(language), text=True, capture_output=True)
    if result.returncode == 0:
        return None
    # Drop the absolute build path so error lines read "main.cpp:23:51: error: ..."
    return (result.stderr or result.stdout).replace(f"{BUILD_DIR}{os.sep}", "")


def run_executable():
    """Run the compiled program. Returns (ok, output)."""
    try:
        result = subprocess.run([str(executable())], text=True, capture_output=True, timeout=RUN_TIMEOUT)
    except subprocess.TimeoutExpired:
        return False, f"The program was still running after {RUN_TIMEOUT} seconds."
    if result.returncode != 0:
        return False, f"The program exited with code {result.returncode}.\n{result.stderr}"
    return True, result.stdout


def compile_and_run(code, language="C++"):
    """What the course's 'Run C++' button does: compile, run, return the output or the error."""
    error = compile_code(code, language)
    if error:
        return f"An error occurred while compiling:\n{error}"
    return run_executable()[1]


def comparable(output):
    """The output minus timing lines, which are supposed to differ between Python and the port."""
    return [line.strip() for line in output.strip().splitlines() if "time" not in line.lower() and line.strip()]


def mismatch_hints(expected, actual):
    """Spell out what differs, so a small model knows what to fix instead of guessing.

    Two failures came up again and again with qwen2.5:14b: printing 3.14159 instead of 12 decimals,
    and summing into an unsigned type, which turns -20 into 18446744073709551596.
    """
    wanted, got = comparable(expected), comparable(actual)
    hints = [f"Expected line: {w}\nYour line:     {g}" for w, g in zip(wanted, got) if w != g]
    if len(wanted) != len(got):
        hints.append(f"Expected {len(wanted)} lines (ignoring timings), got {len(got)}.")
    for w, g in zip(wanted, got):
        w_decimals = [len(d) for d in re.findall(r"\d\.(\d+)", w)]
        g_decimals = [len(d) for d in re.findall(r"\d\.(\d+)", g)]
        if w_decimals and w_decimals != g_decimals:
            hints.append(f"Print numbers with exactly {w_decimals[0]} digits after the decimal point, as the Python does.")
    for number in {int(n) for n in re.findall(r"\d{9,}", actual)}:
        for bits in (32, 64):
            if 0 < 2**bits - number < 10**7:
                hints.append(f"{number} is 2^{bits} - {2**bits - number}: the negative value -{2**bits - number} "
                             f"ended up in an unsigned {bits}-bit type. Use signed types for anything that can go negative.")
    return "\n\n".join(dict.fromkeys(hints))


def timing(output):
    """The first number on a line mentioning time, e.g. 'Execution Time: 0.081 seconds' -> 0.081."""
    for line in output.splitlines():
        if "time" in line.lower() and (match := re.search(r"\d+(?:\.\d+)?(?:e-?\d+)?", line)):
            return float(match.group())
    return None


# ------------------------------------------------------------------ porting


def system_prompt(language):
    return f"""Your task is to convert Python code into high performance {language} code.
Respond only with {language} code. Do not provide any explanation other than occasional comments.
The {language} response needs to produce an identical output in the fastest possible time."""


def user_prompt(python, language, system_info):
    return f"""Port this Python code to {language} with the fastest possible implementation that produces identical output in the least time.
The system information is:
{system_info}
Your response will be written to a file called {source_file(language).name} and then compiled and executed; the compilation command is:
{compile_command(language)}
Respond only with {language} code. It must be one complete file that compiles on its own, with every #include or use it needs.
Watch out for integer overflow: Python integers never overflow, so pick wide enough types.
Python code to port:

```python
{python}
```"""


def extract_code(reply):
    """The code from a reply, whether or not the model wrapped it in a fence (or several)."""
    blocks = re.findall(r"```[\w+#-]*\s*\n(.*?)```", reply, flags=re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    return reply.replace("```", "").strip() + "\n"


def port(model_label, python, language="C++", system_info=""):
    """A single port, exactly like the course: no compiling, no checking. Returns the code."""
    messages = [
        {"role": "system", "content": system_prompt(language)},
        {"role": "user", "content": user_prompt(python, language, system_info)},
    ]
    return extract_code(chat(model_label, messages, temperature=TEMPERATURE))


def port_and_verify(model_label, python, language="C++", system_info="", expected=None, attempts=MAX_ATTEMPTS):
    """Port, compile, run and compare with the Python; on failure send the problem back to the model.

    `expected` is the Python's output, if you already have it (running the Python can be slow).
    A generator: yields (status_log, code, output) after every step, so a UI can show progress.
    The last thing yielded is the final result; check `log[-1]` for whether it succeeded.
    """
    if expected is None:
        expected = run_python(python)
    messages = [
        {"role": "system", "content": system_prompt(language)},
        {"role": "user", "content": user_prompt(python, language, system_info)},
    ]
    log, code, output = [], "", ""

    for attempt in range(1, attempts + 1):
        log.append(f"Attempt {attempt}: asking {model_label} for {language}...")
        yield log, code, output
        start = time.time()
        reply = chat(model_label, messages, temperature=TEMPERATURE)
        code = extract_code(reply)
        messages.append({"role": "assistant", "content": reply})
        log[-1] += f" got {len(code.splitlines())} lines in {time.time() - start:.0f}s"
        yield log, code, output

        error = compile_code(code, language)
        if error:
            first_errors = "\n".join(line for line in error.splitlines() if "error" in line.lower())[:600]
            log.append(f"  ✗ does not compile:\n{first_errors or error[:600]}")
            problem = f"That code does not compile. The compiler said:\n\n{error[:4000]}"
        else:
            ok, output = run_executable()
            if not ok:
                log.append(f"  ✗ compiles but crashes: {output[:300]}")
                problem = f"That code compiles but fails when run:\n\n{output[:4000]}"
            elif comparable(output) != comparable(expected):
                log.append(f"  ✗ runs but prints something different:\n{output.strip()[:300]}")
                problem = (f"That code runs but its output is wrong.\n\n{mismatch_hints(expected, output)}\n\n"
                           "Keep everything that was already right and change only what is needed to fix this.")
            else:
                log.append("  ✓ compiles, runs and matches the Python output")
                yield log, code, output
                return
        yield log, code, output
        messages.append({"role": "user", "content": problem + f"\n\nReply with the complete corrected {language} file only."})

    log.append(f"Gave up after {attempts} attempts.")
    yield log, code, output


def port_until_verified(model_label, python, language="C++", system_info="", expected=None, attempts=MAX_ATTEMPTS, verbose=True):
    """port_and_verify for notebooks: prints progress and returns (succeeded, code, output, attempts_used)."""
    printed = 0
    for log, code, output in port_and_verify(model_label, python, language, system_info, expected, attempts):
        for line in log[printed:]:
            if verbose:
                print(line, flush=True)
        printed = len(log)
    attempts_used = sum(line.startswith("Attempt") for line in log)
    return log[-1].lstrip().startswith("✓"), code, output, attempts_used


# ------------------------------------------------------------------ the app (course day 5, plus verification)


PI = """import time

def calculate(iterations, param1, param2):
    result = 1.0
    for i in range(1, iterations+1):
        j = i * param1 - param2
        result -= (1/j)
        j = i * param1 + param2
        result += (1/j)
    return result

start_time = time.time()
result = calculate(200_000_000, 4, 1) * 4
end_time = time.time()

print(f"Result: {result:.12f}")
print(f"Execution Time: {(end_time - start_time):.6f} seconds")
"""

PYTHON_HARD = """# Be careful to support large numbers

def lcg(seed, a=1664525, c=1013904223, m=2**32):
    value = seed
    while True:
        value = (a * value + c) % m
        yield value

def max_subarray_sum(n, seed, min_val, max_val):
    lcg_gen = lcg(seed)
    random_numbers = [next(lcg_gen) % (max_val - min_val + 1) + min_val for _ in range(n)]
    max_sum = float('-inf')
    for i in range(n):
        current_sum = 0
        for j in range(i, n):
            current_sum += random_numbers[j]
            if current_sum > max_sum:
                max_sum = current_sum
    return max_sum

def total_max_subarray_sum(n, initial_seed, min_val, max_val):
    total_sum = 0
    lcg_gen = lcg(initial_seed)
    for _ in range(20):
        seed = next(lcg_gen)
        total_sum += max_subarray_sum(n, seed, min_val, max_val)
    return total_sum

# Parameters
n = 10000         # Number of random numbers
initial_seed = 42 # Initial seed for the LCG
min_val = -10     # Minimum value of random numbers
max_val = 10      # Maximum value of random numbers

# Timing the function
import time
start_time = time.time()
result = total_max_subarray_sum(n, initial_seed, min_val, max_val)
end_time = time.time()

print("Total Maximum Subarray Sum (20 runs):", result)
print("Execution Time: {:.6f} seconds".format(end_time - start_time))
"""

EXAMPLES = {"Pi series (day 3)": PI, "Max subarray sum (day 5)": PYTHON_HARD}


def speedup_note(python_output, ported_output):
    python_time, ported_time = timing(python_output or ""), timing(ported_output or "")
    if python_time and ported_time:
        return f"\n→ {python_time / max(ported_time, 1e-9):,.0f}x faster than the Python"
    return ""


def build_ui():
    import gradio as gr
    from styles import CSS
    from system_info import retrieve_system_info

    system_info = retrieve_system_info()
    models = available_models()
    languages = available_languages()

    def convert(model_label, python, language, verify, python_output):
        if model_label not in available_models():
            raise gr.Error("No model available - start Ollama (`ollama serve`) or add an API key to .env.")
        if not verify:
            yield port(model_label, python, language, system_info), "Ported once, not compiled or checked."
            return
        expected = python_output if python_output and not python_output.startswith("Error") else None
        if expected is None:
            yield "", "Running the Python first, to know what output to expect..."
            expected = run_python(python)
        for log, code, _ in port_and_verify(model_label, python, language, system_info, expected):
            yield code, "\n".join(log)

    def run_ported(code, language, python_output):
        output = compile_and_run(code, language)
        return output + speedup_note(python_output, output)

    with gr.Blocks(css=CSS, theme=gr.themes.Monochrome(), title="Port Python to C++ / Rust") as ui:
        gr.Markdown("## Port Python to C++ or Rust with open-source models\n"
                    "With **Verify** on, the port is compiled, run and compared with the Python, and any "
                    "compiler error or wrong output goes back to the model to fix.")
        with gr.Row():
            example = gr.Dropdown(list(EXAMPLES), value=list(EXAMPLES)[0], label="Example")
            language = gr.Radio(languages, value=languages[0] if languages else None, label="Target language")
        with gr.Row(equal_height=True):
            python = gr.Code(label="Python (original)", value=PI, language="python", lines=24)
            ported = gr.Code(label="Ported (generated)", value="", language="cpp", lines=24)
        with gr.Row(elem_classes=["controls"]):
            python_run = gr.Button("Run Python", elem_classes=["run-btn", "py"])
            model = gr.Dropdown(models, value=preferred_model(models), show_label=False)
            verify = gr.Checkbox(value=True, label="Verify & fix")
            convert_btn = gr.Button("Port", elem_classes=["convert-btn"])
            ported_run = gr.Button("Run port", elem_classes=["run-btn", "cpp"])
        with gr.Row(equal_height=True):
            python_out = gr.TextArea(label="Python result", lines=8, elem_classes=["py-out"])
            ported_out = gr.TextArea(label="Ported result / porting log", lines=8, elem_classes=["cpp-out"])

        example.change(lambda name: EXAMPLES[name], example, python)
        # gr.Code has no Rust highlighting, so Rust is shown as plain text
        language.change(lambda lang: gr.update(language=None if lang == "Rust" else "cpp"), language, ported)
        python_run.click(run_python, python, python_out)
        convert_btn.click(convert, [model, python, language, verify, python_out], [ported, ported_out])
        ported_run.click(run_ported, [ported, language, python_out], ported_out)
    return ui


if __name__ == "__main__":
    build_ui().launch(inbrowser=True)
