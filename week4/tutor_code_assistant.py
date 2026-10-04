"""Week 4 exercise: docstrings and unit tests for the Spanish tutor's own code, by an open-source model.

The course's end-of-week challenge is a tool that adds docstrings/comments to code, or writes unit tests
for it. This one works on the Spanish tutor project built in weeks 1-3, and - like week4/code_porter.py -
it doesn't trust the model's output until it has been checked:

- Docstrings. The model never rewrites the file. It is shown one function at a time and returns only a
  docstring, which is inserted programmatically. Afterwards the module's syntax tree, with docstrings
  removed, must be identical to the original's: proof that not a single line of code changed.
- Unit tests. The model writes a pytest file, which runs against a copy of the module in a temporary
  folder (so tests that write decks can't touch the real ones). Failures go back to the model, which
  either fixes a wrong expectation or, if the code really is wrong, marks the test xfail as a likely bug.

Results are saved to week4/generated/. Run it with:

    python week4/tutor_code_assistant.py      (needs `ollama serve`, or API keys in .env)
"""

import ast
import difflib
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import gradio as gr

from llm import available_models, chat, preferred_model

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = Path(__file__).parent / "generated"
TEST_TIMEOUT = 180
PASTED = "Paste your own code"

# The tutor project, week by week
SOURCES = {
    "week1/spanish_tutor.py - CLI tutor and CSV cleanup": ROOT / "week1" / "spanish_tutor.py",
    "week2/tutor_tools.py - conjugation and flashcard tools": ROOT / "week2" / "tutor_tools.py",
    "week3/spanish_data_generator.py - synthetic data generator": ROOT / "week3" / "spanish_data_generator.py",
}


# ------------------------------------------------------------------ docstrings


DOCSTRING_SYSTEM = """You write docstrings for Python code in a Spanish-tutor project.
Write in the style of the existing code: one plain sentence saying what the function does or returns,
then - only if it helps - a short line about arguments, edge cases or side effects (such as files written).
No "Args:"/"Returns:" sections, no restating the function name, no type information already in the signature.
Reply with the docstring text only: no quotes, no code, no code fences."""


def needs_docstring(node):
    return (
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and ast.get_docstring(node) is None
        and node.body[0].lineno > node.lineno  # skip one-liners like `def f(): return 1`
    )


def clean_docstring(reply):
    """The docstring text from a reply, whatever wrapping the model added around it."""
    fenced = re.search(r"```\w*\s*\n(.*?)```", reply, flags=re.DOTALL)
    text = (fenced.group(1) if fenced else reply).strip()
    try:  # a bare """string""", or the whole function echoed back with the docstring inside
        tree = ast.parse(text)
        node = next((n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))), tree)
        text = ast.get_docstring(node) or text
    except SyntaxError:
        pass
    text = text.strip().strip('"').strip("'").strip()
    # Nothing in the text may end the docstring early or turn into an escape sequence
    return text.replace("\\", "\\\\").replace('"""', "'''")


def docstring_lines(text, indent):
    lines = textwrap.dedent(text).strip().splitlines()
    if len(lines) == 1:
        return [f'{indent}"""{lines[0]}"""']
    return [f'{indent}"""{lines[0]}'] + [f"{indent}{line}".rstrip() for line in lines[1:]] + [f'{indent}"""']


def without_docstrings(source):
    """The syntax tree as text, with every docstring removed - equal before and after means no code changed."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree)


def add_docstrings(source, model_label):
    """Generator yielding (log, documented_source) as each missing docstring is written."""
    tree = ast.parse(source)
    lines = source.splitlines()
    targets = [node for node in ast.walk(tree) if needs_docstring(node)]
    module_doc = ast.get_docstring(tree) or "(no module docstring)"
    log = [f"{len(targets)} functions/classes have no docstring."]
    yield log, source

    insertions = []  # (line index, lines to insert)
    for node in targets:
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        segment = ast.get_source_segment(source, node)
        prompt = (f"The module's docstring is: {module_doc}\n\n"
                  f"Write the docstring for this {kind}:\n\n```python\n{segment}\n```")
        reply = chat(model_label, [{"role": "system", "content": DOCSTRING_SYSTEM},
                                   {"role": "user", "content": prompt}], temperature=0.2)
        text = clean_docstring(reply)
        first_statement = node.body[0]
        line = lines[first_statement.lineno - 1]
        indent = line[: len(line) - len(line.lstrip())]
        insertions.append((first_statement.lineno - 1, docstring_lines(text, indent)))
        log.append(f"{node.name}: {text.splitlines()[0] if text else '(empty)'}")
        yield log, source

    # Insert from the bottom up so earlier line numbers stay valid
    for index, new_lines in sorted(insertions, reverse=True):
        lines[index:index] = new_lines
    documented = "\n".join(lines) + ("\n" if source.endswith("\n") else "")

    if without_docstrings(documented) != without_docstrings(source):
        log.append("✗ The code changed, which should be impossible - not saving.")
        yield log, source
        return
    log.append("✓ Checked: apart from docstrings, the syntax tree is identical to the original.")
    yield log, documented


# ------------------------------------------------------------------ unit tests


TEST_SYSTEM = """You are an expert Python developer writing pytest unit tests for a Spanish-tutor project.
Rules:
- Import the module under test by its module name; the test file sits next to it.
- Plain pytest functions, small and focused, each named for the behaviour it checks.
- Test what the code actually does: return values, edge cases, error messages and limits.
- For long return strings, assert on the parts that matter (`"hablaría" in result`,
  `result.startswith(...)`) rather than retyping the whole string.
- Never call a real LLM, Ollama, the network, Gradio's launch, or input(). Use monkeypatch to replace a
  client if a function needs one, or leave that function untested.
- Anything that writes files must use pytest's tmp_path, monkeypatching module constants such as directories.
- Tests must be deterministic.
Reply with the complete test file in one ```python code block and nothing else."""

TRIAGE_SYSTEM = """You review one failing pytest test against the module it tests, and decide which one is wrong.
The module is the specification: usually the test is wrong because it misread the code. Before answering,
trace the module by hand for exactly the inputs the test uses, and compare with the failure message.
- If the test is wrong, reply with the corrected test function.
- Only if the module clearly contradicts its own docstring or obvious intent, reply with the original
  test decorated @pytest.mark.xfail(reason="likely bug: <what is wrong>").
Reply with one ```python code block holding just that one test function (same name), plus any imports it needs."""

MAX_DRAFTS = 2
MAX_REPAIR_ROUNDS = 3


def extract_python(reply):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", reply, flags=re.DOTALL)
    return (max(blocks, key=len) if blocks else reply.replace("```", "")).strip() + "\n"


def summarize(pytest_output):
    """Counts from pytest's last line, e.g. {'passed': 12, 'xfailed': 1}."""
    counts = {}
    for number, outcome in re.findall(r"(\d+) (passed|failed|errors?|xfailed|xpassed|skipped)", pytest_output):
        counts[outcome.rstrip("s") if outcome.startswith("error") else outcome] = int(number)
    return counts


def run_tests(test_code, module_name, module_source, source_dir=None):
    """Run the tests against a copy of the module in a temporary folder. Returns (counts, output)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if source_dir:  # siblings too, in case the module imports one of them
            for sibling in source_dir.glob("*.py"):
                shutil.copy(sibling, tmp / sibling.name)
        (tmp / f"{module_name}.py").write_text(module_source, encoding="utf-8")
        (tmp / f"test_{module_name}.py").write_text(test_code, encoding="utf-8")
        command = [sys.executable, "-m", "pytest", "-vv", "-rfE", "--tb=short", "-p", "no:cacheprovider",
                   f"test_{module_name}.py"]
        try:
            result = subprocess.run(command, cwd=tmp, text=True, capture_output=True, timeout=TEST_TIMEOUT)
        except subprocess.TimeoutExpired:
            return {"error": 1}, f"The tests were still running after {TEST_TIMEOUT} seconds."
    output = (result.stdout + result.stderr).replace(f"{tmp}/", "")
    if "No module named pytest" in output:
        raise gr.Error("pytest is not installed: run `uv add pytest` (or `pip install pytest`).")
    return summarize(output), output


def all_good(counts):
    return counts.get("passed", 0) > 0 and not counts.get("failed") and not counts.get("error")


def failing_tests(output):
    """Names of the tests pytest reported as FAILED or ERROR, in order, without duplicates."""
    names = re.findall(r"^(?:FAILED|ERROR) [^\s:]+::(\w+)", output, flags=re.MULTILINE)
    return list(dict.fromkeys(names))


def failure_report(output, name):
    """pytest's section for one test: from its ____ name ____ header to the next header."""
    match = re.search(rf"^_+ (?:ERROR at \w+ of )?{name}(?:\[[^\]]*\])? _+$(.*?)(?=^_{{3,}} |^={{3,}} )",
                      output, flags=re.MULTILINE | re.DOTALL)
    return match.group(1).strip()[:3000] if match else output[-3000:]


def find_test(tree, name):
    return next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), None)


def span(node):
    """0-based (start, end) line range of a function, decorators included."""
    start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
    return start, node.end_lineno


def replace_test(test_code, name, reply_code):
    """Swap one test function for the model's new version, adding any imports the new version brought."""
    new_tree = ast.parse(reply_code)
    new_node = find_test(new_tree, name)
    if new_node is None:
        raise ValueError(f"the reply has no function called {name}")
    new_lines = reply_code.splitlines()
    start, end = span(new_node)
    replacement = new_lines[start:end]

    lines = test_code.splitlines()
    old_start, old_end = span(find_test(ast.parse(test_code), name))
    lines[old_start:old_end] = replacement

    imports = [ast.get_source_segment(reply_code, n) for n in new_tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    missing = [line for line in imports if line not in lines]
    return "\n".join(missing + lines) + "\n"


def drop_tests(test_code, names):
    """Remove tests the model never managed to fix.

    In practice these are wrong tests, not bugs: on week1's normalize_csv, qwen2.5:14b kept feeding in
    JSON-style rows and mispredicting CSV quoting through three rounds, while the code was fine. A model
    that thinks the code is wrong says so with xfail("likely bug: ..."), and those tests are kept.
    """
    lines = test_code.splitlines()
    tree = ast.parse(test_code)
    nodes = [node for name in names if (node := find_test(tree, name))]
    for node in sorted(nodes, key=lambda n: span(n)[0], reverse=True):
        start, end = span(node)
        del lines[start:end]
    return re.sub(r"\n{3,}", "\n\n\n", "\n".join(lines)) + "\n"


ISOLATION_FIXTURE = """

# Added by tutor_code_assistant: whatever the tests do, files go to a temporary folder,
# never into the project (save_flashcards would otherwise write into the real week2/decks/).
@pytest.fixture(autouse=True)
def _keep_files_out_of_the_project(monkeypatch, tmp_path):
    import pathlib
    import {module} as module_under_test
    for name, value in list(vars(module_under_test).items()):
        if name.endswith("_DIR") and isinstance(value, pathlib.Path):
            monkeypatch.setattr(module_under_test, name, tmp_path / name.lower())
"""


def with_isolation(test_code, module_name):
    """Put the isolation fixture after the imports at the top of the test file."""
    if "_keep_files_out_of_the_project" in test_code:
        return test_code
    tree = ast.parse(test_code)
    lines = test_code.splitlines()
    after_imports = max((n.end_lineno for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))), default=0)
    fixture = ISOLATION_FIXTURE.format(module=module_name).splitlines()
    if "import pytest" not in lines:
        fixture = ["import pytest"] + fixture
    lines[after_imports:after_imports] = fixture
    return "\n".join(lines) + "\n"


def draft_tests(source, module_name, model_label, source_dir, log):
    """Ask for a whole test file until one parses and pytest can collect it. Returns (test_code, counts, output)."""
    messages = [
        {"role": "system", "content": TEST_SYSTEM},
        {"role": "user", "content": f"Write unit tests for the module `{module_name}`:\n\n```python\n{source}\n```"},
    ]
    test_code, counts, output = "", {}, ""
    for attempt in range(1, MAX_DRAFTS + 1):
        reply = chat(model_label, messages, temperature=0.2)
        messages.append({"role": "assistant", "content": reply})
        test_code = extract_python(reply)
        try:
            test_code = with_isolation(test_code, module_name)
        except SyntaxError as e:
            log.append(f"Draft {attempt}: not valid Python ({e})")
            messages.append({"role": "user", "content": f"That is not valid Python ({e}). Reply with the complete corrected file."})
            continue
        counts, output = run_tests(test_code, module_name, source, source_dir)
        if counts.get("passed") or counts.get("failed"):
            return test_code, counts, output
        log.append(f"Draft {attempt}: pytest could not run it")
        messages.append({"role": "user", "content": f"pytest could not run that file:\n\n{output[-4000:]}\n\nReply with the complete corrected file."})
    return test_code, counts, output


def write_tests(source, module_name, model_label, source_dir=None):
    """Generator yielding (log, test_code, pytest_output) while tests are written, run and repaired.

    The model drafts a whole file once. After that it is shown one failing test at a time - the test, its
    failure and the module - which a small model handles far better than a wall of pytest output.
    """
    log = [f"Asking {model_label} for a test file..."]
    yield log, "", ""
    test_code, counts, output = draft_tests(source, module_name, model_label, source_dir, log)
    if not (counts.get("passed") or counts.get("failed")):
        log.append("✗ Could not get a test file that pytest can run.")
        yield log, test_code, output
        return

    def tally():
        return ", ".join(f"{n} {k}" for k, n in counts.items())

    log.append(f"Draft: {tally()}")
    yield log, test_code, output

    for repair_round in range(1, MAX_REPAIR_ROUNDS + 1):
        failing = failing_tests(output)
        if all_good(counts) or not failing:
            break
        log.append(f"Repair round {repair_round}: {len(failing)} failing")
        for name in failing:
            test_tree = ast.parse(test_code)
            node = find_test(test_tree, name)
            if node is None:
                continue
            start, end = span(node)
            prompt = (f"The module `{module_name}`:\n\n```python\n{source}\n```\n\n"
                      f"The failing test:\n\n```python\n{chr(10).join(test_code.splitlines()[start:end])}\n```\n\n"
                      f"pytest says:\n\n{failure_report(output, name)}")
            reply = chat(model_label, [{"role": "system", "content": TRIAGE_SYSTEM},
                                       {"role": "user", "content": prompt}], temperature=0.2)
            try:
                test_code = replace_test(test_code, name, extract_python(reply))
                verdict = "marked as a likely bug" if "likely bug" in reply else "rewritten"
            except (SyntaxError, ValueError) as e:
                verdict = f"left alone ({e})"
            log.append(f"  {name}: {verdict}")
            yield log, test_code, output
        counts, output = run_tests(test_code, module_name, source, source_dir)
        log.append(f"  → {tally()}")
        yield log, test_code, output

    if not all_good(counts) and (unresolved := failing_tests(output)):
        test_code = drop_tests(test_code, unresolved)
        counts, output = run_tests(test_code, module_name, source, source_dir)
        log.append(f"Dropped {len(unresolved)} test(s) the model could not make agree with the code: {', '.join(unresolved)}")

    likely_bugs = test_code.count("likely bug")
    log.append(f"{'✓' if all_good(counts) else '✗'} Final: {tally()}"
               + (f" - {likely_bugs} test(s) flag a likely bug in the module" if likely_bugs else ""))
    yield log, test_code, output


# ------------------------------------------------------------------ the app


def load_source(choice):
    return "" if choice == PASTED else SOURCES[choice].read_text(encoding="utf-8")


def module_name_for(choice):
    return SOURCES[choice].stem if choice in SOURCES else "pasted_module"


def save(name, text):
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / name
    path.write_text(text, encoding="utf-8")
    return path


def document(choice, source, model_label):
    if not source.strip():
        raise gr.Error("Pick a file or paste some code first.")
    try:
        ast.parse(source)
    except SyntaxError as e:
        raise gr.Error(f"That isn't valid Python: {e}")
    documented = source
    for log, documented in add_docstrings(source, model_label):
        yield documented, "\n".join(log), "", None
    if documented == source:
        return
    diff = "".join(difflib.unified_diff(source.splitlines(True), documented.splitlines(True), "original", "documented"))
    path = save(f"{module_name_for(choice)}_documented.py", documented)
    yield documented, "\n".join(log) + f"\nSaved to {path.relative_to(ROOT)}", diff, str(path)


def test(choice, source, model_label):
    if not source.strip():
        raise gr.Error("Pick a file or paste some code first.")
    module = module_name_for(choice)
    source_dir = SOURCES[choice].parent if choice in SOURCES else None
    test_code = ""
    for log, test_code, output in write_tests(source, module, model_label, source_dir):
        yield test_code, "\n".join(log), output, None
    if not test_code:
        return
    run_hint = f"PYTHONPATH={source_dir.relative_to(ROOT)} " if source_dir else ""
    header = (f"# Generated by week4/tutor_code_assistant.py ({model_label}).\n"
              f"# Run from the repo root: {run_hint}python -m pytest week4/generated/test_{module}.py\n\n")
    path = save(f"test_{module}.py", header + test_code)
    yield test_code, "\n".join(log) + f"\nSaved to {path.relative_to(ROOT)}", output, str(path)


def build_ui():
    models = available_models()
    first = list(SOURCES)[1]
    with gr.Blocks(title="Spanish tutor code assistant") as ui:
        gr.Markdown("## 🇪🇸 Spanish tutor code assistant\n"
                    "Docstrings and unit tests for the tutor's code, written by an open-source model and "
                    "checked before they're saved: docstrings must leave the code untouched, and tests must pass.")
        with gr.Row():
            choice = gr.Dropdown(list(SOURCES) + [PASTED], value=first, label="Code")
            model = gr.Dropdown(models, value=preferred_model(models), label="Model")
        with gr.Row(equal_height=True):
            source = gr.Code(SOURCES[first].read_text(encoding="utf-8"), language="python", label="Original", lines=28)
            result = gr.Code(language="python", label="Result", lines=28)
        with gr.Row():
            docs_btn = gr.Button("Add docstrings", variant="primary")
            tests_btn = gr.Button("Write unit tests", variant="primary")
        with gr.Row(equal_height=True):
            log = gr.Textbox(label="Log", lines=10)
            details = gr.Textbox(label="Diff / pytest output", lines=10)
        file = gr.File(label="Saved file")

        choice.change(load_source, choice, source)
        docs_btn.click(document, [choice, source, model], [result, log, details, file])
        tests_btn.click(test, [choice, source, model], [result, log, details, file])
    return ui


if __name__ == "__main__":
    build_ui().launch(inbrowser=True)
