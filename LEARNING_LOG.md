# My course work: every week on local models

This fork holds my solutions to Ed Donner's LLM Engineering course. I had no paid API keys, so each week's labs
are rebuilt to run on local open-source models (Ollama, sentence-transformers, PyTorch on the Mac GPU, MLX), and
each week's exercise extends one project: a Spanish tutor.

**The finished tutor, cleaned up into one package with an agent, tests and CI, is its own repository:
[MrEug3n1o/spanish-tutor-ai](https://github.com/MrEug3n1o/spanish-tutor-ai).** This fork is the lab notebook
behind it.

| Week | Labs (local rebuild) | Exercise | Results |
|---|---|---|---|
| 1 | `week1/day*.ipynb` on Ollama | [`spanish_tutor.py`](week1/spanish_tutor.py): CLI tutor that writes Flashcards.com CSV decks | |
| 2 | `week2/day*.ipynb` | [`spanish_tutor_gradio.py`](week2/spanish_tutor_gradio.py) + [`tutor_tools.py`](week2/tutor_tools.py): streaming Gradio tutor, tool calling, voice | |
| 3 | [`local_labs.ipynb`](week3/local_labs.ipynb), [`day5_local.ipynb`](week3/day5_local.ipynb): the Colab-only days on MPS | [`spanish_data_generator.py`](week3/spanish_data_generator.py): validated synthetic datasets | |
| 4 | [`local_labs.ipynb`](week4/local_labs.ipynb) via [`code_porter.py`](week4/code_porter.py): Python → C++ with compile-and-fix loop | [`tutor_code_assistant.py`](week4/tutor_code_assistant.py): docstrings (AST-verified) and pytest suites | qwen2.5:14b ports: **44×** (pi) and **38×** (max subarray) faster than Python |
| 5 | [`local_labs.ipynb`](week5/local_labs.ipynb) via [`local_rag.py`](week5/local_rag.py): BGE, Chroma, query rewriting, rerankers, evals | [`spanish_tutor_rag.py`](week5/spanish_tutor_rag.py): the tutor answers from its notes | reranking: MRR **0.648 → 0.904**; judge accuracy 4.81/5 |
| 6 | [`local_labs.ipynb`](week6/local_labs.ipynb) via [`local_pricer.py`](week6/local_pricer.py): LoRA on Qwen2.5-0.5B instead of OpenAI fine-tuning | [`precio_justo.py`](week6/precio_justo.py): price-guessing game in Spanish | LoRA Qwen 0.5B: **$251 → $62** error, ahead of qwen2.5:14b ($84) |
| 7 | [`local_labs.ipynb`](week7/local_labs.ipynb) via [`local_finetune.py`](week7/local_finetune.py): QLoRA of Llama 3.2 3B on MLX | [`article_finetune.py`](week7/article_finetune.py): el/la classifier from AnCora | pricer **$190 → $56.25** (GPT-4.1 Nano: $62.51); articles **98.9%** vs 70% for llama3.2 |
| 8 | — | The study-coach agent, built directly in [spanish-tutor-ai](https://github.com/MrEug3n1o/spanish-tutor-ai) | see that repo's README |

Every notebook keeps its outputs, so the numbers above can be checked without re-running anything.
