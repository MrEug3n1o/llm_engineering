"""Week 5's RAG pipeline on a Mac with no API keys: the course's ingest, answer and evaluation code, rebuilt on
local models so that days 1-5 (and the week's exercise) can run end to end.

What replaces what:

- OpenAI embeddings -> sentence-transformers encoders running on the Mac (MiniLM from day 2, BGE as the upgrade)
- gpt-4.1-nano -> qwen2.5:14b on Ollama, called through the native /api/chat so that `num_ctx` is honoured
  (the /v1 endpoint silently truncates at 4096 tokens, and a reranking prompt with 20 chunks is bigger than that)
- litellm's `response_format=SomeModel` -> Ollama's `format` parameter with the model's JSON schema

The pieces mirror the course modules: `implementation/` (day 3 basic RAG), `evaluation/eval.py` (day 4 metrics and
LLM judge) and `pro_implementation/` (day 5 LLM chunking, query rewriting and reranking). A RAG pipeline is a `RAG`
object, so the evaluation can compare configurations side by side instead of editing module constants.
"""

import json
import math
from functools import cache
from pathlib import Path

import requests
from chromadb import PersistentClient
from pydantic import BaseModel, Field
from tqdm import tqdm

HERE = Path(__file__).parent
KNOWLEDGE_BASE = HERE / "knowledge-base"
DB_PATH = HERE / "local_db"

OLLAMA_API = "http://localhost:11434/api"
CHAT_MODEL = "qwen2.5:14b"
NUM_CTX = 16384
REQUEST_TIMEOUT = 900

# label -> (Hugging Face model, prefix the model expects on search queries)
ENCODERS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", ""),
    "bge": ("BAAI/bge-base-en-v1.5", "Represent this sentence for searching relevant passages: "),
}
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ---------------------------------------------------------------------------------------------------------------
# Models


def ollama_running():
    try:
        return requests.get(f"{OLLAMA_API}/tags", timeout=2).ok
    except requests.RequestException:
        return False


def llm(messages, schema=None, model=CHAT_MODEL, temperature=0):
    """Chat with a local Ollama model. With a pydantic `schema`, the reply is constrained to it and parsed."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": NUM_CTX, "temperature": temperature},
    }
    if schema is not None:
        payload["format"] = schema.model_json_schema()
    response = requests.post(f"{OLLAMA_API}/chat", json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    content = response.json()["message"]["content"]
    return schema.model_validate_json(content) if schema is not None else content


@cache
def sentence_model(name):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


@cache
def cross_encoder():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(CROSS_ENCODER)


def embed(texts, encoder="minilm", query=False):
    """Unit-length vectors for the texts. Queries get the prefix some encoders (BGE) are trained with."""
    name, prefix = ENCODERS[encoder]
    if query:
        texts = [prefix + text for text in texts]
    return sentence_model(name).encode(texts, normalize_embeddings=True, batch_size=32).tolist()


# ---------------------------------------------------------------------------------------------------------------
# Ingest: documents -> chunks


class Result(BaseModel):
    """The course's stand-in for LangChain's Document."""

    page_content: str
    metadata: dict


def fetch_documents(knowledge_base=KNOWLEDGE_BASE):
    """Every markdown file in the knowledge base, typed by the folder it sits in."""
    documents = []
    for folder in sorted(Path(knowledge_base).iterdir()):
        if not folder.is_dir():
            continue
        for file in sorted(folder.rglob("*.md")):
            documents.append({
                "type": folder.name,
                "source": file.relative_to(Path(knowledge_base).parent).as_posix(),
                "text": file.read_text(encoding="utf-8"),
            })
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Day 2's RecursiveCharacterTextSplitter, keeping the type and source on every chunk."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for document in documents:
        metadata = {"type": document["type"], "source": document["source"]}
        chunks.extend(Result(page_content=text, metadata=metadata) for text in splitter.split_text(document["text"]))
    return chunks


class Chunk(BaseModel):
    headline: str = Field(description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query")
    summary: str = Field(description="A few sentences summarizing the content of this chunk to answer common questions")
    original_text: str = Field(description="The original text of this chunk from the provided document, exactly as is, not changed in any way")

    def as_result(self, document):
        metadata = {"source": document["source"], "type": document["type"]}
        return Result(page_content=f"{self.headline}\n\n{self.summary}\n\n{self.original_text}", metadata=metadata)


class Chunks(BaseModel):
    chunks: list[Chunk]


def llm_chunk_document(document, average_chunk_size=500, model=CHAT_MODEL):
    """Day 5 exactly: the LLM decides where to cut and writes a headline and summary for each piece.

    The LLM has to repeat the whole document back as `original_text`, so this costs a few minutes per document
    on a 14B local model - fine for one document, too slow for the whole knowledge base (see `enrich_chunks`).
    """
    how_many = len(document["text"]) // average_chunk_size + 1
    prompt = f"""
You take a document and you split the document into overlapping chunks for a KnowledgeBase.

The document is from the shared drive of a company called Insurellm.
The document is of type: {document["type"]}
The document has been retrieved from: {document["source"]}

A chatbot will use these chunks to answer questions about the company.
You should divide up the document as you see fit, being sure that the entire document is returned in the chunks - don't leave anything out.
This document should probably be split into {how_many} chunks, but you can have more or less as appropriate.
There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

For each chunk, you should provide a headline, a summary, and the original text of the chunk.
Together your chunks should represent the entire document with overlap.

Here is the document:

{document["text"]}

Respond with the chunks.
"""
    chunks = llm([{"role": "user", "content": prompt}], schema=Chunks, model=model).chunks
    return [chunk.as_result(document) for chunk in chunks]


class Heading(BaseModel):
    headline: str = Field(description="A brief heading for the chunk, a few words, most likely to be surfaced in a query")
    summary: str = Field(description="One or two sentences summarizing the chunk, naming the people, products and companies in it")


class Headings(BaseModel):
    headings: list[Heading]


def enrich_document(document, pieces, model=CHAT_MODEL):
    """Headline and summary for each of a document's pieces, in one call. The pieces stay word for word."""
    numbered = "\n\n".join(f"### PIECE {i + 1}\n{piece}" for i, piece in enumerate(pieces))
    prompt = f"""
This document of type "{document["type"]}" comes from the shared drive of a company called Insurellm
({document["source"]}). It has been cut into {len(pieces)} pieces for a knowledge base that a chatbot searches.

For every piece, in order, write a headline and a short summary that will help a search surface it. Always name
the person, product, client or contract the piece is about, even when the piece itself only says "she" or "the client".

The whole document, for context:

{document["text"]}

The pieces:

{numbered}

Respond with exactly {len(pieces)} headings, one per piece, in the same order.
"""
    headings = llm([{"role": "user", "content": prompt}], schema=Headings, model=model).headings
    if len(headings) != len(pieces):
        raise ValueError(f"expected {len(pieces)} headings, got {len(headings)}")
    return headings


def enrich_chunks(documents, chunk_size=500, chunk_overlap=100, model=CHAT_MODEL, cache_file=None):
    """The affordable version of day 5's pre-processing.

    Text is split deterministically (so nothing is lost or paraphrased), and the LLM only writes the headline and
    summary that get prepended to each chunk - a few dozen output tokens per chunk instead of the whole document.
    Results are cached as JSON because this is the slow step (about 50 minutes for the Insurellm knowledge base on qwen2.5:14b).
    """
    if cache_file and Path(cache_file).exists():
        return [Result(**item) for item in json.loads(Path(cache_file).read_text())]

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    results = []
    for document in tqdm(documents, desc="Enriching"):
        pieces = splitter.split_text(document["text"])
        metadata = {"source": document["source"], "type": document["type"]}
        try:
            headings = enrich_document(document, pieces, model=model)
        except Exception as error:  # a wrong count or bad JSON: fall back to the document's title for each piece
            print(f"  {document['source']}: {error}; using the file name as headline")
            title = Path(document["source"]).stem
            headings = [Heading(headline=title, summary="") for _ in pieces]
        for heading, piece in zip(headings, pieces):
            text = f"{heading.headline}\n\n{heading.summary}\n\n{piece}".replace("\n\n\n\n", "\n\n")
            results.append(Result(page_content=text, metadata=metadata))

    if cache_file:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_file).write_text(json.dumps([r.model_dump() for r in results], ensure_ascii=False, indent=1))
    return results


# ---------------------------------------------------------------------------------------------------------------
# Vector store


class Store:
    """A Chroma collection plus the encoder its vectors came from (queries must use the same one)."""

    def __init__(self, name, encoder="minilm", path=DB_PATH):
        self.name = name
        self.encoder = encoder
        self.collection = PersistentClient(path=str(path)).get_or_create_collection(
            name, metadata={"hnsw:space": "cosine"}
        )

    def count(self):
        return self.collection.count()

    def search(self, question, k=10):
        query = embed([question], self.encoder, query=True)[0]
        results = self.collection.query(query_embeddings=[query], n_results=k)
        return [
            Result(page_content=text, metadata=meta)
            for text, meta in zip(results["documents"][0], results["metadatas"][0])
        ]

    def everything(self):
        return self.collection.get(include=["embeddings", "documents", "metadatas"])


def build_store(name, chunks, encoder="minilm", path=DB_PATH, rebuild=False):
    """Embed the chunks into a fresh collection, or reuse an existing one that already holds them."""
    client = PersistentClient(path=str(path))
    if name in [c.name for c in client.list_collections()]:
        existing = client.get_collection(name)
        if not rebuild and existing.count() == len(chunks):
            return Store(name, encoder, path)
        client.delete_collection(name)

    store = Store(name, encoder, path)
    texts = [chunk.page_content for chunk in chunks]
    vectors = embed(texts, encoder)
    store.collection.add(
        ids=[str(i) for i in range(len(chunks))],
        embeddings=vectors,
        documents=texts,
        metadatas=[chunk.metadata for chunk in chunks],
    )
    return store


# ---------------------------------------------------------------------------------------------------------------
# Answering


SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{context}

With this context, please answer the user's question. Be accurate, relevant and complete.
"""


class RankOrder(BaseModel):
    order: list[int] = Field(description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number")


def rerank_with_llm(question, chunks, model=CHAT_MODEL):
    """Day 5's LLM reranker. Local models sometimes drop or repeat ids, so the order is repaired, not trusted."""
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\nHere are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    order = llm(messages, schema=RankOrder, model=model).order

    seen, ranked = set(), []
    for i in order:
        if 1 <= i <= len(chunks) and i not in seen:
            seen.add(i)
            ranked.append(chunks[i - 1])
    ranked.extend(chunk for i, chunk in enumerate(chunks, start=1) if i not in seen)
    return ranked


def rerank_with_cross_encoder(question, chunks):
    """A small model trained for exactly this job: it scores each (question, chunk) pair. Milliseconds, not seconds."""
    scores = cross_encoder().predict([(question, chunk.page_content) for chunk in chunks])
    return [chunk for _, chunk in sorted(zip(scores, chunks), key=lambda pair: -pair[0])]


def rewrite_query(question, history=None, model=CHAT_MODEL):
    """Day 5's query rewriter: a short, specific search query, with pronouns resolved from the history."""
    history = history or []
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history) or "(none)"
    message = f"""
You are in a conversation with a user.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{transcript}

And this is the user's current question:
{question}

Since the conversation is contextual, understand the meaning of the user question and add details based on the history.
Condense everything in a single contextually-rich VERY short and specific question, most likely to surface content.

IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.
"""
    return llm([{"role": "system", "content": message}], model=model).strip().strip('"')


def merge_chunks(first, second):
    merged = first[:]
    seen = {chunk.page_content for chunk in first}
    merged.extend(chunk for chunk in second if chunk.page_content not in seen)
    return merged


class RAG:
    """One RAG configuration.

    - `rewrite`: also search with an LLM-rewritten query and merge the two result lists (day 5)
    - `rerank`: None, "llm" (day 5) or "cross-encoder"
    - `retrieval_k` chunks are fetched per query; `final_k` go to the answering model
    """

    def __init__(self, store, rewrite=False, rerank=None, retrieval_k=10, final_k=10, model=CHAT_MODEL,
                 system_prompt=SYSTEM_PROMPT):
        self.store = store
        self.rewrite = rewrite
        self.rerank = rerank
        self.retrieval_k = retrieval_k
        self.final_k = final_k
        self.model = model
        self.system_prompt = system_prompt
        self.last_query = None

    def __repr__(self):
        return (f"RAG({self.store.name}, rewrite={self.rewrite}, rerank={self.rerank}, "
                f"k={self.retrieval_k}->{self.final_k})")

    def fetch_context(self, question, history=None):
        chunks = self.store.search(question, self.retrieval_k)
        self.last_query = question
        if self.rewrite:
            self.last_query = rewrite_query(question, history, self.model)
            chunks = merge_chunks(chunks, self.store.search(self.last_query, self.retrieval_k))
        # The course reranks against the original question. For a follow-up such as "and the other one?" that
        # question means nothing without the history, so rank against the rewritten, self-contained query instead.
        rank_by = self.last_query if history else question
        if self.rerank == "llm":
            chunks = rerank_with_llm(rank_by, chunks, self.model)
        elif self.rerank == "cross-encoder":
            chunks = rerank_with_cross_encoder(rank_by, chunks)
        return chunks[: self.final_k]

    def messages(self, question, history, chunks):
        context = "\n\n".join(f"Extract from {c.metadata['source']}:\n{c.page_content}" for c in chunks)
        system = self.system_prompt.format(context=context)
        return [{"role": "system", "content": system}] + list(history or []) + [{"role": "user", "content": question}]

    def answer_question(self, question, history=None):
        """The same contract as the course's answer_question: (answer, retrieved chunks)."""
        history = [{"role": m["role"], "content": m["content"]} for m in (history or [])]
        chunks = self.fetch_context(question, history)
        return llm(self.messages(question, history, chunks), model=self.model), chunks


# ---------------------------------------------------------------------------------------------------------------
# Evaluation (day 4): the course's metrics, but taking the pipeline as an argument


class TestQuestion(BaseModel):
    question: str
    keywords: list[str]
    reference_answer: str
    category: str


def load_tests(path=HERE / "evaluation" / "tests.jsonl"):
    with open(path, encoding="utf-8") as f:
        return [TestQuestion(**json.loads(line)) for line in f if line.strip()]


def reciprocal_rank(keyword, docs):
    keyword = keyword.lower()
    for rank, doc in enumerate(docs, start=1):
        if keyword in doc.page_content.lower():
            return 1.0 / rank
    return 0.0


def ndcg(keyword, docs, k=10):
    keyword = keyword.lower()
    relevances = [1 if keyword in doc.page_content.lower() else 0 for doc in docs[:k]]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))
    ideal = sum(rel / math.log2(i + 2) for i, rel in enumerate(sorted(relevances, reverse=True)))
    return dcg / ideal if ideal else 0.0


def evaluate_retrieval(rag, test, k=10):
    docs = rag.fetch_context(test.question)
    ranks = [reciprocal_rank(keyword, docs) for keyword in test.keywords]
    found = sum(1 for r in ranks if r > 0)
    return {
        "mrr": sum(ranks) / len(ranks),
        "ndcg": sum(ndcg(keyword, docs, k) for keyword in test.keywords) / len(test.keywords),
        "coverage": 100 * found / len(test.keywords),
    }


class AnswerEval(BaseModel):
    feedback: str = Field(description="Concise feedback on the answer quality, comparing it to the reference answer")
    accuracy: float = Field(description="1 (wrong - any wrong answer must score 1) to 5 (perfectly accurate). An acceptable answer scores 3.")
    completeness: float = Field(description="1 (missing key information) to 5 (ALL the information in the reference answer is provided)")
    relevance: float = Field(description="1 (off-topic) to 5 (directly answers the question and gives no additional information)")


def judge(test, generated_answer, model=CHAT_MODEL):
    """Day 4's LLM-as-a-judge prompt. Locally the judge is usually the same model that answered - keep that bias in mind."""
    messages = [
        {"role": "system", "content": "You are an expert evaluator assessing the quality of answers. Evaluate the generated answer by comparing it to the reference answer. Only give 5/5 scores for perfect answers."},
        {"role": "user", "content": f"""Question:
{test.question}

Generated Answer:
{generated_answer}

Reference Answer:
{test.reference_answer}

Please evaluate the generated answer on three dimensions:
1. Accuracy: How factually correct is it compared to the reference answer? Only give 5/5 scores for perfect answers.
2. Completeness: How thoroughly does it address all aspects of the question, covering all the information from the reference answer?
3. Relevance: How well does it directly answer the specific question asked, giving no additional information?

Provide detailed feedback and scores from 1 (very poor) to 5 (ideal) for each dimension. If the answer is wrong, then the accuracy score must be 1."""},
    ]
    return llm(messages, schema=AnswerEval, model=model)


def evaluate_answer(rag, test, judge_model=CHAT_MODEL):
    answer, chunks = rag.answer_question(test.question)
    return judge(test, answer, judge_model), answer, chunks


def retrieval_report(rag, tests, k=10):
    """One row per test question with its retrieval metrics."""
    import pandas as pd

    rows = []
    for test in tqdm(tests, desc=f"Retrieval {rag.store.name}"):
        rows.append({"question": test.question, "category": test.category, **evaluate_retrieval(rag, test, k)})
    return pd.DataFrame(rows)


def answer_report(rag, tests, judge_model=CHAT_MODEL):
    """One row per test question with the generated answer and the judge's scores."""
    import pandas as pd

    rows = []
    for test in tqdm(tests, desc="Answers"):
        verdict, answer, _ = evaluate_answer(rag, test, judge_model)
        rows.append({
            "question": test.question, "category": test.category, "answer": answer,
            "accuracy": verdict.accuracy, "completeness": verdict.completeness, "relevance": verdict.relevance,
            "feedback": verdict.feedback,
        })
    return pd.DataFrame(rows)


def sample_tests(tests, per_category=3):
    """The first few questions of each category: a fair spread when the full set would take hours locally."""
    taken, sample = {}, []
    for test in tests:
        if taken.get(test.category, 0) < per_category:
            taken[test.category] = taken.get(test.category, 0) + 1
            sample.append(test)
    return sample


# ---------------------------------------------------------------------------------------------------------------
# The Insurellm stores used across the notebook


def insurellm_store(variant="minilm-1000", rebuild=False):
    """The knowledge base chunked and embedded one of the ways the week tries.

    - "minilm-1000": day 2 - 1000-character chunks, all-MiniLM-L6-v2
    - "minilm-500" / "bge-500": day 4's chunk-size and encoder experiments
    - "bge-enriched": day 5 - LLM headline + summary on each 500-character chunk, BGE vectors
    """
    documents = fetch_documents()
    if variant == "bge-enriched":
        chunks = enrich_chunks(documents, cache_file=DB_PATH / "enriched_chunks.json")
        return build_store("insurellm-bge-enriched", chunks, "bge", rebuild=rebuild)
    encoder, size = variant.split("-")
    chunks = split_documents(documents, chunk_size=int(size), chunk_overlap=int(size) // 5)
    return build_store(f"insurellm-{variant}", chunks, encoder, rebuild=rebuild)


def format_context(chunks):
    """The retrieved-context panel from the course's app.py."""
    result = "<h2 style='color: #ff7800;'>Relevant Context</h2>\n\n"
    for chunk in chunks:
        result += f"<span style='color: #ff7800;'>Source: {chunk.metadata['source']}</span>\n\n"
        result += chunk.page_content + "\n\n"
    return result


def build_chat_ui(rag, title="Insurellm Expert Assistant", subtitle="Ask me anything about Insurellm!"):
    """The course's app.py layout - chat on the left, the retrieved context on the right - for any RAG object."""
    import gradio as gr

    def put_message_in_chatbot(message, history):
        return "", history + [{"role": "user", "content": message}]

    def chat(history):
        answer, chunks = rag.answer_question(history[-1]["content"], history[:-1])
        return history + [{"role": "assistant", "content": answer}], format_context(chunks)

    with gr.Blocks(title=title, theme=gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])) as ui:
        gr.Markdown(f"# {title}\n{subtitle}")
        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(label="Conversation", height=600, type="messages", show_copy_button=True)
                message = gr.Textbox(placeholder="Ask a question...", show_label=False)
            with gr.Column(scale=1):
                context = gr.Markdown(value="*Retrieved context will appear here*", container=True, height=600)
        message.submit(put_message_in_chatbot, [message, chatbot], [message, chatbot]).then(
            chat, chatbot, [chatbot, context]
        )
    return ui


if __name__ == "__main__":
    store = insurellm_store("bge-enriched" if (DB_PATH / "enriched_chunks.json").exists() else "bge-500")
    build_chat_ui(RAG(store, rewrite=True, rerank="cross-encoder", retrieval_k=20, final_k=10)).launch(inbrowser=True)
