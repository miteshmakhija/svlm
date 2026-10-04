"""Catalogue gateway (M5 preview): serve the live GGUF of each model in the model store on CPU.

    uvicorn gateway.app:app --port 8000          # from the repo root, with requirements/gateway.txt

Endpoints
    GET  /health                    loaded models and versions
    GET  /v1/models                 OpenAI-style model list
    POST /v1/fim                    {"prefix", "suffix", "max_tokens"?, "span_type"?} -> trimmed completion
    POST /v1/completions            OpenAI-compatible (prompt + optional suffix, stream supported), for IDE plugins
    POST /v1/chat/completions       same, for chat-only clients: the last user message is the raw prompt

Settings (environment)
    SVLM_LOG_FILE      per-request log, also printed to the console (default: gateway/logs/gateway.log)
    SVLM_LOG_PROMPTS   1 = also log the line before and after the cursor  (default: 0)
    SVLM_MODEL_STORE   model store folder with catalogue.yaml   (default: ../model_store next to the repo)
    SVLM_THREADS       CPU threads for llama.cpp                 (default: physical cores, llama.cpp's choice)
    SVLM_N_CTX         context window in tokens                  (default: 4096, room for chat history)

Prompts are built and trimmed with the same svlm.fim code the pipeline evaluated, and the prefix/suffix
are cut to the lengths the model was trained on (configs/code_fast.yaml: data.fim.max_*_chars).
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from starlette.concurrency import iterate_in_threadpool
from pydantic import BaseModel, Field

from svlm.chat import CHAT_STOP, format_chat, trim_reply
from svlm.fim import FIM_PREFIX, STOP_STRINGS, fim_prompt, trim_completion

REPO = Path(__file__).resolve().parents[1]
STORE = Path(os.environ.get("SVLM_MODEL_STORE", REPO.parent / "model_store"))
N_CTX = int(os.environ.get("SVLM_N_CTX", "4096"))
THREADS = int(os.environ["SVLM_THREADS"]) if os.environ.get("SVLM_THREADS") else None
MAX_PREFIX_CHARS, MAX_SUFFIX_CHARS = 2200, 900  # as trained (data.fim in configs/code_fast.yaml)
CHAT_TASKS = {"code"}  # catalogue tasks whose models were trained for chat (svlm.chat format)
CHAT_MAX_TOKENS_CAP = 1024
MAX_TOKENS_CAP = 256  # larger requests are clamped, not rejected (IDE plugins often ask for more)
LOG_FILE = Path(os.environ.get("SVLM_LOG_FILE", REPO / "gateway" / "logs" / "gateway.log"))
LOG_PROMPTS = os.environ.get("SVLM_LOG_PROMPTS", "0") == "1"  # also log the code around the cursor
log = logging.getLogger("svlm.gateway")


def _setup_logging() -> None:
    """One line per request to the console and to LOG_FILE (tail it while you type in the IDE)."""
    if log.handlers:
        return
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    for h in (logging.StreamHandler(), logging.FileHandler(LOG_FILE, encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.INFO)
    log.propagate = False


def _preview(text: str, n: int = 80) -> str:
    # ascii() keeps it on one log line (newlines show as \n) and is safe for any console encoding
    return ascii(text if len(text) <= n else text[:n] + "...")


@dataclass
class Served:
    name: str
    version: str
    task: str
    gguf: Path
    llm: object = None
    lock: threading.Lock = field(default_factory=threading.Lock)  # llama.cpp contexts are not thread-safe


MODELS: dict[str, Served] = {}


def load_catalogue(store: Path) -> dict[str, Served]:
    cat = yaml.safe_load((store / "catalogue.yaml").read_text())
    out = {}
    for name, m in (cat.get("models") or {}).items():
        live = m.get("live")
        if not live:
            continue
        v = m["versions"][live]
        out[name] = Served(name=name, version=live, task=v.get("task", "fim"), gguf=store / v["path"] / v["gguf"])
    return out


@asynccontextmanager
async def lifespan(_: FastAPI):
    from llama_cpp import Llama

    _setup_logging()
    for name, s in load_catalogue(STORE).items():
        if not s.gguf.exists():
            raise RuntimeError(f"{name}:{s.version}: {s.gguf} not found")
        kwargs = {"n_threads": THREADS} if THREADS else {}
        s.llm = Llama(model_path=str(s.gguf), n_ctx=N_CTX, verbose=False, **kwargs)
        MODELS[name] = s
        log.info("loaded %s:%s from %s (n_ctx %d, threads %s)", name, s.version, s.gguf, N_CTX, THREADS or "auto")
    log.info("logging to %s%s", LOG_FILE, " (prompts included)" if LOG_PROMPTS else "")
    yield
    MODELS.clear()


app = FastAPI(title="SVLM catalogue gateway", version="0.1.0", lifespan=lifespan)


def _model(name: str | None) -> Served:
    if not MODELS:
        raise HTTPException(503, "no models loaded")
    if name is None:
        return next(iter(MODELS.values()))
    if name not in MODELS:
        raise HTTPException(404, f"unknown model {name!r}; available: {sorted(MODELS)}")
    return MODELS[name]


def _fim(prefix: str, suffix: str) -> str:
    return fim_prompt(prefix[-MAX_PREFIX_CHARS:], suffix[:MAX_SUFFIX_CHARS])


class Generation:
    """One llama.cpp generation, run on its own thread; iterate it for text pieces.

    The worker holds the model lock and checks `cancel()` after every token, so a client that hangs
    up mid-stream (the IDE cancels whenever you keep typing) frees the model at once instead of
    blocking every later request. Logs one line when the request starts and one when it ends: queue
    wait, prompt tokens, time to first token, generated tokens, tokens/s, finish reason and the start
    of the completion.
    """

    _END = object()

    def __init__(self, s: Served, prompt: str, max_tokens: int, temperature: float, stop: list[str], kind: str,
                 cap: int = MAX_TOKENS_CAP):
        self.s, self.prompt, self.temperature, self.stop = s, prompt, temperature, stop
        self.max_tokens = min(max_tokens, cap)
        self.rid = uuid.uuid4().hex[:6]
        self._q: queue.Queue = queue.Queue()
        self._cancel = threading.Event()
        n_prompt = len(s.llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True))
        fim = FIM_PREFIX in prompt
        log.info("[%s] %s %s | prompt %d tok%s | max_tokens %d", self.rid, kind, s.name, n_prompt,
                 " (FIM)" if fim else "", self.max_tokens)
        if LOG_PROMPTS and fim:
            before, _, after = prompt.partition("<|fim_suffix|>")
            after_lines = [ln for ln in after.replace("<|fim_middle|>", "").splitlines() if ln.strip()]
            log.info("[%s]   before cursor: ...%s", self.rid, _preview(before.splitlines()[-1] if before.strip() else "", 120))
            log.info("[%s]   after cursor : %s", self.rid, _preview(after_lines[0] if after_lines else "", 120))
        threading.Thread(target=self._work, daemon=True).start()

    def cancel(self) -> None:
        self._cancel.set()

    def __iter__(self) -> Iterator[str]:
        try:
            while (item := self._q.get()) is not self._END:
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:  # consumer stopped early (disconnect, error): let the worker release the model
            self.cancel()

    def _work(self) -> None:
        t0 = time.perf_counter()
        ttft = wait = None
        n_tok, parts, finish, state = 0, [], None, "done"
        try:
            with self.s.lock:
                wait = (time.perf_counter() - t0) * 1000
                for chunk in self.s.llm.create_completion(prompt=self.prompt, max_tokens=self.max_tokens,
                                                          temperature=self.temperature, stop=self.stop, stream=True):
                    if self._cancel.is_set():
                        state = "cancelled"
                        break
                    c = chunk["choices"][0]
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                    n_tok += 1
                    parts.append(c["text"])
                    finish = c.get("finish_reason") or finish
                    self._q.put(c["text"])
        except Exception as e:  # surface to the consumer
            state = f"error {type(e).__name__}: {e}"
            self._q.put(e)
        finally:
            self._q.put(self._END)
            total = (time.perf_counter() - t0) * 1000
            gen_s = (total - (ttft or total)) / 1000
            log.info("[%s] %s | %d tok | wait %.0f ms | ttft %s | total %.0f ms | %s | %s | %s", self.rid, state, n_tok,
                     wait or 0, f"{ttft:.0f} ms" if ttft is not None else "-", total,
                     f"{(n_tok - 1) / gen_s:.0f} tok/s" if n_tok > 1 and gen_s > 0 else "- tok/s",
                     finish or ("length" if n_tok >= self.max_tokens else "-"), _preview("".join(parts)))





# ---------------------------------------------------------------------------- native FIM


class FimRequest(BaseModel):
    prefix: str = Field(description="code before the cursor")
    suffix: str = Field("", description="code after the cursor")
    model: str | None = None
    max_tokens: int = Field(64, ge=1)
    span_type: Literal["line", "block", "function_body"] = "block"


class FimResponse(BaseModel):
    model: str
    version: str
    completion: str
    ttft_ms: float
    total_ms: float


@app.post("/v1/fim", response_model=FimResponse)
def fim(req: FimRequest) -> FimResponse:
    s = _model(req.model)
    t0 = time.perf_counter()
    ttft, parts = None, []
    for piece in Generation(s, _fim(req.prefix, req.suffix), req.max_tokens, 0.0, STOP_STRINGS, "fim"):
        if ttft is None:
            ttft = (time.perf_counter() - t0) * 1000
        parts.append(piece)
    total = (time.perf_counter() - t0) * 1000
    return FimResponse(model=s.name, version=s.version, completion=trim_completion("".join(parts), req.span_type, req.suffix),
                       ttft_ms=round(ttft if ttft is not None else total, 1), total_ms=round(total, 1))


# ---------------------------------------------------------------------------- OpenAI-compatible


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    suffix: str | None = None
    max_tokens: int = Field(64, ge=1)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    stop: list[str] | str | None = None
    stream: bool = False


@app.post("/v1/completions")
def completions(req: CompletionRequest):
    s = _model(req.model)
    if req.suffix is not None:
        prompt = _fim(req.prompt, req.suffix)    # prefix + suffix given separately
    else:
        prompt = req.prompt                      # caller already formatted it (FIM tokens or plain code)
    return _openai(s, prompt, req.max_tokens, req.temperature, req.stop, req.stream, req.suffix, chat=False)


class ChatMessage(BaseModel):
    role: str
    content: str | list | None = None


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = Field(None, ge=1)
    max_completion_tokens: int | None = Field(None, ge=1)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    stop: list[str] | str | None = None
    stream: bool = False


def _text(content) -> str:
    """OpenAI message content: a string, or a list of parts of which the text parts are kept."""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in (content or []) if isinstance(p, dict))


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    """Chat models (task `code`): the conversation is formatted with svlm.chat, exactly as in training.
    Completion-only models (task `fim`): the last user message is used as the raw prompt, for clients
    that only speak chat (e.g. an IDE sending an already-formatted FIM prompt)."""
    s = _model(req.model)
    msgs = [{"role": m.role, "content": _text(m.content)} for m in req.messages if m.role in ("system", "user", "assistant")]
    if not any(m["role"] == "user" for m in msgs):
        raise HTTPException(400, "no user message")
    if s.task in CHAT_TASKS:
        n = req.max_tokens or req.max_completion_tokens or 512
        return _openai(s, format_chat(msgs), n, req.temperature, req.stop, req.stream, None, chat=True, chat_model=True)
    n = req.max_tokens or req.max_completion_tokens or 64
    return _openai(s, [m for m in msgs if m["role"] == "user"][-1]["content"], n, req.temperature, req.stop, req.stream,
                   None, chat=True)


def _openai(s: Served, prompt: str, max_tokens: int, temperature: float, stop_extra, stream: bool,
            suffix: str | None, chat: bool, chat_model: bool = False):
    stop = list(CHAT_STOP if chat_model else STOP_STRINGS) + ([stop_extra] if isinstance(stop_extra, str) else (stop_extra or []))
    cid, created = f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex[:24]}", int(time.time())
    kind = ("chat" if chat else "completions") + (" stream" if stream else "")
    pieces = Generation(s, prompt, max_tokens, temperature, stop, kind, cap=CHAT_MAX_TOKENS_CAP if chat_model else MAX_TOKENS_CAP)

    def body(text: str, finish: str | None, delta: bool) -> dict:
        if chat:
            choice = {"index": 0, "finish_reason": finish,
                      **({"delta": {"content": text}} if delta else {"message": {"role": "assistant", "content": text}})}
            return {"id": cid, "object": "chat.completion.chunk" if delta else "chat.completion", "created": created,
                    "model": s.name, "choices": [choice]}
        return {"id": cid, "object": "text_completion", "created": created, "model": s.name,
                "choices": [{"index": 0, "text": text, "finish_reason": finish, "logprobs": None}]}

    if stream:
        def sse() -> Iterator[str]:
            for piece in pieces:
                yield f"data: {json.dumps(body(piece, None, True))}\n\n"
            yield f"data: {json.dumps(body('', 'stop', True))}\n\n"
            yield "data: [DONE]\n\n"

        async def sse_async():
            # on client disconnect Starlette cancels this task; the sync generator would never be
            # closed, so cancel the generation explicitly here
            try:
                async for line in iterate_in_threadpool(sse()):
                    yield line
            finally:
                pieces.cancel()
        return StreamingResponse(sse_async(), media_type="text/event-stream")

    text = "".join(pieces)
    if chat_model:
        text = trim_reply(text)
    elif suffix is not None or FIM_PREFIX in prompt:
        text = trim_completion(text, "block", suffix or "")
    return body(text, "stop", False)


# ---------------------------------------------------------------------------- info


@app.get("/v1/models")
def models() -> dict:
    return {"object": "list", "data": [{"id": s.name, "object": "model", "owned_by": "svlm", "version": s.version}
                                       for s in MODELS.values()]}


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if MODELS else "loading", "store": str(STORE),
            "models": {s.name: {"version": s.version, "task": s.task, "gguf": s.gguf.name} for s in MODELS.values()}}
