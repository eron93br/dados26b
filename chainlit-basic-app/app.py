"""
RAG Chatbot — versão Chainlit (Groq + ChromaDB)
-----------------------------------------------
LLM via API do Groq (nuvem) e busca semântica num servidor ChromaDB local
(collection "my_collection"). Os embeddings das perguntas são gerados
localmente com FastEmbed — precisa ser o MESMO modelo usado na ingestão.

Configuração: copie .env.example para .env e preencha GROQ_API_KEY.

Rodar:
    chainlit run app.py -w
"""

import os
import time
from dataclasses import dataclass

import chainlit as cl
import chromadb
import groq
from chainlit.input_widget import Select, Slider, Switch
from dotenv import load_dotenv

load_dotenv()

# ── Configuração (sobrescritível via .env) ─────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_TIMEOUT = float(os.getenv("GROQ_TIMEOUT", "120"))

CHAT_MODELS = [
    m.strip()
    for m in os.getenv(
        "GROQ_CHAT_MODELS",
        "openai/gpt-oss-120b,openai/gpt-oss-20b,qwen/qwen3.8-27b",
    ).split(",")
    if m.strip()
]
DEFAULT_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")

CHROMA_HOST     = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT     = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "my_collection")

# Modelo usado na ingestão da my_collection (384 dims, multilíngue)
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

DEFAULT_N_RESULTS   = int(os.getenv("RAG_N_RESULTS", "8"))
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS  = 1024
MAX_CONTEXT_CHARS   = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "6000"))
MAX_HISTORY_TURNS   = int(os.getenv("RAG_MAX_HISTORY_TURNS", "6"))

_STATUS_TTL = 120  # segundos
_status_cache: dict = {"ts": 0.0, "value": None}

_PROMPT_TEMPLATE = """\
Você é um assistente especializado em responder perguntas utilizando exclusivamente \
as informações recuperadas de uma base documental indexada no ChromaDB.

Instruções:
- Utilize SOMENTE as informações presentes no CONTEXTO abaixo.
- Se a resposta não estiver no contexto, responda: \
"Não encontrei informações suficientes na base documental para responder a essa pergunta."
- Nunca invente nomes, datas, valores ou documentos.
- Cite as fontes utilizadas ao final da resposta no formato do contexto (ex.: [1] arquivo.pdf, pág. 3).
- Responda em português, de forma objetiva e organizada.

CONTEXTO:
{context}

{history}PERGUNTA:
{question}
"""


# ── Singletons (ChromaDB, embeddings, Groq) ───────────────────────────────────

_collection = None
_embedder = None
_groq_client: groq.AsyncGroq | None = None


def get_collection():
    """Conecta ao ChromaDB local e retorna a collection (singleton)."""
    global _collection
    if _collection is None:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        # get_collection (não get_or_create): a collection já existe e foi populada
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def get_embedder():
    """Modelo FastEmbed local (ONNX, CPU). Baixado do Hugging Face na 1ª execução."""
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding
        _embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
    return _embedder


def get_groq() -> groq.AsyncGroq:
    global _groq_client
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY não configurada. Crie uma chave em https://console.groq.com/keys "
            "e adicione `GROQ_API_KEY=...` no arquivo .env."
        )
    if _groq_client is None:
        _groq_client = groq.AsyncGroq(api_key=GROQ_API_KEY, timeout=GROQ_TIMEOUT)
    return _groq_client


def _friendly_error(exc: Exception) -> str:
    """Converte exceções do SDK do Groq em mensagens claras para a UI."""
    if isinstance(exc, groq.AuthenticationError):
        return "GROQ_API_KEY inválida. Verifique a chave no arquivo .env."
    if isinstance(exc, groq.RateLimitError):
        return "Limite de requisições/tokens do Groq atingido. Aguarde alguns segundos e tente novamente."
    if isinstance(exc, groq.APITimeoutError):
        return f"Timeout na chamada ao LLM (>{GROQ_TIMEOUT:.0f}s)."
    if isinstance(exc, groq.APIConnectionError):
        return "Não foi possível conectar à API do Groq. Verifique sua conexão com a internet."
    if isinstance(exc, groq.APIStatusError):
        return f"Erro na API do Groq ({exc.status_code}): {exc.message}"
    return str(exc)


# ── Status do Groq ─────────────────────────────────────────────────────────────

async def _check_groq_status() -> dict:
    """Verifica se a API do Groq responde com a chave configurada (cache de _STATUS_TTL s)."""
    now = time.time()
    if _status_cache["value"] is not None and now - _status_cache["ts"] < _STATUS_TTL:
        return _status_cache["value"]

    try:
        models = await get_groq().models.list()
        available = {m.id for m in models.data}
        missing = [m for m in CHAT_MODELS if m not in available]
        status = {"online": True, "detail": f"Modelos indisponíveis: {', '.join(missing)}" if missing else ""}
    except Exception as exc:  # noqa: BLE001
        status = {"online": False, "detail": _friendly_error(exc)}

    _status_cache["ts"] = now
    _status_cache["value"] = status
    return status


async def _groq_status_markdown() -> str:
    status = await _check_groq_status()
    label = "🟢 **Groq Conectado**" if status["online"] else "🔴 **Groq Desconectado**"
    if status["detail"]:
        label += f"  \n_{status['detail']}_"
    return label


# ── Recuperação (ChromaDB) ─────────────────────────────────────────────────────

@dataclass
class Chunk:
    id:      str
    text:    str
    score:   float          # similaridade 0–1 (maior = mais relevante)
    source:  str = ""
    page:    int | None = None
    section: str = ""


def retrieve(query: str, n_results: int) -> list[Chunk]:
    """Busca por similaridade na collection. Síncrono — chamar via cl.make_async."""
    collection = get_collection()
    total = collection.count()
    if total == 0:
        return []

    query_vec = next(iter(get_embedder().query_embed(query))).tolist()
    res = collection.query(
        query_embeddings=[query_vec],
        n_results=min(n_results, total),
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[Chunk] = []
    for doc_id, text, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        meta = meta or {}
        chunks.append(Chunk(
            id=meta.get("doc_id", doc_id),
            text=text or "",
            score=round(max(0.0, 1.0 - float(dist)), 4),  # distância cosine → similaridade
            source=meta.get("source", ""),
            page=meta.get("page"),
            section=meta.get("section", ""),
        ))
    return chunks


def _select_within_budget(chunks: list[Chunk], max_chars: int) -> list[Chunk]:
    """Mantém quantos chunks INTEIROS couberem no orçamento (sempre ao menos 1)."""
    selected: list[Chunk] = []
    total = 0
    for c in chunks:
        block_len = len(c.text) + 80  # overhead aproximado do rótulo da fonte
        if selected and total + block_len > max_chars:
            break
        selected.append(c)
        total += block_len
    return selected


def _chunk_label(chunk: Chunk) -> str:
    label = chunk.source
    if chunk.source and chunk.page is not None:
        label += f", pág. {chunk.page}"
    return label


def _format_context(chunks: list[Chunk]) -> str:
    if not chunks:
        return "(Nenhum contexto relevante encontrado.)"
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        label = f"[{i}] {_chunk_label(chunk)}".rstrip()
        if chunk.section:
            label += f" — {chunk.section}"
        parts.append(f"{label}\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


def _build_prompt(query: str, chunks: list[Chunk], history: list[dict]) -> str:
    history_block = ""
    if history:
        recent = history[-MAX_HISTORY_TURNS:]
        turns = [
            f"{'Usuário' if m['role'] == 'user' else 'Assistente'}: {m['content']}"
            for m in recent
        ]
        history_block = "HISTÓRICO DA CONVERSA:\n" + "\n".join(turns) + "\n\n"
    return _PROMPT_TEMPLATE.format(
        context=_format_context(chunks), history=history_block, question=query
    )


# ── Helpers de renderização ────────────────────────────────────────────────────

def _format_sources(chunks: list[Chunk]) -> str:
    lines = []
    for chunk in chunks:
        label = f"`{chunk.id}`"
        if chunk.source:
            label += f" — **{_chunk_label(chunk)}**"
        if chunk.section:
            label += f" · *{chunk.section}*"
        label += f" (score {chunk.score:.3f})"
        snippet = chunk.text[:300] + ("…" if len(chunk.text) > 300 else "")
        lines.append(f"{label}\n\n> {snippet}")
    return "\n\n---\n\n".join(lines)


# ── Ciclo de vida da sessão Chainlit ──────────────────────────────────────────

@cl.on_chat_start
async def start() -> None:
    try:
        collection = await cl.make_async(get_collection)()
        await cl.make_async(get_embedder)()  # aquece o modelo de embedding
        count = await cl.make_async(collection.count)()
    except Exception as e:  # noqa: BLE001
        await cl.Message(
            content=f"❌ Não foi possível conectar ao ChromaDB em "
                    f"`{CHROMA_HOST}:{CHROMA_PORT}` (collection `{COLLECTION_NAME}`): {e}"
        ).send()
        return

    cl.user_session.set("history", [])  # list[dict]: {"role", "content"}

    settings = await cl.ChatSettings(
        [
            Select(
                id="model",
                label="Modelo LLM (Groq)",
                values=CHAT_MODELS,
                initial_index=CHAT_MODELS.index(DEFAULT_MODEL) if DEFAULT_MODEL in CHAT_MODELS else 0,
            ),
            Slider(id="n_results", label="Context chunks (k)", initial=DEFAULT_N_RESULTS, min=1, max=30, step=1),
            Slider(id="temperature", label="Temperature", initial=DEFAULT_TEMPERATURE, min=0.0, max=1.0, step=0.05),
            Slider(id="max_tokens", label="Max tokens da resposta", initial=DEFAULT_MAX_TOKENS, min=128, max=4096, step=128),
            Switch(id="show_sources", label="Show source citations", initial=True),
        ]
    ).send()
    cl.user_session.set("settings", settings)

    lines = [
        "## 🤖 RAG Chatbot",
        "Olá, como posso te ajudar hoje?",
        "",
        f"Collection: **{COLLECTION_NAME}** ({count} chunks) — ChromaDB `{CHROMA_HOST}:{CHROMA_PORT}`",
        "",
        await _groq_status_markdown(),
    ]
    if count == 0:
        lines += ["", "⚠️ A collection está vazia — ingira documentos primeiro."]

    await cl.Message(content="\n".join(lines), author="System").send()


@cl.on_settings_update
async def update_settings(settings: dict) -> None:
    cl.user_session.set("settings", settings)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    settings = cl.user_session.get("settings") or {}
    model        = settings.get("model", DEFAULT_MODEL)
    n_results    = int(settings.get("n_results", DEFAULT_N_RESULTS))
    temperature  = float(settings.get("temperature", DEFAULT_TEMPERATURE))
    max_tokens   = int(settings.get("max_tokens", DEFAULT_MAX_TOKENS))
    show_sources = bool(settings.get("show_sources", True))
    history: list[dict] = cl.user_session.get("history", [])

    reply = cl.Message(content="")
    await reply.send()

    try:
        all_chunks = await cl.make_async(retrieve)(message.content, n_results)
    except Exception as e:  # noqa: BLE001
        reply.content = f"Erro ao consultar o ChromaDB: {e}"
        await reply.update()
        return

    sources = _select_within_budget(all_chunks, MAX_CONTEXT_CHARS)
    prompt = _build_prompt(message.content, sources, history)

    full_answer = ""
    try:
        stream = await get_groq().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and (token := chunk.choices[0].delta.content):
                full_answer += token
                await reply.stream_token(token)
        if not full_answer:
            reply.content = f"O modelo '{model}' não retornou nenhum conteúdo."
    except Exception as e:  # noqa: BLE001
        note = f"Erro: {_friendly_error(e)}"
        reply.content = (full_answer + "\n\n" if full_answer else "") + note

    if show_sources and sources:
        reply.elements = [cl.Text(name="📚 Fontes", content=_format_sources(sources), display="inline")]

    await reply.update()

    if full_answer:
        history.append({"role": "user", "content": message.content})
        history.append({"role": "assistant", "content": full_answer})
        cl.user_session.set("history", history)
