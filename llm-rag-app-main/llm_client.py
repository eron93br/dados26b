"""Cliente LLM — chama a API do Groq (nuvem) para completar conversas."""

import os
from collections.abc import Iterator

import groq
from dotenv import load_dotenv

load_dotenv()

# Chave lida do .env — crie a sua em https://console.groq.com/keys
_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Tempo máximo (s) de cada chamada à API
_TIMEOUT = float(os.getenv("GROQ_TIMEOUT", "120"))

_client: groq.Groq | None = None


def _get_client() -> groq.Groq:
    """Retorna o cliente Groq (singleton)."""
    global _client
    if not _GROQ_API_KEY:
        raise Exception(
            "GROQ_API_KEY não configurada. Crie uma chave em https://console.groq.com/keys "
            "e adicione `GROQ_API_KEY=...` no arquivo .env."
        )
    if _client is None:
        _client = groq.Groq(api_key=_GROQ_API_KEY, timeout=_TIMEOUT)
    return _client


def _friendly_error(exc: Exception) -> Exception:
    """Converte exceções do SDK do Groq em mensagens claras para a UI."""
    if isinstance(exc, groq.AuthenticationError):
        return Exception("GROQ_API_KEY inválida. Verifique a chave no arquivo .env.")
    if isinstance(exc, groq.RateLimitError):
        return Exception("Limite de requisições/tokens do Groq atingido. Aguarde alguns segundos e tente novamente.")
    if isinstance(exc, groq.APITimeoutError):
        return Exception(f"Timeout na chamada ao LLM (>{_TIMEOUT:.0f}s).")
    if isinstance(exc, groq.APIConnectionError):
        return Exception("Não foi possível conectar à API do Groq. Verifique sua conexão com a internet.")
    if isinstance(exc, groq.APIStatusError):
        return Exception(f"Erro na API do Groq ({exc.status_code}): {exc.message}")
    return exc


def _wrap_text(text: str, max_length: int = 1000) -> str:
    lines = text.split()
    wrapped, current = [], ""
    for word in lines:
        if len(current) + len(word) + 1 <= max_length:
            current = f"{current} {word}" if current else word
        else:
            wrapped.append(current)
            current = word
    if current:
        wrapped.append(current)
    return "\n".join(wrapped)


def invoke(
    prompt: str,
    model: str,
    temperature: float = 0,
    max_tokens: int = 2048,
) -> str | None:
    """Envia uma mensagem ao modelo no Groq e retorna a resposta como string."""
    try:
        completion = _get_client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
    except groq.GroqError as exc:
        raise _friendly_error(exc) from exc

    content = completion.choices[0].message.content
    return _wrap_text(content) if content else None


def invoke_stream(
    prompt: str,
    model: str,
    temperature: float = 0,
    max_tokens: int = 2048,
) -> Iterator[str]:
    """Envia uma mensagem ao Groq e produz pedaços da resposta conforme chegam (streaming)."""
    got_content = False
    finish_reason = None

    try:
        stream = _get_client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            content = choice.delta.content or ""
            if content:
                got_content = True
                yield content
            if choice.finish_reason:
                finish_reason = choice.finish_reason
    except groq.GroqError as exc:
        raise _friendly_error(exc) from exc

    if not got_content:
        detail = f" (motivo: {finish_reason})" if finish_reason else ""
        raise Exception(f"O modelo '{model}' não retornou nenhum conteúdo{detail}.")


def ping() -> tuple[bool, list[str]]:
    """
    Verifica se a API do Groq está acessível com a chave configurada.

    Retorna (online, modelos_disponíveis). Usado para o indicador de status na UI.
    """
    try:
        models = _get_client().models.list()
    except Exception:
        return False, []
    return True, [m.id for m in models.data]


# ── Modelos de chat disponíveis no Groq ───────────────────────────────────────
# Lista completa e atualizada: https://console.groq.com/docs/models
# Pode ser sobrescrita no .env com GROQ_CHAT_MODELS (separados por vírgula).

CHAT_MODELS = [
    m.strip()
    for m in os.getenv(
        "GROQ_CHAT_MODELS",
        "openai/gpt-oss-120b,openai/gpt-oss-20b,qwen/qwen3.8-27b",
    ).split(",")
    if m.strip()
]
