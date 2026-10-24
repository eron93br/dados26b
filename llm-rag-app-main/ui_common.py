"""Componentes de UI compartilhados entre as páginas Streamlit."""

from __future__ import annotations

import streamlit as st

from llm_client import CHAT_MODELS, ping

_STATUS_CACHE_TTL = 120  # segundos


@st.cache_data(ttl=_STATUS_CACHE_TTL, show_spinner=False)
def _check_groq_status() -> dict:
    """Verifica se a API do Groq está acessível e se os modelos configurados estão disponíveis."""
    online, available = ping()
    if not online:
        return {"online": False, "detail": "Verifique a `GROQ_API_KEY` no .env e sua conexão com a internet."}

    missing = [m for m in CHAT_MODELS if m not in available]
    if missing:
        return {"online": True, "detail": f"Indisponíveis no Groq: {', '.join(missing)}"}

    return {"online": True, "detail": ""}


def render_llm_status() -> None:
    """Renderiza no sidebar uma caixinha centralizada verde/vermelha com o status do Groq (cacheado por 120s)."""
    status = _check_groq_status()

    if status["online"]:
        bg, fg, label = "#84f584", "#0b3d0b", "Groq Conectado"
    else:
        bg, fg, label = "#f58484", "#5c0b0b", "Groq Desconectado"

    st.sidebar.markdown(
        f"""
        <div style="
            background-color:{bg};
            color:{fg};
            text-align:center;
            padding:10px 8px;
            border-radius:10px;
            font-weight:600;
            margin-bottom:10px;
        ">
            {label}
        </div>
        """,
        unsafe_allow_html=True,
    )
    if status["detail"]:
        st.sidebar.caption(status["detail"])
