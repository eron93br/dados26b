"""
Inicializador do Chainlit compatível com Python 3.14.

O CLI do Chainlit chama nest_asyncio.apply() ao ser importado, e o nest_asyncio
é incompatível com o asyncio do Python 3.14: toda conexão WebSocket falha com
"RuntimeError: Timeout should be used inside a task". Aqui o apply() vira no-op
antes de carregar o CLI.

Uso (mesmos argumentos do comando `chainlit`):
    .venv/bin/python run.py run app.py -w --port 8501
"""

import nest_asyncio

nest_asyncio.apply = lambda *args, **kwargs: None

from chainlit.cli import cli  # noqa: E402

if __name__ == "__main__":
    cli()
