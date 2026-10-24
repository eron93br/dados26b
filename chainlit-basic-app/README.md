# chainlit-basic-app

RAG chatbot em Chainlit: LLM via Groq + ChromaDB local (collection `my_collection`).

```bash
cp .env.example .env          # preencha GROQ_API_KEY
.venv/bin/chainlit run app.py -w --port 8501
```

## Python 3.14

O CLI do Chainlit aplica `nest_asyncio`, que quebra o asyncio do Python 3.14
(`Timeout should be used inside a task` / `NoEventLoopError`). O `.venv` deste
projeto tem `site-packages/zz_disable_nest_asyncio.pth`, que desativa isso.
Se recriar o venv, recrie esse arquivo ou rode via `run.py`:

```bash
.venv/bin/python run.py run app.py -w --port 8501
```
