# rag-app (Groq + ChromaDB)

App RAG com interface Streamlit:

- **LLM:** API do [Groq](https://console.groq.com) (nuvem, compatível com OpenAI)
- **Banco vetorial:** [ChromaDB](https://docs.trychroma.com) rodando como servidor local (HTTP)
- **Embeddings:** [FastEmbed](https://github.com/qdrant/fastembed), gerados localmente na CPU via ONNX.
  O Groq não oferece endpoint de embeddings, então essa etapa roda na sua máquina, sem Ollama e sem GPU.

```
PDF ─► PyPDFLoader ─► chunking por seções ─► FastEmbed (local) ─► ChromaDB
                                                                     │
Pergunta ─► FastEmbed (local) ─► busca no ChromaDB ─► prompt ─► Groq (LLM) ─► resposta
```

## 1. Pré-requisitos

- Python 3.11+
- Uma conta no Groq (gratuita) para gerar a API key
- Internet: para chamar o Groq e, só na primeira execução, baixar o modelo de embeddings (~220 MB)

## 2. Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Configurar a API key do Groq

1. Acesse https://console.groq.com/keys, faça login e clique em **Create API Key**.
2. Copie a chave (começa com `gsk_`). Ela só é exibida uma vez.
3. Crie o arquivo `.env` a partir do exemplo e cole a chave:

```bash
cp .env.example .env
```

```dotenv
GROQ_API_KEY=gsk_sua_chave_aqui
LLM_MODEL=openai/gpt-oss-20b
```

> O `.env` está no `.gitignore`. Nunca faça commit da sua chave.

Os modelos que aparecem no seletor da UI vêm de `GROQ_CHAT_MODELS` (separados por vírgula).
A lista atual de modelos fica em https://console.groq.com/docs/models. Se algum modelo configurado
não existir mais, o indicador **Groq Conectado** no sidebar mostra quais estão indisponíveis.

Para testar a chave pelo terminal:

```bash
python -c "from llm_client import invoke; print(invoke('Diga olá', 'openai/gpt-oss-20b'))"
```

## 4. Rodar o ChromaDB

O app conecta ao Chroma via HTTP (`CHROMA_HOST`/`CHROMA_PORT`, padrão `localhost:8000`).
Por isso o servidor precisa estar rodando **antes** de abrir o app. Escolha **uma** das opções abaixo.

### Opção A: CLI do próprio pacote `chromadb` (já vem no requirements)

Em um terminal separado, com o venv ativado:

```bash
chroma run --path ./chroma_data --host localhost --port 8000
```

- `--path` é a pasta onde os dados ficam persistidos. Os dados continuam lá depois de reiniciar.
- Deixe esse terminal aberto enquanto usa o app.

### Opção B: Docker

```bash
docker run -d --name chromadb \
  -p 8000:8000 \
  -v "$(pwd)/chroma_data:/data" \
  chromadb/chroma
```

Para parar/iniciar depois: `docker stop chromadb` / `docker start chromadb`.

### Verificar se o Chroma está no ar

```bash
curl http://localhost:8000/api/v2/heartbeat
```

Deve retornar algo como `{"nanosecond heartbeat": ...}`.

## 5. Rodar o app

Com o Chroma rodando e o `.env` configurado:

```bash
streamlit run app.py
```

Abra http://localhost:8501. Fluxo sugerido:

1. **Ingest PDFs**: envie seus PDFs. Na primeira vez o modelo de embeddings é baixado, o que leva alguns segundos.
2. **RAG Chatbot**: faça perguntas sobre os documentos. A resposta é gerada pelo Groq.
3. **Chat LLM**: conversa direta com o modelo, sem RAG.

Também dá para ingerir pela linha de comando:

```bash
python -m pipeline.pdf_ingestor arquivo1.pdf arquivo2.pdf
```

## 6. Migrando de uma coleção criada com Ollama

Coleções indexadas com o `bge-m3` do Ollama têm vetores de 1024 dimensões. O modelo padrão agora gera
384, então essas coleções **não são compatíveis** e o chatbot mostra o aviso *Embeddings incompatíveis*.
Você tem duas saídas:

- apagar a coleção (página **Delete Document** → *Apagar toda a coleção*) e re-ingerir os PDFs; ou
- usar uma coleção nova: `CHROMA_COLLECTION_NAME=minha_colecao_groq` no `.env`.

O mesmo vale sempre que você trocar o `EMBEDDING_MODEL`.

## 7. Variáveis de ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `GROQ_API_KEY` | (obrigatória) | Chave da API do Groq |
| `LLM_MODEL` | `openai/gpt-oss-20b` | Modelo padrão do chat |
| `GROQ_CHAT_MODELS` | `openai/gpt-oss-120b,openai/gpt-oss-20b,qwen/qwen3.8-27b` | Modelos no seletor da UI |
| `GROQ_TIMEOUT` | `120` | Timeout (s) das chamadas ao Groq |
| `CHROMA_HOST` / `CHROMA_PORT` | `localhost` / `8000` | Endereço do servidor Chroma |
| `CHROMA_COLLECTION_NAME` | `my_collection` | Coleção usada pelo app |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Modelo FastEmbed (multilíngue, 384 dims). Para mais qualidade: `intfloat/multilingual-e5-large` (1024 dims, ~2,2 GB) |
| `EMBEDDING_CACHE_DIR` | cache do FastEmbed | Pasta onde o modelo de embeddings é salvo |
| `RAG_N_RESULTS` | `8` | Chunks recuperados por pergunta |
| `RAG_MAX_CONTEXT_CHARS` | `6000` | Limite de caracteres de contexto no prompt |
| `INGEST_CHUNK_SIZE` / `INGEST_CHUNK_OVERLAP` | `1500` / `200` | Chunking na ingestão |

## Problemas comuns

- **"Groq Desconectado"**: `GROQ_API_KEY` ausente ou inválida no `.env`, ou sem internet. Reinicie o Streamlit depois de editar o `.env`.
- **"Limite de requisições/tokens do Groq atingido"**: o plano gratuito tem limite por minuto. Aguarde, ou reduza *Context chunks (k)* e *Max tokens* no sidebar.
- **"Could not connect to ChromaDB"**: o servidor Chroma não está rodando (passo 4) ou a porta no `.env` está diferente.
- **"Embeddings incompatíveis"**: veja a seção 6.
