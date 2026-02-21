# Desafio MBA Engenharia de Software com IA - Full Cycle

Sistema RAG (Retrieval-Augmented Generation) que integra busca vetorial com LLM do Google para responder perguntas baseadas em documentos PDF.

## 🎯 Características

- **Ingestão de PDF**: Carrega e processa documentos PDF
- **Vetorização**: Converte texto em embeddings usando Google Generative AI
- **Busca Vetorial**: Recupera os 10 documentos mais relevantes via PGVector
- **Chat RAG**: Interface CLI para fazer perguntas e obter respostas contextualizadas
- **Banco de Dados**: PostgreSQL com extensão pgvector

## 📋 Pré-requisitos

- Python 3.10+
- Docker e Docker Compose
- Chave de API do Google Generative AI

## 🚀 Instalação

### 1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/mba-ia-desafio-ingestao-busca.git
cd mba-ia-desafio-ingestao-busca
```

### 2. Configure o ambiente
Crie um arquivo `.env` na raiz do projeto:

```env
# Google API
GOOGLE_API_KEY=sua-chave-api-aqui
GOOGLE_EMBEDDING_MODEL=models/text-embedding-004
GOOGLE_CHAT_MODEL=models/gemini-2.0-flash

# Banco de Dados
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rag
PG_VECTOR_COLLECTION_NAME=documents

# Caminho do PDF
PDF_PATH=document.pdf
```

### 3. Instale as dependências Python
```bash
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Inicie o PostgreSQL com Docker
```bash
docker-compose up -d
```

Aguarde até que o PostgreSQL esteja saudável (verificar logs com `docker-compose logs`).

## 📝 Uso

### Passo 1: Ingerir documento PDF

```bash
python src/ingest.py
```

Este script irá:
- 📄 Carregar o PDF definido em `PDF_PATH`
- ✂️ Dividir o texto em chunks de 1000 caracteres com sobreposição
- 🔢 Gerar embeddings para cada chunk
- 💾 Armazenar no PGVector para busca vetorial

**Saída esperada:**
```
📄 Carregando PDF: document.pdf
📚 Total de páginas carregadas: 50
✂️ Total de chunks gerados: 245
🚀 Iniciando ingestão no PGVector...
✅ Inseridos 5 de 245
...
🎉 Ingestão concluída com sucesso!
```

### Passo 2: Iniciar o Chat RAG

```bash
python src/chat.py
```

**Fluxo de execução:**
1. O script conecta ao banco vetorial
2. Aguarda sua pergunta
3. **Vetoriza** a pergunta usando embeddings
4. **Busca** os 10 documentos mais relevantes (k=10)
5. **Monta o prompt** combinando contexto + pergunta
6. **Chama a LLM** (Gemini) para gerar resposta
7. **Retorna** a resposta formatada

**Exemplo de interação:**
```
==================================================
Bem-vindo ao Chat com Busca Vetorial (RAG)
Digite 'sair' para encerrar

==================================================

Faça sua pergunta: Como funciona o sistema de autenticação?

==================================================
Pergunta: Como funciona o sistema de autenticação?
==================================================
Resposta: O sistema de autenticação utiliza tokens JWT...
==================================================
```

Para sair, digite: `sair`, `exit` ou `quit`

## 🗂️ Estrutura do Projeto

```
.
├── src/
│   ├── chat.py              # CLI para interação com usuário
│   ├── ingest.py            # Ingestão e vetorização de PDFs
│   ├── search.py            # Lógica de busca e RAG
│   └── list_models.py       # Utilitário para listar modelos disponíveis
├── docker-compose.yml       # Configuração PostgreSQL + pgvector
├── requirements.txt         # Dependências Python
├── .env                     # Variáveis de ambiente (não versionado)
├── .gitignore              # Arquivos ignorados
└── README.md               # Este arquivo
```

## 🔑 Variáveis de Ambiente

| Variável | Descrição | Padrão |
|----------|-----------|--------|
| `GOOGLE_API_KEY` | Chave API do Google Generative AI | - |
| `GOOGLE_EMBEDDING_MODEL` | Modelo de embeddings | `models/text-embedding-004` |
| `GOOGLE_CHAT_MODEL` | Modelo de chat/geração | `models/gemini-2.0-flash` |
| `DATABASE_URL` | Connection string PostgreSQL | - |
| `PG_VECTOR_COLLECTION_NAME` | Nome da coleção vetorial | `documents` |
| `PDF_PATH` | Caminho do arquivo PDF | `document.pdf` |

## 🛠️ Componentes Principais

### [`GoogleClient`](src/ingest.py) e [`GoogleClient`](src/search.py)
Gerencia integração com Google Generative AI:
- `get_embeddings()`: Vetoriza textos
- `chat_completions()`: Gera respostas com LLM

### [`GoogleEmbeddings`](src/ingest.py)
Wrapper LangChain para embeddings do Google com suporte a batch processing

### [`RAGSearch`](src/search.py)
Orquestra o pipeline RAG:
- `search_documents(query, k=10)`: Busca os k resultados mais relevantes
- `generate_answer(query, k=10)`: Gera resposta combinando contexto + LLM

## 📊 Fluxo de Dados

```
PDF → Carregamento → Chunking → Embeddings → PGVector
                                               ↓
                              Pergunta do Usuário
                                      ↓
                           Vetorização da Pergunta
                                      ↓
                        Busca Vetorial (k=10)
                                      ↓
                          Montagem do Prompt RAG
                                      ↓
                            Chamada à LLM (Gemini)
                                      ↓
                              Resposta ao Usuário
```

## 🧪 Utilitários

### Listar modelos disponíveis
```bash
python src/list_models.py
```

## 📦 Dependências Principais

- **LangChain**: Framework para aplicações LLM
- **pgvector**: Extensão PostgreSQL para busca vetorial
- **google-genai**: SDK oficial do Google Generative AI
- **pypdf**: Processamento de arquivos PDF

## ⚠️ Troubleshooting

### Erro: "GOOGLE_API_KEY não encontrada"
- Verifique se o arquivo `.env` existe e contém `GOOGLE_API_KEY`
- A chave deve ser gerada em [Google AI Studio](https://aistudio.google.com/)

### Erro: "DATABASE_URL não definida"
- Confirme que PostgreSQL está rodando: `docker-compose ps`
- Verifique a URL de conexão no `.env`

### Erro na ingestão: "Arquivo PDF não encontrado"
- Certifique-se que o arquivo PDF existe no caminho indicado em `PDF_PATH`
- Use caminho absoluto ou relativo correto

### Erro ao conectar ao PostgreSQL
```bash
# Reinicie os containers
docker-compose down
docker-compose up -d
```

## 👨‍💻 Autor

Gabriel Simon - MBA Engenharia de Software com IA - Full Cycle