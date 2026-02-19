import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from typing import List, Dict

from google import genai
from google.genai import types
from langchain_core.embeddings import Embeddings

load_dotenv()


# ==========================================================
# GOOGLE CLIENT
# ==========================================================
class GoogleClient:
    def __init__(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.embedding_model = os.getenv("GOOGLE_EMBEDDING_MODEL")
        self.chat_model = os.getenv("GOOGLE_CHAT_MODEL")

        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY não encontrada no .env")

        self.client = genai.Client(api_key=self.google_api_key)

    # ------------------------------------------------------
    # EMBEDDINGS
    # ------------------------------------------------------
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = []

        for text in texts:
            try:
                response = self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=text
                )

                embeddings.append(response.embeddings[0].values)

            except Exception as e:
                print(f"Erro ao gerar embedding para o texto: {text[:50]}... Erro: {e}")
                embeddings.append([0.0] * 768)

        return embeddings

    # ------------------------------------------------------
    # CHAT
    # ------------------------------------------------------
    def chat_completions(self, messages: List[dict], temperature: float = 0.1) -> str:
        try:
            prompt = "\n".join(m["content"] for m in messages)

            response = self.client.models.generate_content(
                model=self.chat_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=1000,
                )
            )

            return response.text or ""

        except Exception as e:
            print(f"Erro ao gerar resposta do chat: {e}")
            return "Desculpe, ocorreu um erro ao processar sua solicitação."


# ==========================================================
# LANGCHAIN EMBEDDINGS WRAPPER
# ==========================================================
class GoogleEmbeddings(Embeddings):
    def __init__(self):
        self.client = GoogleClient()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        batch_size = 10
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.client.get_embeddings(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.client.get_embeddings([text])[0]


def get_google_client() -> GoogleClient:
    return GoogleClient()


# ==========================================================
# PROMPT TEMPLATE (CORRIGIDO - REMOVIDO DUPLICADO)
# ==========================================================
PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


# ==========================================================
# RAG SEARCH
# ==========================================================
class RAGSearch:
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL")
        self.collection_name = os.getenv("PG_VECTOR_COLLECTION_NAME")

        if not self.database_url:
            raise ValueError("DATABASE_URL não definida no .env")

        if not self.collection_name:
            raise ValueError("PG_VECTOR_COLLECTION_NAME não definida no .env")

        self.embeddings = GoogleEmbeddings()
        self.google_client = get_google_client()
        self._initialize_vectorstore()

    def _initialize_vectorstore(self):
        self.vectorstore = PGVector(
            embeddings=self.embeddings,
            collection_name=self.collection_name,
            connection=self.database_url,
            use_jsonb=True,
        )

    # ------------------------------------------------------
    # BUSCA
    # ------------------------------------------------------
    def search_documents(self, query: str, k: int = 5) -> List[Dict]:
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)

            formatted = []
            for document, score in results:
                formatted.append({
                    "content": document.page_content,
                    "metadata": document.metadata,
                    "score": score
                })

            return formatted

        except Exception as e:
            print(f"Erro ao buscar documentos: {e}")
            return []

    # ------------------------------------------------------
    # GERA RESPOSTA
    # ------------------------------------------------------
    def generate_answer(self, query: str) -> str:
        documents = self.search_documents(query, k=5)

        if not documents:
            return "Não tenho informações necessárias para responder sua pergunta."

        context = "\n\n".join(doc["content"] for doc in documents)

        full_prompt = PROMPT_TEMPLATE.format(
            contexto=context,
            pergunta=query
        )

        messages = [
            {"role": "user", "content": full_prompt}
        ]

        response = self.google_client.chat_completions(
            messages,
            temperature=0.1
        )

        return response.strip()