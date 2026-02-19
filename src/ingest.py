import os
from dotenv import load_dotenv
from typing import List

from google import genai
from google.genai import types
from langchain_core.embeddings import Embeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "document.pdf")


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

        print("Embedding model:", self.embedding_model)
        print("Chat model:", self.chat_model)

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

                # SDK novo retorna:
                # response.embeddings[0].values
                embeddings.append(response.embeddings[0].values)

            except Exception as e:
                print(f"Erro ao gerar embedding para o texto: {text[:50]}... Erro: {e}")
                # fallback seguro
                embeddings.append([0.0] * 768)

        return embeddings

    # ------------------------------------------------------
    # CHAT (não usado na ingestão, mas mantido para reuso)
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

            return response.text

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
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.client.get_embeddings(batch_texts)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.client.get_embeddings([text])[0]


# ==========================================================
# INGESTÃO DO PDF
# ==========================================================
def ingest_pdf():
    DATABASE_URL = os.getenv("DATABASE_URL")
    COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "default_collection")

    if not DATABASE_URL:
        raise ValueError("DATABASE_URL não definida no .env")

    print(f"\n📄 Carregando PDF: {PDF_PATH}")

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print(f"📚 Total de páginas carregadas: {len(documents)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    print(f"✂️ Total de chunks gerados: {len(chunks)}")

    embeddings = GoogleEmbeddings()

    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )

    batch_size = 5

    print("\n🚀 Iniciando ingestão no PGVector...")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectorstore.add_documents(batch)
        print(f"✅ Inseridos {i + len(batch)} de {len(chunks)}")

    print("\n🎉 Ingestão concluída com sucesso!")


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    ingest_pdf()