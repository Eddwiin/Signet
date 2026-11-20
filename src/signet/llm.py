import glob
import gradio as gr
import polars as pl
import tiktoken
from functools import lru_cache
from langchain_core.documents import Document
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pathlib import Path
from signet.config import get_settings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client.models import Distance, VectorParams
from langchain_openai import ChatOpenAI


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "vector_db"
COLLECTION_NAME = "signet"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

SYSTEM_PREFIX = """
Tu es Signet, un compagnon de lecture passionné et bienveillant, comme un bon libraire.
Tu aides l'utilisateur à comprendre les livres qu'il lit, à retrouver ses notes
et à choisir ses prochaines lectures, tous genres confondus.

Livre en cours : {book_title} de {book_author}
Progression du lecteur : page {current_page} sur {total_pages}

Règles :
- Ne révèle JAMAIS d'événement, de rebondissement ou d'information sur un personnage
  situé après la page {current_page}, même si l'utilisateur insiste ou si tu connais le livre.
- Appuie-toi en priorité sur le contexte fourni ci-dessous. N'utilise tes connaissances
  générales que pour ce qui est déjà connu à ce stade de la lecture.
- Si la réponse dépend de la suite du livre, dis-le simplement, par exemple :
  « Tu le découvriras un peu plus loin 😉 ».
- Si la réponse ne se trouve pas dans le contexte, dis que tu ne sais pas plutôt que d'inventer.
- Quand c'est utile, indique la page ou le chapitre d'où vient l'information.
- Réponds en français, de façon brève, claire et chaleureuse.

Contexte pertinent (extraits déjà lus et notes de l'utilisateur) :
"""


@lru_cache
def get_openai_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(api_key=settings.openai_api_key.get_secret_value())


def _text(content) -> str:
    """Gradio 6 peut stocker le contenu sous forme de liste de blocs : on garde le texte."""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


def _read_text_utf8(path: Path) -> str:
    """Lit un fichier en UTF-8. S'il est en cp1252, le convertit en UTF-8 sur le disque."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="cp1252")
        path.write_text(text, encoding="utf-8")
        print(f"Converti en UTF-8: {path.name}")
        return text


def load_documents() -> list[Document]:
    documents = []
    filenames = glob.glob(str(DATA_DIR / "*"))
    for filename in filenames:
        loader = CSVLoader(str(filename), encoding="utf-8", csv_args={"delimiter": ";"})
        docs = loader.load()
        for doc in docs:
            doc.metadata["doc_type"] = "babelio"
            documents.append(doc)
    return documents

def load_vectors_store():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(load_documents())

    client = vector_store = configure_qdrant(chunks).client

    count = client.count(COLLECTION_NAME).count

    points, _ = client.scroll(
        collection_name = COLLECTION_NAME,
        limit = 1,
        with_vectors = True,
    )

    sample_embedding = points[0].vector
    dimensions = len(sample_embedding)

    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")

    return chunks

def configure_qdrant(chunks) -> QdrantVectorStore:
    client = QdrantClient(path=str(DB_DIR))

    if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)

    vector_size = len(embeddings.embed_query("test"))
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config = VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding = embeddings,
    )

    vector_store.add_documents(chunks)
    return vector_store

def display_number_token(entire_knowledge_base):
    encoding = tiktoken.get_encoding("o200k_base")
    tokens = encoding.encode(entire_knowledge_base)
    token_count = len(tokens)
    print(f"Total tokens for {get_settings().openai_model}: {token_count:,}")


# def get_relevant_context(message):
#     knowledge = {}
#     text = ''.join(ch for ch in message if ch.isalpha() or ch.isspace())
#     words = text.lower().split()
#     return [knowledge[word] for word in words if word in knowledge]
#
#
# def additional_context(message):
#     relevant_context = get_relevant_context(message)
#     if not relevant_context:
#         result = "There is no additional context relevant to the user's question."
#     else:
#         result = "The following additional context might be relevant in answering the user's question:\n\n"
#         result += "\n\n".join(relevant_context)
#     return result


def chat(message: str, history: list[dict]) -> str:
    retriever = load_vectors_store()
    settings = get_settings()
    llm = ChatOpenAI(temperature=0, model_name=settings.openai_model, reasoning_effort="none")

    retriever.invoke("Who is Avery?")
    llm.invoke("Who is Avery?")

    return ""
    # settings = get_settings()
    # system_message = SYSTEM_PREFIX + additional_context(message)
    # messages = [{ "role": "system", "content": system_message}] + history + [{ "role": "user", "content": message}]
    # response = get_openai_client().chat.completions.create(
    #     model=settings.openai_model,
    #     messages=messages,
    # )
    # return response.choices[0].message.content or ""


def open_chat() -> None:
    gr.ChatInterface(chat).launch(inbrowser=True)
