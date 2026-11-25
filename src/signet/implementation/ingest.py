import glob

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import Distance, VectorParams
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

BASE_DIR = Path(__file__).parent.parent
BABELIO_PATH = str(BASE_DIR / "data" / "*")
DB_DIR = BASE_DIR / "vector_db_evaluation"
COLLECTION_NAME = "signet_test_evaluation"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def fetch_documents() -> list[Document]:
    filenames = glob.glob(BABELIO_PATH)
    documents = []

    for filename in filenames:
        loader = CSVLoader(str(filename), encoding="utf-8", csv_args={"delimiter": ";"})
        docs = loader.load()

        for doc in docs:
            doc.metadata["doc_type"] = "Babelio"
            documents.append(doc)
    return documents

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_embeddings(chunks):
    client = QdrantClient(path=str(DB_DIR))

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    vector_size = len(embeddings.embed_query("test"))
    client.create_collection(
        collection_name = COLLECTION_NAME,
        vectors_config = VectorParams(size=vector_size, distance=Distance.COSINE)
    )

    vector_store = QdrantVectorStore(
        client = client,
        collection_name = COLLECTION_NAME,
        embedding = embeddings,
    )

    vector_store.add_documents(chunks)
    return vector_store

if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")


