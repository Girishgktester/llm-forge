import hashlib

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "./web_chroma_db"
COLLECTION_NAME = "webpage_knowledge"

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
)


def index_sources(results: list[dict]) -> list[dict]:
    vstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DB_PATH,
    )

    successful_sources = []

    for result in results:
        status = result["status"]
        source_id = result["source_id"]

        if status == "UNCHANGED":
            print(f"SKIPPED: {source_id}")
            continue

        document = Document(
            page_content=result["content"],
            metadata={"source": source_id},
        )

        chunks = splitter.split_documents([document])

        chunk_ids = [
            hashlib.sha256(
                f"{source_id}:{result['content_hash']}:{i}".encode("utf-8")
            ).hexdigest()
            for i in range(len(chunks))
        ]

        # For a changed source, remove its previous chunks.
        if status == "CHANGED":
            vstore.delete(where={"source": source_id})

        # Insert the new chunks and generate their embeddings.
        vstore.add_documents(
            documents=chunks,
            ids=chunk_ids,
        )

        successful_sources.append({
            "source_id": source_id,
            "content_hash": result["content_hash"],
        })

        print(f"INDEXED: {source_id} ({len(chunks)} chunks)")

    return successful_sources

def delete_sources(source_ids: list[str]) -> None:
    if not source_ids:
        return

    vstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DB_PATH,
    )

    for source_id in source_ids:
        vstore.delete(where={"source": source_id})
        print(f"DELETED: {source_id}")