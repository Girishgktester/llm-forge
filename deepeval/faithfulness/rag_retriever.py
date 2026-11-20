from pathlib import Path
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAGRetriever:
    def __init__(self, dataset_path: Path):
        with dataset_path.open("r", encoding="utf-8") as file:
            document = Document(
                page_content=file.read(),
                metadata={"source": str(dataset_path)},
            )

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=10,
        ).split_documents([document])

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=f"qa_{uuid4().hex}",
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": 2})
