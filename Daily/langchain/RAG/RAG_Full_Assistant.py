from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAG_Full_Assistant:
    """Single-class end-to-end RAG assistant."""

    def __init__(self, pdf_path: str, db_dir: str | None = None):
        self.pdf_path = Path(pdf_path)
        self.db_dir = (Path(db_dir) if db_dir else self.pdf_path.parent / "rag_db").resolve()
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.vectorstore: Chroma | None = None

    # ── Load + Chunk ──────────────────────────────────────
    def load_pdf(self) -> list[str]:
        docs = PyPDFLoader(str(self.pdf_path)).load()
        chunks = self.splitter.split_documents(docs)
        print(f"Chunked → {len(chunks)} pieces\n")
        self.chunks = [c.page_content for c in chunks]
        return self.chunks

    # ── Ingest into Chroma ────────────────────────────────
    def ingest(self) -> Chroma:
        if not hasattr(self, "chunks"):
            self.load_pdf()

        self.vectorstore = Chroma.from_texts(
            texts=self.chunks,
            embedding=self.embeddings,
            persist_directory=str(self.db_dir),
            collection_name="assistant_db",
        )
        print(f"Stored → {len(self.chunks)} vectors at {self.db_dir}\n")
        return self.vectorstore

    # ── Query ─────────────────────────────────────────────
    def query(self, question: str, k: int = 3) -> dict:
        if not self.vectorstore:
            self.ingest()

        retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        docs = retriever.invoke(question)

        results = []
        for i, doc in enumerate(docs, 1):
            results.append({
                "chunk_no": i,
                "content": doc.page_content[:300],
                "source": doc.metadata.get("source", "unknown"),
            })

        return {"question": question, "answer_candidates": results}

    # ── One-liner run ─────────────────────────────────────
    def run(self, question: str) -> dict:
        self.load_pdf()
        self.ingest()
        return self.query(question)


# ── Usage ──────────────────────────────────────────────────
if __name__ == "__main__":
    rag = RAG_Full_Assistant("sample.pdf")
    answer = rag.run("What is this document about?")
    for r in answer["answer_candidates"]:
        print(f"\n--- Chunk {r['chunk_no']} ---")
        print(r["content"])
