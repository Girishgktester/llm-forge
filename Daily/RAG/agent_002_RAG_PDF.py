from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset"
pdf_files = sorted(DATASET_DIR.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError(
        f"No PDF found in {DATASET_DIR}. Add one PDF to that folder and run again."
    )
if len(pdf_files) > 1:
    raise ValueError(
        f"Expected one PDF in {DATASET_DIR}, found: "
        + ", ".join(path.name for path in pdf_files)
    )

documents = PyPDFLoader(str(pdf_files[0])).load()
chunks = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
).split_documents(documents)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="rag_pdf",
)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

query = "What is selenium?"
retrieved_documents = retriever.invoke(query)
context = "\n\n".join(document.page_content for document in retrieved_documents)

prompt = ChatPromptTemplate.from_template(
    """Answer using only the context below.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}"""
)
llm = ChatOpenAI(model="gpt-6-luna")
answer = (prompt | llm).invoke({"context": context, "question": query})

print(answer.content)
