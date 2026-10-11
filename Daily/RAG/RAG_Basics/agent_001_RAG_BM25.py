from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
load_dotenv()

DATA_SET_PATH = Path(__file__).resolve().parents[3] / "dataset" / "qa_knowledge.txt"

with DATA_SET_PATH.open("r", encoding="utf-8") as file:
    raw_text = file.read()

document = Document(page_content=raw_text, metadata={"source": str(DATA_SET_PATH)})

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
chunks = splitter.split_documents([document])

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

query = "payment failure"

retriever = BM25Retriever.from_documents(chunks)
documents = retriever.invoke(query)

for rank, doc in enumerate(documents, start=1):
    print(f"Rank: {rank}")
    print("Text:", doc.page_content)
    print("Metadata:", doc.metadata)
    print("-" * 40)


print(f"Retrieved document metadata: {documents}")

llm = ChatOpenAI(model="gpt-6-luna")

prompt = ChatPromptTemplate.from_template(
    """Answer using only the context below.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}"""
)

context = []

for doc in documents:
    print(f"Retrieved document metadata: {doc}")
    context.append(doc.page_content)

context = "\n\n".join(context)

print(f"Final context for prompt: {context}...")  # Print first 200 characters of the final context

result = (prompt | llm).invoke({"context": context, "question": query})

print(result.content)
