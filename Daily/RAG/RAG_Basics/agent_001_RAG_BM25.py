from dotenv import load_dotenv
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. Load the knowledge base
DATA_SET_PATH = (
    Path(__file__).resolve().parents[3]
    / "dataset"
    / "qa_knowledge.txt"
)

with DATA_SET_PATH.open("r", encoding="utf-8") as file:
    raw_text = file.read()

document = Document(
    page_content=raw_text,
    metadata={"source": str(DATA_SET_PATH)}
)

# 2. Split the document into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10
)

chunks = splitter.split_documents([document])

print(f"Total chunks: {len(chunks)}")

# 3. Build the BM25 keyword-search retriever
retriever = BM25Retriever.from_documents(chunks)

# Number of chunks to retrieve
retriever.k = 3

# 4. Search using keywords
query = "payment failure"

documents = retriever.invoke(query)

# 5. Display retrieved chunks
for rank, doc in enumerate(documents, start=1):
    print(f"\nRank: {rank}")
    print("Text:", doc.page_content)
    print("Metadata:", doc.metadata)
    print("-" * 40)

# 6. Prepare the retrieved context
context = "\n\n".join(
    doc.page_content for doc in documents
)

print("\nFinal context:")
print(context)

# 7. Pass retrieved context to the LLM
llm = ChatOpenAI(model="gpt-6-luna")

prompt = ChatPromptTemplate.from_template(
    """Answer using only the context below.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}"""
)

chain = prompt | llm

result = chain.invoke({
    "context": context,
    "question": query
})

print("\nFinal answer:")
print(result.content)
