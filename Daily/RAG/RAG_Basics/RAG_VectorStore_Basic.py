from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

DATA_SET_PATH = Path(__file__).resolve().parents[2] / "dataset" / "qa_knowledge.txt"

with DATA_SET_PATH.open("r", encoding="utf-8") as file:
    raw_text = file.read()

document = Document(page_content=raw_text, metadata={"source": str(DATA_SET_PATH)})

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
chunks = splitter.split_documents([document])

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="qa_knowledge",
    persist_directory="./chroma_db",
)

retriever = vstore.as_retriever(search_kwargs={"k": 3})

query = "Login falire is what severity"
print(f"Querying retriever with: {retriever}")
documents = retriever.invoke(query)

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

# result = (prompt | llm).invoke({"context": context, "question": query})

# print(result.content)
