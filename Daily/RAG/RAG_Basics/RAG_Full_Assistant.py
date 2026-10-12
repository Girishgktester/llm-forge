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

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
chunks = splitter.split_documents([document])

for i, chunk in enumerate(chunks):
    print(f"\n--- CHUNK {i} ---")
    print(chunk.page_content)
    print("Metadata:", chunk.metadata)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vstore = Chroma.from_documents(documents=chunks,
                      embedding=embeddings,
                      collection_name="qa_knowledge",
                      persist_directory="./chroma_db",
)

retriver = vstore.as_retriever(search_kwargs={"k":2})

query = "What severity is a payment failure, and what severity is a login failure for all users?"

llm = ChatOpenAI(model="gpt-6-luna")

prompt = ChatPromptTemplate.from_template(
    "Answer the question using only the context below. "
    "Do not assume a standard severity scale or add information. "
    "If the answer is missing from the context, say you don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)
documents = retriver.invoke(query)
for document in documents:
    print(f"\n--- DOCUMENT ---")
    print(document.page_content)
    print("Metadata:", document.metadata)
    
# context = []

# for document in documents:
#     context.append(document.page_content)

# context = "\n\n".join(context)
    
context = "\n\n".join(document.page_content for document in documents)
result = (prompt | llm).invoke({"context": context, "question": query})

print(result.content)


