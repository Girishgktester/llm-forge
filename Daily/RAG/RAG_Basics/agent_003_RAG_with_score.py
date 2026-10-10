from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

DATA_SET_PATH = Path(__file__).resolve().parents[3] / "dataset" / "qa_knowledge.txt"

with DATA_SET_PATH.open("r", encoding="utf-8") as file:
    raw_text = file.read()

document = Document(page_content=raw_text, metadata={"source": str(DATA_SET_PATH)})

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
chunks = splitter.split_documents([document])


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vstore = Chroma.from_documents(documents=chunks,
                      embedding=embeddings,
                      collection_name="qa_knowledge",
                      persist_directory="./chroma_db",
)
query = "fkjshfds"

results = vstore.similarity_search_with_score(
    query,
    k=3
)
for document, score in results:
    print("Content:", document.page_content)
    print("Distance:", score)
    print("Metadata:", document.metadata)
    print("-" * 50)


