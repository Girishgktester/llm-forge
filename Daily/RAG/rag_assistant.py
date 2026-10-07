from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

DATA_SET_PATH = Path(__file__).resolve().parents[2] / "dataset" / "qa_knowledge.txt"

with DATA_SET_PATH.open("r", encoding="utf-8") as file:
    raw_text = file.read()

document = Document(page_content=raw_text, metadata={"source": str(DATA_SET_PATH)})

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
chunk = splitter.split_documents([document])

print(len(chunk))

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vstore = Chroma.from_documents(documents=chunk,
                      embedding=embeddings,
                      collection_name="qa_knowledge",
                      persist_directory="./chroma_db",
)

data = vstore.get(include=["documents"])

print(data)
