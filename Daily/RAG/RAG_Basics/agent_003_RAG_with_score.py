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
query = "Login is failing with error code 500. What could be the reason?"

# results = vstore.similarity_search_with_score(
#     query,
#     k=3
# )




retriever = vstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 2,
        "fetch_k": 5,
        "lambda_mult": 0.5
    }
)

documents = retriever.invoke(query)

for document in documents:
    print(document.page_content)