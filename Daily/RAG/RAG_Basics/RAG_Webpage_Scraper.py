from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from langchain_core.documents import Document
import requests

load_dotenv()

DATA_SET_PATH = Path(__file__).resolve().parents[2] / "dataset" / "qa_knowledge.txt"

with DATA_SET_PATH.open("r", encoding="utf-8") as file:
    raw_text = file.read()

url = "https://playwright.dev/docs/test-assertions"

response = requests.get(
    url,
    timeout=15,
    headers={"User-Agent": "Mozilla/5.0"},
)

response.raise_for_status()

html_content = response.text

soup = BeautifulSoup(html_content, "html.parser")

# Remove elements we don't need
for element in soup(["script", "style", "nav", "footer", "header"]):
    element.decompose()

# Extract readable text
page_text = soup.get_text(separator=" ", strip=True)

documents = [
    Document(page_content=page_text, metadata={"source": url}),
    Document(page_content=raw_text, metadata={"source": str(DATA_SET_PATH)})
]
splitter = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap=100)

chunks = splitter.split_documents([documents])

embedings = OpenAIEmbeddings(model="text-embedding-3-small")

vstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedings,
            collection_name="webpage_knowledge",
            persist_directory="./web_chroma_db"
)

retriever = vstore.as_retriever(search_kwargs={"k": 2})

query = "how to verify assert equals"

documents = retriever.invoke(query)

context = []

for doc in documents:
    context.append(doc.page_content)

context = "\n\n".join(context)

llm = ChatOpenAI(model="gpt-6-luna")


prompt = ChatPromptTemplate.from_template(""" 
                 Answer only from the webpage mentioned 
                 if you dont know the answer say it loud
                 dont invent any answers    
                 dont diretly USE LLM for generating answers if you dont have the ground
                 
                 "Context" : {context} 
                 
                 "question" : {question}                     
                
                                          """)

result = (prompt | llm).invoke({"context" : context, "question": query})

print(result.content)