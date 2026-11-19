from deepeval import evaluate
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_SET_PATH = PROJECT_ROOT / "dataset" / "qa_knowledge.txt"
with DATA_SET_PATH.open("r", encoding="utf-8") as file:
    raw_text = file.read()

document = Document(page_content=raw_text, metadata={"source": str(DATA_SET_PATH)})

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
chunks = splitter.split_documents([document])

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

CHROMA_DB_PATH = str(
    Path(__file__).resolve().parents[2] / ".." / ".." / "chroma_db"
).replace("\\", "/")
vstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="qa_knowledge",
    persist_directory=CHROMA_DB_PATH,
)

retriver = vstore.as_retriever(search_kwargs={"k": 2})

query = "What are the password requirements for the application?"

llm = ChatOpenAI(model="gpt-6-luna")

prompt = ChatPromptTemplate.from_template(
    "Answer the question using only the context below. "
    "Do not assume a standard severity scale or add information. "
    "If the answer is missing from the context, say you don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)
documents = retriver.invoke(query)

retrieved_context = "\n\n".join(document.page_content for document in documents)
result = (prompt | llm).invoke({"context": retrieved_context, "question": query})

goldens = [
    Golden(
        input="What is the minimum password length?",
        expected_output="The password must contain at least 8 characters.",
         context=["The application password policy requires a minimum of 8 characters. Passwords must also contain one uppercase letter and one number."]
    )
]
dataset = EvaluationDataset(goldens=goldens)
golden = goldens[0]
query = golden.input

test_case = LLMTestCase(
    input=golden.input,
    actual_output=result.content,
    expected_output=golden.expected_output,
    retrieval_context=[doc.page_content for doc in documents],
)
metric = FaithfulnessMetric(threshold=0.8, model="gpt-6-luna", include_reason=True)
evaluate(test_cases=[test_case], metrics=[metric],display_config=evaluate.DisplayConfig(
        print_results=True, 
        results_folder="./deepeval_results", 
        file_type="html",
        file_output_dir="./deepeval_results"))
