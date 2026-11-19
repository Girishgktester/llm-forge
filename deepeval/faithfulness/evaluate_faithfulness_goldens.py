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
from deepeval.evaluate import DisplayConfig
from deepeval.dataset import EvaluationDataset, Golden
import json

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLDENS_PATH = PROJECT_ROOT / "goldens" / "faithfulness_goldens.json"
DATA_SET_PATH = PROJECT_ROOT / "dataset" / "faithfulness_test_data.txt"

with DATA_SET_PATH.open("r", encoding="utf-8") as file:
    raw_text = file.read()

with GOLDENS_PATH.open("r", encoding="utf-8") as file:
    golden_data = json.load(file)

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

retriever = vstore.as_retriever(search_kwargs={"k": 2})
llm = ChatOpenAI(model="gpt-6-luna")

prompt = ChatPromptTemplate.from_template(
    "Answer the question using only the context below. "
    "Do not assume a standard severity scale or add information. "
    "If the answer is missing from the context, say you don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)
goldens = [Golden(**item) for item in golden_data]
dataset = EvaluationDataset(goldens=goldens)

test_cases = []
for golden in dataset.goldens:
    query = golden.input
    documents = retriever.invoke(query)
    retrieved_context = [doc.page_content for doc in documents]

    result = (prompt | llm).invoke({
        "context": "\n\n".join(retrieved_context),
        "question": query,
    })

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=result.content,
        expected_output=golden.expected_output,
        retrieval_context=retrieved_context,
    )
    test_cases.append(test_case)

metric = FaithfulnessMetric(threshold=0.8, model="gpt-6-luna", include_reason=True, strict_mode=True)
evaluate(
    test_cases=test_cases,
    metrics=[metric],
    display_config=DisplayConfig(
        print_results=True,
        results_folder="./deepeval_results",
        file_type="html",
        file_output_dir="./deepeval_results",
    ),
)
