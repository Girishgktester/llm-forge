import json
import sys
from pathlib import Path

from deepeval import evaluate
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate import DisplayConfig
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from rag_support.rag_retriever import RAGRetriever
GOLDENS_PATH = PROJECT_ROOT / "goldens" / "hallucination_goldens.json"
DATA_SET_PATH = PROJECT_ROOT / "dataset" / "hallucination_goldens.txt"

print("=" * 80)
print("Hallucination metric debug run")
print(f"Golden file: {GOLDENS_PATH}")
print(f"Knowledge file: {DATA_SET_PATH}")

with GOLDENS_PATH.open("r", encoding="utf-8") as file:
    golden_data = json.load(file)

goldens = [Golden(**item) for item in golden_data]
dataset = EvaluationDataset(goldens=goldens)
print(f"Loaded {len(dataset.goldens)} Golden test cases.")

retriever = RAGRetriever(DATA_SET_PATH).get_retriever()
print("RAG retriever is ready.")

llm = ChatOpenAI(model="gpt-6-luna")
prompt = ChatPromptTemplate.from_template(
    "Answer the question using only the context below. "
    "If the answer is missing from the context, say you don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)

test_cases = []
for case_number, golden in enumerate(dataset.goldens, start=1):
    print("\n" + "-" * 80)
    print(f"Test case {case_number}/{len(dataset.goldens)}")
    print(f"Question: {golden.input}")

    documents = retriever.invoke(golden.input)
    retrieved_context = [document.page_content for document in documents]
    print(f"Retrieved {len(documents)} document(s):")
    print(f"retrieved_context {retrieved_context} ")

    for document_number, document in enumerate(documents, start=1):
        print(f"\n  Document {document_number}")
        print(f"  Source: {document.metadata.get('source', 'unknown')}")
        print(f"  Content:\n{document.page_content}")

    answer = (prompt | llm).invoke(
        {
            "context": "\n\n".join(retrieved_context),
            "question": golden.input,
        }
    ).content
    print(f"\nGenerated answer: {answer}")

    test_cases.append(
        LLMTestCase(
            input=golden.input,
            actual_output=answer,
            context=retrieved_context,
        )
    )

metric = HallucinationMetric(
    threshold=0.8,
    model="gpt-6-luna",
    include_reason=True,
)
print("\nRunning Hallucination evaluation...")
print("The metric checks whether the answer contradicts the retrieved context.")
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
print("\nHallucination evaluation finished.")