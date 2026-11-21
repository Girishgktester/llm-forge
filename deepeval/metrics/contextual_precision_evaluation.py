import json
import sys
from pathlib import Path

from deepeval import evaluate
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate import DisplayConfig
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from rag_support.rag_retriever import RAGRetriever
GOLDENS_PATH = PROJECT_ROOT / "goldens" / "contextual_relevancy_goldens.json"
DATA_SET_PATH = PROJECT_ROOT / "dataset" / "playwright_course_knowledge.txt"


with GOLDENS_PATH.open("r", encoding="utf-8") as file:
    golden_data = json.load(file)

goldens = [Golden(**item) for item in golden_data]
dataset = EvaluationDataset(goldens=goldens)


rag_retriever = RAGRetriever(DATA_SET_PATH)
retriever = rag_retriever.get_retriever()
vector_store = rag_retriever.get_vector_store()
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

    ranked_documents = vector_store.similarity_search_with_score(
        golden.input,
        k=3,
    )
    retrieved_context = [document.page_content for document, _score in ranked_documents]
    print(f"Retrieved {len(ranked_documents)} document(s):")
    for document_number, (document, score) in enumerate(ranked_documents, start=1):
        print(f"\n  Rank {document_number}")
        print(f"  Retrieval score: {score:.6f}")
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
            expected_output=golden.expected_output,
            retrieval_context=retrieved_context,
        )
    )

metric = ContextualPrecisionMetric(
    threshold=0.8,
    model="gpt-6-luna",
    include_reason=True,
)
print("\nRunning Contextual Relevancy evaluation...")
print("The metric evaluates each question against its retrieved context.")
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
print("\nContextual Relevancy evaluation finished.")