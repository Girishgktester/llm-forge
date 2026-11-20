import json
from pathlib import Path

from deepeval import evaluate
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate import DisplayConfig
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

from rag_retriever import RAGRetriever

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLDENS_PATH = PROJECT_ROOT / "goldens" / "contextual_relevancy_goldens.json"
DATA_SET_PATH = PROJECT_ROOT / "dataset" / "playwright_course_knowledge.txt"

with GOLDENS_PATH.open("r", encoding="utf-8") as file:
    golden_data = json.load(file)

goldens = [Golden(**item) for item in golden_data]
dataset = EvaluationDataset(goldens=goldens)
retriever = RAGRetriever(DATA_SET_PATH).get_retriever()

test_cases = []
for golden in dataset.goldens:
    documents = retriever.invoke(golden.input)
    retrieved_context = [document.page_content for document in documents]
    test_cases.append(
        LLMTestCase(
            input=golden.input,
            retrieval_context=retrieved_context,
        )
    )

metric = ContextualRelevancyMetric(
    threshold=0.8,
    model="gpt-6-luna",
    include_reason=True
    )
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