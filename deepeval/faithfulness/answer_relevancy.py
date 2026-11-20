import json
from pathlib import Path

from deepeval import evaluate
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate import DisplayConfig
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from rag_retriever import RAGRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLDENS_PATH = PROJECT_ROOT / "goldens" / "answer_relevancy_goldens.json"
DATA_SET_PATH = PROJECT_ROOT / "dataset" / "answer_relevancy_knowledge_source.txt"

class AnswerRelevancyTest:
    def __init__(self):
        load_dotenv()

        with GOLDENS_PATH.open("r", encoding="utf-8") as file:
            golden_data = json.load(file)

        goldens = [Golden(**item) for item in golden_data["goldens"]]
        self.dataset = EvaluationDataset(goldens=goldens)
        self.retriever = RAGRetriever(DATA_SET_PATH).get_retriever()
        self.llm = ChatOpenAI(model="gpt-6-luna")
        self.prompt = ChatPromptTemplate.from_template(
            "Answer the question using only the context below. "
            "Do not assume a standard severity scale or add information. "
            "If the answer is missing from the context, say you don't know.\n\n"
            "Context:\n{context}\n\nQuestion: {question}"
        )

    def run(self) -> None:
        test_cases = []
        for golden in self.dataset.goldens:
            documents = self.retriever.invoke(golden.input)
            retrieved_context = [document.page_content for document in documents]
            answer = (self.prompt | self.llm).invoke(
                {
                    "context": "\n\n".join(retrieved_context),
                    "question": golden.input,
                }
            ).content
            print(f"Answer-------------------->: {answer}")

            test_cases.append(
                LLMTestCase(
                    input="What is the price of the Playwright with TypeScript course?",
                    actual_output="The Playwright with TypeScript course costs $250. It runs for three months. We also offer Selenium, Java, and API testing courses.",
                )
            )

        metric = AnswerRelevancyMetric(
            threshold=0.7,
            model="gpt-6-luna",
            async_mode=False,
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


def main() -> None:
    AnswerRelevancyTest().run()


if __name__ == "__main__":
    main()
