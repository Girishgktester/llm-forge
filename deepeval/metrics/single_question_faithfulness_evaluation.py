import json
from math import sqrt
from pathlib import Path

from deepeval import evaluate
from deepeval.dataset import Golden
from deepeval.evaluate import DisplayConfig
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLDENS_PATH = PROJECT_ROOT / "goldens" / "faithfulness_goldens.json"
DATA_SET_PATH = PROJECT_ROOT / "dataset" / "faithfulness_test_data.txt"
GOLDEN_MATCH_THRESHOLD = 0.75


class SingleQuestionFaithfulnessTest:
    def __init__(self):
        load_dotenv()

        with GOLDENS_PATH.open("r", encoding="utf-8") as file:
            self.goldens = [Golden(**item) for item in json.load(file)]
        if not self.goldens:
            raise ValueError(f"No goldens found in {GOLDENS_PATH}")

        with DATA_SET_PATH.open("r", encoding="utf-8") as file:
            document = Document(
                page_content=file.read(),
                metadata={"source": str(DATA_SET_PATH)},
            )

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=10,
        ).split_documents([document])

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.golden_embeddings = self.embeddings.embed_documents(
            [golden.input for golden in self.goldens]
        )

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name="single_question_qa_knowledge",
        )
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        self.llm = ChatOpenAI(model="gpt-6-luna")
        self.prompt = ChatPromptTemplate.from_template(
            "Answer the question using only the context below. "
            "Do not assume a standard severity scale or add information. "
            "If the answer is missing from the context, say you don't know.\n\n"
            "Context:\n{context}\n\nQuestion: {question}"
        )

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        denominator = sqrt(sum(value * value for value in left)) * sqrt(
            sum(value * value for value in right)
        )
        if denominator == 0:
            return 0.0
        return sum(a * b for a, b in zip(left, right)) / denominator

    def find_matching_golden(self, question: str) -> Golden | None:
        question_embedding = self.embeddings.embed_query(question)
        scores = [
            self._cosine_similarity(question_embedding, golden_embedding)
            for golden_embedding in self.golden_embeddings
        ]
        best_index = max(range(len(scores)), key=scores.__getitem__)
        if scores[best_index] < GOLDEN_MATCH_THRESHOLD:
            return None
        return self.goldens[best_index]

    def create_test_case(self, question: str) -> LLMTestCase:
        matching_golden = self.find_matching_golden(question)
        documents = self.retriever.invoke(question)
        retrieved_context = [document.page_content for document in documents]
        answer = (self.prompt | self.llm).invoke(
            {
                "context": "\n\n".join(retrieved_context),
                "question": question,
            }
        ).content

        if matching_golden is not None:
            print(f"Matched Golden question: {matching_golden.input}")
            return LLMTestCase(
                input=question,
                actual_output=answer,
                expected_output=matching_golden.expected_output,
                retrieval_context=retrieved_context,
            )

        print("No sufficiently similar Golden found; evaluating faithfulness only.")
        return LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=retrieved_context,
        )

    def run(self, question: str) -> None:
        test_case = self.create_test_case(question)
        print(f"Answer: {test_case.actual_output}")

        metric = FaithfulnessMetric(
            threshold=0.8,
            model="gpt-6-luna",
            include_reason=True,
            strict_mode=True,
        )
        evaluate(
            test_cases=[test_case],
            metrics=[metric],
            display_config=DisplayConfig(
                print_results=True,
                results_folder="./deepeval_results",
                file_type="html",
                file_output_dir="./deepeval_results",
            ),
        )


def main() -> None:
    question = input("Enter your question: ").strip()
    if not question:
        raise ValueError("Question cannot be empty.")
    SingleQuestionFaithfulnessTest().run(question)


if __name__ == "__main__":
    main()
