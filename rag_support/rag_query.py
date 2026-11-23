from collections.abc import Mapping, Sequence
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAGQuery:
    def __init__(
        self,
        url: str = "https://www.descope.com/learn/post/mcp",
        model_name: str = "qwen2.5:latest",
        embedding_model: str = "nomic-embed-text:latest",
        base_url: str = "http://localhost:11434",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        num_results: int = 4,
    ) -> None:
        documents = WebBaseLoader(url).load()
        if not documents:
            raise ValueError(f"No documents were loaded from {url!r}.")

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ).split_documents(documents)

        embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=base_url,
        )
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=f"mcp_rag_{uuid4().hex}",
        )
        self.retriever = vector_store.as_retriever(
            search_kwargs={"k": num_results}
        )
        self.llm = ChatOllama(
            base_url=base_url,
            model=model_name,
            temperature=0.5,
            num_predict=250,
        )
        self.model_name = model_name
        self.base_url = base_url
        self.prompt = ChatPromptTemplate.from_template(
            "Answer the question based only on the following context. "
            "Give a summary, not the full detail.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}"
        )
        self.answer_chain = self.prompt | self.llm | StrOutputParser()

    @staticmethod
    def format_documents(documents: Sequence[Document]) -> str:
        return "\n\n".join(document.page_content for document in documents)

    def query_with_context(self, question: str) -> tuple[str, str]:
        documents = self.retriever.invoke(question)
        context = self.format_documents(documents)
        answer = self.answer_chain.invoke(
            {"context": context, "question": question}
        )
        return context, answer

    def query(self, question: str) -> str:
        _, answer = self.query_with_context(question)
        return answer

    def evaluate(
        self,
        test_data: Sequence[Mapping[str, str]],
    ) -> object:
        from deepeval import evaluate as deepeval_evaluate
        from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
        from deepeval.models import OllamaModel
        from deepeval.test_case import LLMTestCase

        test_cases = []
        for sample in test_data:
            context, answer = self.query_with_context(sample["input"])
            test_cases.append(
                LLMTestCase(
                    input=sample["input"],
                    actual_output=answer,
                    expected_output=sample["expected_output"],
                    retrieval_context=[context],
                )
            )

        judge_model = OllamaModel(
            model=self.model_name,
            base_url=self.base_url,
        )
        return deepeval_evaluate(
            test_cases=test_cases,
            metrics=[
                AnswerRelevancyMetric(model=judge_model),
                FaithfulnessMetric(model=judge_model),
            ],
        )


class BasicAgent:
    def __init__(
        self,
        model_name: str = "gpt-6-luna",
        base_url: str = "https://api.commandcode.ai/provider/v1",
    ) -> None:
        import os

        from dotenv import load_dotenv
        from langchain.agents import create_agent
        from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
        from langchain_openai import ChatOpenAI

        load_dotenv()

        @tool
        def add(a: float, b: float) -> float:
            """Add two numbers."""
            return a + b

        @tool
        def subtract(a: float, b: float) -> float:
            """Subtract the second number from the first."""
            return a - b

        search_wrapper = DuckDuckGoSearchAPIWrapper()

        @tool
        def duckduckgo_search(query: str) -> str:
            """Search the web with DuckDuckGo."""
            return search_wrapper.run(query)

        self.agent = create_agent(
            model=ChatOpenAI(
                model=model_name,
                base_url=base_url,
                api_key=os.getenv("COMMAND_CODE_API_KEY"),
            ),
            tools=[add, subtract, duckduckgo_search],
            system_prompt="Use the available tools when needed. Be concise.",
        )

    def query(self, question: str) -> str:
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": question}]}
        )
        return result["messages"][-1].content


if __name__ == "__main__":
    agent = BasicAgent()
    print(agent.query("What is 12 minus 4?"))
