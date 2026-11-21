from dotenv import load_dotenv
import sys
import os
load_dotenv()

if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")

from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from deepeval.evaluate import evaluate
from deepeval.evaluate import DisplayConfig

llm_api = ChatOpenAI(model="gpt-6-luna")

def ask(q: str) -> str:
    prompt = ChatPromptTemplate.from_template("{question}")
    return (prompt | llm_api).invoke({"question": q}).content

test_case = LLMTestCase(
    input="What is the capital of France?",
    actual_output=ask("What is the capital of France?"),
)

metric = AnswerRelevancyMetric(threshold=0.7, model="gpt-6-luna", async_mode=False)

evaluate(
    test_cases=[test_case], 
    metrics=[metric], 
    display_config=DisplayConfig(
        print_results=True, 
        results_folder="./deepeval_results", 
        file_type="html",
        file_output_dir="./deepeval_results"))

