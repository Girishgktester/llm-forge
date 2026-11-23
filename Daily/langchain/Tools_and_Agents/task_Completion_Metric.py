import os
from dotenv import load_dotenv
load_dotenv()
from deepeval.metrics import TaskCompletionMetric
from deepeval.tracing import observe
from deepeval.dataset import Golden, EvaluationDataset
from langchain_openai import ChatOpenAI

@observe()
def test_case_agent(requirement: str) -> str:

    @observe()
    def generate_testcases(requirement: str) -> str:
        """Generate automated tests from the given test cases."""
        try:
               model = ChatOpenAI(model="gpt-6-luna")
               response = model.invoke(
                   [
                       {
                           "role": "system",
                           "content": f"""Act as a senior QA.
       
                           Generate testcases based on the user requirement provided. 
                           Each test case should include:
                           - Test Case ID
                           - Scenario
                           - Actual Result
                           - Expected Result
                           - Precondition
                           - Test Data
                           {requirement} """,
                       }
                   ]
               )
               return response.content
        except Exception as e:
               return f"ERROR: Failed to generate test code: {str(e)}"
    return generate_testcases(requirement)
    

dataset = EvaluationDataset(
    goldens=[
        Golden(
            input="Generate 2 negative edage case which should be very rare test cases for a login page"
        )
    ]
)

# Metric
metric = TaskCompletionMetric(
    threshold=0.7,
    model="gpt-6-luna",
    include_reason=True,
    verbose_mode=True
)

# Execute evaluation
for golden in dataset.evals_iterator(metrics=[metric]):
    test_case_agent(golden.input)


