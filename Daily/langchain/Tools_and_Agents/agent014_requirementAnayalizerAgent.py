from langchain.agents import create_agent
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel
from typing import List

class RequirementAnalysis(BaseModel):
    main_features: List[str]
    acceptance_criteria: List[str]
    missing_information: List[str]
    testing_risks: List[str]

user_input = input("Enter yor reuirement...")
prompt = PromptTemplate.from_template("""
            Based on the requirement given by the user:

            {user_input}

            Analyze the requirement and return the response in JSON format.
            Do not invent any requirement that is not provided.
            """)
finalprompt = prompt.invoke({"user_input": user_input})

agent = create_agent(model="gpt-4o-mini", 
                     system_prompt="act as a Senior QA Requirement Analyst ")

response = agent.invoke({
    "messages": [
        {"role": "user", "content": finalprompt.text}
    ]
})

final_message = response["messages"][-1]
print(final_message)
print(final_message.content)