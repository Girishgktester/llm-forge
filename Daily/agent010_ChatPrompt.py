from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini",
    temperature=0,
    max_tokens=200
)

promptTemplate = ChatPromptTemplate([
    ("system", "you are a senior QA"),
    ("user", "What is the advanatges of AI modles")
])

prompt = promptTemplate.invoke({"env": "machine"})

response = llm.invoke(prompt)
print(response.content)


