from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini",
    temperature=0,
    max_tokens=200
)

prompt_Template = PromptTemplate.from_template("What is the use of local {env}")

prompt = prompt_Template.invoke({"env":"machine"})

response =  llm.invoke(prompt)

print(response.content)