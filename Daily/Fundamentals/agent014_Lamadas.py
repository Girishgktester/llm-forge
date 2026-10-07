from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama

small_model = ChatOllama(model="qwen3:8b", temperature=0)
large_model = ChatOllama(model="qwen3:14b", temperature=0)


def choose_model(question):
    if len(question) < 50:
        return small_model.invoke(question)
    else:
        return large_model.invoke(question)


chain = RunnableLambda(choose_model)

questions = [
    "What is Playwright?",
    "Explain how Playwright works with browser automation and why it is useful for end-to-end testing."
]

for question in questions:
    response = chain.invoke(question)

    print("\nQuestion:", question)
    print("Answer:", response.content)