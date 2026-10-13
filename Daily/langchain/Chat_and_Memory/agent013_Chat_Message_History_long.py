from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory

from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=200,
)
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = template | llm

store = {}

def get_session_history(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection="sqlite:///chathistory.db",
    )
    
history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)
session_id = "Gk"
get_session_history(session_id).clear()

response1 = history.invoke( {"input": "I am building an AI testing project. I am using Python and LangChain."},
                           config={"configurable": {"session_id": session_id}})
response2 = history.invoke({"input": "Which language and framework am I using? for building AI testing project"}, 
                           config={"configurable": {"session_id": session_id}},)
print("response1:", response1.content)
print("response2:", response2.content)
