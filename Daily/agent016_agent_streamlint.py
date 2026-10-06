import os
import sys

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
import streamlit as st

from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.langsmith_config import configure_langsmith

load_dotenv()
configure_langsmith()


class ChatAssistant:
    def __init__(self):
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
        self.history = RunnableWithMessageHistory(
            chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def get_session_history(self, session_id: str):
        return SQLChatMessageHistory(
            session_id=session_id,
            connection="sqlite:///chathistory.db",
        )

    def clear_history(self, session_id: str):
        self.get_session_history(session_id).clear()

    def stream(self, session_id: str, prompt: str):
        for response in self.history.stream(
            {"input": prompt},
            config={"configurable": {"session_id": session_id}},
        ):
            yield response.content


assistant = ChatAssistant()

session_id = "Gk"
st.title("LangChain Chat with Message History")
st.write("This is a simple chat application using LangChain with message history stored in SQLite.")
session_id = st.text_input("Enter session ID:", value=session_id)

if st.button("Start new conversation"):
    st.session_state.chat_history = []
    assistant.clear_history(session_id)
        
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        with st.chat_message("user"):
            st.markdown(chat["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(chat["content"])


prompt = st.chat_input("Enter your query: ")
if prompt:
   
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            streamResponse = st.write_stream(assistant.stream(session_id, prompt))

    st.session_state.chat_history.append({"role": "assistant", "content": streamResponse})
    # st.balloons()
