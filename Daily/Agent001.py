import sys
from pathlib import Path

# Add the project folder to Python's import path when this file is run directly.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.langsmith_config import configure_langsmith
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from utils.env_loader import load_env


OLLAMA_MODEL = "qwen3:8b"
OLLAMA_URL = "http://localhost:11434"

class DirectLLMExample:
    """Example 1: call the language model directly with invoke()."""

    def __init__(self) -> None:
        self.language_model = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_URL,
        )

    def run(self) -> None:
        response = self.language_model.invoke("What is the capital of India?")
        print("Direct llm.invoke response:")
        print(response.content)


class WeatherAgentExample:
    """Example 2: create an agent that can use a weather tool."""

    def __init__(self) -> None:
        self.language_model = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_URL,
        )
        self.agent = create_agent(
            model=self.language_model,
            tools=[self.get_weather],
        )

    def get_weather(self, city: str) -> str:
        """Return the example weather for a city."""
        return f"The weather in {city} is 30°C"

    def run(self) -> None:
        question = {
            "messages": [
                {"role": "user", "content": "What is the weather in Bangalore?"}
            ]
        }
        result = self.agent.invoke(question)
        print("Weather agent response:")
        print(result["messages"][-1].content)


def main() -> None:
    load_env()
    configure_langsmith()

    direct_llm_example = DirectLLMExample()
    direct_llm_example.run()

    weather_agent_example = WeatherAgentExample()
    weather_agent_example.run()


if __name__ == "__main__":
    main()
    


