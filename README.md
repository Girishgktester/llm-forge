# LLM Forge

A small learning project for running one LangChain agent in two ways: an online OpenAI model, or a local Qwen model through Ollama. LangSmith tracing is turned on so each run can be inspected.

## What it does

`main.py` loads environment variables, enables LangSmith, builds an agent with LangChain `create_agent()`, and sends one example question:

> Say hello in one sentence.

The model depends on a single mode value, `online` or `offline`.

| Mode | Model | Where it runs |
| --- | --- | --- |
| `online` | `gpt-4o-mini` | OpenAI API |
| `offline` | `qwen3:8b` | Ollama on this machine |

## Project layout

```text
LLM_Forge/
├── core/
│   ├── langsmith_config.py   # LangSmith tracing settings
│   ├── llm_config.py          # online/offline model names and the system prompt
│   └── agent_factory.py       # create_agent() for the selected mode
├── utils/
│   └── env_loader.py          # loads .env
├── .env.example               # template for local secrets and LLM_MODE
├── .gitignore                 # keeps .env and the virtualenv out of git
├── requirements.txt
├── main.py
└── README.md
```

## Prerequisites

- Python 3.12
- An OpenAI API key, if you use `online`
- A LangSmith API key, if you want traces in the LangSmith UI
- [Ollama](https://ollama.com/) installed locally, if you use `offline`

This laptop already has Ollama and the `qwen3:8b` model. If you set the project up on another machine, pull that model once:

```powershell
ollama pull qwen3:8b
```

Ollama must be running before an offline call. Starting any `ollama` command usually starts the local server on `http://localhost:11434`.

## Setup

From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and replace the placeholder key. `.env` is listed in `.gitignore` and must stay on your machine. Do not commit it.

```text
OPENAI_API_KEY=sk-your-key-here
LANGSMITH_API_KEY=lsv2-your-key-here
LLM_MODE=online
```

`LANGSMITH_API_KEY` is optional for a local reply, and required if you want the run to show up in LangSmith. `LLM_MODE` is used only when you do not pass a mode on the command line.

## Choose online or offline

Pass the mode as the first argument. That value overrides `LLM_MODE` in `.env`.

```powershell
.\.venv\Scripts\python.exe main.py online
.\.venv\Scripts\python.exe main.py offline
```

To make one mode the default, set it in `.env`:

```text
LLM_MODE=offline
```

Then run:

```powershell
.\.venv\Scripts\python.exe main.py
```

If neither the argument nor `LLM_MODE` is set, the default is `online`.

Accepted values are only `online` and `offline`. Anything else raises `ValueError`.

The names themselves live in `core/llm_config.py`:

```python
ONLINE_MODEL = "openai:gpt-4o-mini"
OFFLINE_MODEL = "ollama:qwen3:8b"
```

`create_agent()` receives a `provider:model` string. For Ollama that is `ollama:qwen3:8b`, which LangChain splits into provider `ollama` and model `qwen3:8b`.

## LangSmith

`core/langsmith_config.py` sets these only when they are not already in the environment:

- `LANGSMITH_TRACING=true`
- `LANGCHAIN_TRACING_V2=true`
- `LANGSMITH_PROJECT=LLM_Forge`

Traces for both modes are grouped under the **LLM_Forge** project when `LANGSMITH_API_KEY` is present.

## Change the example question

The question is the user message in `main.py`:

```python
agent.invoke({"messages": [("user", "Say hello in one sentence.")]})
```

Replace that string to ask something else. The printed result is the full agent state, including the human message and the model reply.

## Notes

- The first offline run loads `qwen3:8b` (about 5.2 GB) into memory, so it is slower than later runs.
- Online runs need a working `OPENAI_API_KEY` and network access.
- The virtual environment (`.venv`) and `.env` are local only.
