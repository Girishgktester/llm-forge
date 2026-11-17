# LLM Forge

LLM Forge is a hands-on collection of experiments for building LLM applications with LangChain and LangGraph. It includes chat agents, tool calling, conversational memory, retrieval-augmented generation (RAG), QA/test automation workflows, and browser automation examples.

The repository is a learning lab rather than a single packaged application: most files under `Daily/` are runnable examples that can be explored independently. The examples may use different providers and credentials; check each script's configuration before running it.

## What is included

- **LangChain fundamentals:** chat models, prompt templates, runnables, message history, and agent patterns.
- **Tools and integrations:** tool calling, web search, Jira/MCP experiments, and test-case generation.
- **LangGraph workflows:** stateful chat, summarization, arithmetic tools, branching QA flows, human approval, feedback/retry loops, and audit-oriented examples.
- **RAG experiments:** vector search, keyword and hybrid retrieval, MMR, reranking, metadata, PDF and web-page ingestion, plus source-change detection and indexing utilities.
- **QA automation:** example flows that classify a requirement, generate manual tests, and produce UI or API automation; some workflows add DeepEval evaluation and human review.
- **Playwright and TypeScript:** browser-test examples and Playwright runner configuration.
- **A simple root agent:** `main.py` selects an online Command Code model or a local Ollama model.

## Repository layout

```text
.
├── core/                         # Root agent/model selection and LangSmith setup
├── utils/                        # Shared environment-loading helper
├── Daily/
│   ├── langchain/
│   │   ├── Fundamentals/         # Model, prompt, and runnable examples
│   │   ├── Tools_and_Agents/     # Tool and agent examples
│   │   ├── Chat_and_Memory/      # Chat history and Streamlit example
│   │   ├── Integrations/         # External integrations such as Jira/MCP
│   │   └── RAG/                  # RAG demos and ingestion utilities
│   └── langgraph/                # Graph definitions, Studio config, and guide
├── dataset/                      # Example QA knowledge and guidelines
├── tests/                        # Playwright TypeScript tests
├── main.py                       # Root LangChain agent demo
├── requirements.txt              # Pinned Python dependencies
├── package.json                  # Playwright test scripts and Node dependencies
└── .env.example                  # Local environment-variable template
```

## Requirements

- Python 3.12 (the examples and local environment are set up for Python 3.12).
- Node.js and npm for the Playwright suite.
- Credentials for the provider used by the example you want to run.
- [Ollama](https://ollama.com/) for local-model examples.
- Optional services/tools such as LangSmith, Tavily, Jira/MCP, or DeepEval may require their own credentials or setup.

## Set up the environment

Run commands from the repository root.

### Python

Create a virtual environment and install the pinned dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and replace placeholders with your own credentials. Never commit `.env` or paste API keys into source files.

### Node.js / Playwright

Install the JavaScript dependencies and the Chromium browser used by the configured Playwright project:

```powershell
npm install
npx playwright install chromium
```

The Playwright configuration targets Chromium. The examples currently visit public websites, so those tests need network access.

## Configure model providers

The root `main.py` agent selects its provider using the first command-line argument, or `LLM_MODE` if no argument is supplied:

| Mode | Provider | Model | Configuration |
| --- | --- | --- | --- |
| `online` (default) | Command Code OpenAI-compatible API | `deepseek/deepseek-v4-flash` by default | `COMMAND_CODE_API_KEY`, optional `COMMAND_CODE_BASE_URL` and `COMMAND_CODE_MODEL` |
| `offline` | Ollama running locally | `qwen3:8b` | Install Ollama and pull the model |

For the local model, run:

```powershell
ollama pull qwen3:8b
```

Ensure Ollama is running before invoking an offline example. The root agent's online model configuration is in `core/command_code_llm.py`; its online/offline selection is in `core/llm_config.py`.

The LangGraph examples are independent and may instantiate `ChatOpenAI` directly with a model name in their source. Those examples may require `OPENAI_API_KEY` and a model available to your OpenAI-compatible endpoint. The root `COMMAND_CODE_*` settings do not automatically reconfigure every standalone example. Review the model initialization in the specific graph or script before running it.

### Environment variables

`.env.example` contains placeholders for:

| Variable | Used for |
| --- | --- |
| `COMMAND_CODE_API_KEY` | Root agent's online Command Code model |
| `COMMAND_CODE_BASE_URL` | Optional Command Code API endpoint override |
| `COMMAND_CODE_MODEL` | Optional online model override |
| `OPENAI_API_KEY` | Examples that call OpenAI directly |
| `LANGSMITH_API_KEY` | Sending traces to LangSmith |
| `LLM_MODE` | Default root-agent mode: `online` or `offline` |
| `MCP_BASE_URL`, `MCP_SITE_URL`, `MCP_ISSUE_KEY` | Jira/MCP example configuration |

Additional integrations can require other variables. For example, `main.py` also contains a Tavily search example; it imports `tavily-python` and reads `TAVILI_API_KEY`. That optional package/key is not included in `requirements.txt` or `.env.example`.

## Run the root agent

Choose a mode explicitly:

```powershell
.\.venv\Scripts\python.exe main.py online
.\.venv\Scripts\python.exe main.py offline
```

Or set `LLM_MODE` in `.env` and omit the argument:

```powershell
.\.venv\Scripts\python.exe main.py
```

The command-line mode takes precedence over `LLM_MODE`. Accepted values are `online` and `offline`; an unsupported value raises an error. The root agent's sample prompt is in `main.py`.

## Explore LangGraph

Graph definitions and a graph catalog are in [`Daily/langgraph/README.md`](Daily/langgraph/README.md). The catalog describes each graph's purpose, sample inputs, LangSmith tracing, and Studio setup.

To start the local LangGraph development server, run it from the graph directory:

```powershell
Set-Location .\Daily\langgraph
$env:PYTHONIOENCODING = "utf-8"
langgraph dev
```

The CLI reads `langgraph.json`, which registers graph IDs such as `graph_simple_chat`, `graph_testcase_automation`, `graph_qa_automation`, and `graph_tool_message_state`. Open the Studio URL printed by the CLI and provide the input shape expected by the selected graph. Some workflows pause for human approval and need to be resumed with a decision.

`langgraph.json` points to the repository-level `.env`. Set up that file before starting graphs that call external model APIs. Use LangSmith tracing by setting `LANGSMITH_API_KEY`; the graph README contains tracing examples.

## Explore LangChain and RAG

Browse the topic folders under `Daily/langchain/`. Most scripts are standalone experiments and can be run from the project root, for example:

```powershell
.\.venv\Scripts\python.exe .\Daily\langchain\Fundamentals\agent001.py
```

RAG examples are under `Daily/langchain/RAG/RAG_Basics/` and cover:

- Basic vector-store retrieval and a full assistant flow.
- PDF and web-page loading.
- Keyword (BM25), hybrid/ensemble retrieval, and MMR diversification.
- Metadata annotations and cross-encoder reranking.

The RAG folder also contains ingestion/index-management helpers and focused tests for content hashing, source-change detection, manifest management, and source loading. Check a script's imports and configuration before execution; optional integrations may have dependencies or credentials beyond the shared requirements.

## Run tests

Run the Python tests for the RAG ingestion utilities:

```powershell
.\.venv\Scripts\python.exe -m pytest .\Daily\langchain\RAG
```

Run the Playwright suite:

```powershell
npm test
```

Other useful Playwright commands:

```powershell
npm run test:headed
npm run test:ui
npm run test:debug
npm run report
```

The configured browser tests include a Playwright documentation smoke test and a login-flow example against a public practice site. Treat these as external-site examples, not tests of a locally hosted LLM Forge web application.

## Tracing and troubleshooting

- Set `LANGSMITH_API_KEY` to send supported LangChain/LangGraph runs to LangSmith. The root setup enables tracing and uses the `LLM_Forge` project name unless the environment already sets one.
- Online model requests require valid provider credentials and network access.
- Local model requests require Ollama to be running and the configured model to be available.
- Human-in-the-loop graphs pause by design; provide the requested approval or feedback to continue.
- If an individual example fails due to a missing optional integration, inspect its imports and environment-variable reads and install/configure only that integration.

## Security

Keep API keys, access tokens, and personal service URLs out of commits. Use `.env` for local secrets; `.env` and `.venv/` are ignored by Git. Before pushing changes, review `git status` and the staged diff to ensure no credentials, local databases, or generated artifacts are included unintentionally.
