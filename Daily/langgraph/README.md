# LangGraph Modules — Configuration & Tracking

## Dependencies (from requirements.txt)
- `langchain==1.4.2`
- `langchain-openai==1.6.4`
- `langgraph==1.2.12`
- `langgraph-sdk==0.4.5`
- `langsmith==0.14.0`
- `python-dotenv==1.2.3`
- `pydantic==2.13.5`
- Plus transitive deps (httpx, openai, tiktoken, etc.)

## Environment Variables (`.env`)

### LLM Provider (Command Code API)
| Variable | Value Example | Required |
|---|---|---|
| `COMMAND_CODE_API_KEY` | `cmd-your-key-here` | Yes |
| `COMMAND_CODE_BASE_URL` | `https://api.commandcode.ai/provider/v1` | Yes |
| `COMMAND_CODE_MODEL` | `deepseek/deepseek-v4-flash` | Optional |

### LangSmith Tracing
| Variable | Value Example | Required |
|---|---|---|
| `LANGSMITH_API_KEY` | `lsv2_pt_your-key-here` | Yes |
| `LANGSMITH_TRACING_V2` | `true` | Optional |
| `LANGSMITH_PROJECT` | Project name (e.g. `llm-forge`) | Optional |

### Runtime Settings
| Variable | Value Example | Required |
|---|---|---|
| `LLM_MODE` | `online` / `local` | Yes |

### Jira / MCP (if needed)
| Variable | Value |
|---|---|
| `MCP_BASE_URL` | `https://mcp.atlassian.com/v2/mcp` |
| `MCP_SITE_URL` | `https://girishgk403.atlassian.net/` |
| `MCP_ISSUE_KEY` | `SCRUM-3` |

## Graphs

### 1. basic_chat_summary (`./basic_chat_summary.py:grpah`)
- **State:** `TypeState` → `prompt`, `response`, `summary`
- **Nodes:** `node1` (LLM invoke), `node2` (LLM summarize)
- **Edges:** START → node1 → node2 → END
- **Purpose:** Two-step chat flow with auto-summary

### 2. simple_chat_workflow (`./simple_chat_workflow.py:app`)
- **State:** `GraphState` → `messages`, `result`
- **Nodes:** `chat_node` (single LLM call)
- **Edges:** START → chat → END
- **Purpose:** Simple single-turn chatbot

### 3. test_case_generation_flow (`./test_case_generation_flow.py:graph`)
- **State:** `TestState` → `feature`, `description`, `category`, `manual_testcases`, `ui_automation_code`, `api_automation_code`
- **Nodes:** 
  - `analyze` → classify feature as ui/api
  - `testcases` → generate manual test cases
  - `ui_automation` → generate Playwright TS code
  - `api_automation` → generate pytest code
- **Edges:** START → analyze → testcases → *[conditional]* → ui/api automation → END
- **Purpose:** Requirement analyzer with test case generation + conditional routing

## LangSmith / LangStudio Setup

### To enable tracing on all graphs:
```bash
export LANGSMITH_API_KEY=lsv2_pt_your-key-here
export LANGSMITH_TRACING_V2=true
export LANGSMITH_PROJECT="llm-forge"
```

### Run with tracing (example):
```python
import os
os.environ["LANGSMITH_TRACING_V2"] = "true"

from test_case_generation_flow import graph
result = graph.invoke({
    "feature": "Create User API",
    "description": "POST /api/users endpoint"
})
```

Runs will appear in [LangSmith](https://smith.langchain.com) under the project set in `LANGSMITH_PROJECT`.

### Deploy with LangGraph CLI
```bash
pip install langgraph-cli
langgraph up          # Start local dev server
langgraph cloud login # Login to LangGraph Cloud
```
