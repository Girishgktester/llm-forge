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

Studio graph IDs use the same `graph_` prefix as their module filenames:

| Studio graph ID | Module | Purpose |
|---|---|---|
| `graph_chat_summary` | `graph_chat_summary.py` | Summarize a chat in two LLM steps |
| `graph_simple_chat` | `graph_simple_chat.py` | Single-turn chat |
| `graph_testcase_automation` | `graph_testcase_automation.py` | Classify a requirement, create test cases, then generate UI or API automation |
| `graph_qa_automation` | `graph_qa_automation.py` | Generate and evaluate test cases and automation with human review |
| `graph_playwright_test_generation` | `graph_playwright_test_generation.py` | Generate test cases, pause for approval, then generate and validate Playwright code |
| `graph_tool_calling_workflow` | `graph_tool_calling_workflow.py` | Chat assistant that can calculate basic arithmetic and return the current UTC time |

To try `graph_tool_calling_workflow` in Studio, provide an initial state like:
```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is (19 + 2) * 2? Use the calculator."
    }
  ]
}
```

These standalone examples are also named with the `graph_` prefix:

| Module | Purpose |
|---|---|
| `graph_human_approval_qa.py` | Human approval before answering a question |
| `graph_testcase_approval.py` | Generate test cases and request approval |
| `graph_testcase_approval_simple.py` | Basic test-case approval example |
| `graph_testcase_feedback.py` | Generate test cases with an edit/approval loop |
| `graph_testcase_review_audit.py` | Test-case review with retries, authorization, and audit logging |

The standalone examples use console input or their own in-memory checkpointers and are intentionally not registered as Studio graphs.

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

from graph_testcase_automation import graph
result = graph.invoke({
    "feature": "Create User API",
    "description": "POST /api/users endpoint"
})
```

Runs will appear in [LangSmith](https://smith.langchain.com) under the project set in `LANGSMITH_PROJECT`.

### Start the local Studio API
From the repository root, run:
```bash
cd Daily/langgraph
langgraph dev
```
Keep the server running and open the Studio URL printed by the command.

For LangGraph Cloud deployment, install the CLI and authenticate separately:
```bash
pip install langgraph-cli
langgraph cloud login
```
