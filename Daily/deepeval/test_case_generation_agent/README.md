# Test Case Generation Workflow

The LangGraph workflow reads a requirement, generates structured test cases, evaluates them with DeepEval, and waits for human review.

## Workflow

```text
┌─────────┐
│  START  │
└────┬────┘
     │ initial state:
     │ input_file, output_file, attempts = 0
     v
┌──────────────────────┐
│ docuemnt_loader      │
│ Read input_file      │
│ -> user_story        │
└──────────┬───────────┘
           │
           v
┌────────────────────────────┐
│ requirement_Analyzer       │
│ LLM analyzes user_story    │
│ -> analysis                │
└────────────┬───────────────┘
             │
             v
┌──────────────────────────────────┐
│ testcase_generator               │
│ If feedback is empty:            │
│   generate exactly one case      │
│ Else:                            │
│   regenerate from reviewer input │
│ Pydantic validates TestCases     │
│ -> JSON test_cases               │
│ -> attempts = attempts + 1       │
└──────────────┬───────────────────┘
               │
               v
┌──────────────────────────────────┐
│ validate_testcases               │
│ Build LLMTestCase                │
│ Run one DeepEval evaluation:     │
│   - AnswerRelevancyMetric        │
│   - FaithfulnessMetric           │
│ -> scores, reasons, pass flags   │
└──────────────┬───────────────────┘
               │
               v
┌──────────────────────────────────┐
│ human_review                     │
│ interrupt for decision + feedback│
└──────────────┬───────────────────┘
               │
       ┌───────┼───────────────┐
       │       │               │
   approve    edit           reject
       │       │               │
       v       │               v
┌─────────────┐│          ┌─────────┐
│save_testcases││          │   END   │
│ Write        ││          │ rejected│
│ output/      ││          └─────────┘
│test_cases.json││
└──────┬──────┘│
       │       │
       v       │ attempts < 3
   ┌─────┐     └───────────────┐
   │ END │                     │
   │saved│                     v
   └─────┘       ┌────────────────────────┐
                 │ testcase_generator    │
                 │ feedback-driven retry │
                 └────────────┬───────────┘
                              │
                              └──> validate_testcases

If edit is selected when attempts >= 3:
    edit ──> END (rejected)
```

## Retry behavior

- The first generation creates one test case.
- `edit` sends reviewer feedback back to `testcase_generator`.
- Regeneration stops after three attempts and routes to rejection.
- `approve` writes the generated JSON to `output/test_cases.json`.
- `reject` ends the workflow without saving the test cases.
