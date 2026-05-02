# hebrew-agent-eval

> **A Hebrew-language evaluation harness for LLMs and AI agents.** Tests how well Claude, GPT-4, Gemini, and other models actually handle real-world Israeli business prompts — not just benchmarks built for English.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)

Maintained by [**Agentics**](https://agentics.co.il) — based on a benchmark we built internally for [our 60+ Hebrew AI agent deployments](https://agentics.co.il/case-studies).

---

## Why this exists

Existing LLM benchmarks (MMLU, HellaSwag, MT-Bench, AlpacaEval) are overwhelmingly English. The few that include Hebrew use **translated** test sets — which means they measure translation accuracy, not native Hebrew competence.

Israeli business prompts have unique properties:

- **Mixed register** — formal vocabulary mixed with slang and Anglicisms (חברה / "business" / "סטארט-אפ")
- **Code-switching** — Hebrew + English in the same sentence is the norm, not the exception
- **Israeli-specific entities** — חשבונית ירוקה, רשם החברות, ביטוח לאומי, תעודת זהות, ח״פ
- **Right-to-left + numerals** — currency, dates, phone numbers all flip orientation
- **No vowel marks (niqqud)** — production text rarely has them, so models trained on niqqud-heavy data underperform

This harness tests all of that.

## What's included

- **150+ Hebrew test cases** across 8 categories (customer service, sales qualification, scheduling, invoicing, lead routing, BI queries, technical support, internal ops)
- **LLM-as-judge** evaluation harness (uses Claude Opus 4.7 by default; configurable)
- **Side-by-side comparison** — run multiple models on the same set, get a leaderboard
- **Detailed diagnostics** — per-category accuracy, common failure patterns, latency, token cost
- **Reproducible** — fixed seeds, frozen test sets, semver-pinned dependencies

## Install

```bash
pip install hebrew-agent-eval
```

Or from source:

```bash
git clone https://github.com/algotouch/hebrew-agent-eval.git
cd hebrew-agent-eval
pip install -e .
```

## Quick start

```python
from hebrew_agent_eval import Suite, OpenAIProvider, AnthropicProvider

suite = Suite.load("customer_service")  # or 'all'

results = suite.run([
    AnthropicProvider("claude-sonnet-4-6"),
    AnthropicProvider("claude-opus-4-7"),
    OpenAIProvider("gpt-4o"),
])

results.print_leaderboard()
results.save_html("report.html")
```

Output:

```
====================================================================
Hebrew Agent Eval — customer_service (n=24)
====================================================================
                        accuracy  latency_p50  cost_per_run
claude-opus-4-7         0.917     2.4s         $0.087
claude-sonnet-4-6       0.875     1.1s         $0.018
gpt-4o                  0.792     1.3s         $0.022
====================================================================
```

## Test categories

| Category | # cases | What it tests |
| --- | --- | --- |
| `customer_service` | 24 | Empathy, complaint handling, mixed-register Hebrew |
| `sales_qualification` | 18 | BANT-style discovery, Israeli sales etiquette |
| `scheduling` | 20 | Calendar logic with Israeli holidays + Hebrew date formats |
| `invoicing` | 22 | Tax-invoice math, VAT logic, ח״פ validation |
| `lead_routing` | 15 | Industry classification from Hebrew job titles + company names |
| `bi_queries` | 18 | Hebrew → SQL, with Israeli column-name conventions |
| `technical_support` | 17 | Mixed Hebrew/English error messages, tone calibration |
| `internal_ops` | 18 | Approvals, expense reports, vendor follow-ups |

## Adding test cases

Each test is a YAML file in `tests/<category>/`:

```yaml
id: cs_017
input: |
  היי, ביקשתי החזר לפני שבוע ועדיין לא קיבלתי כלום. זה ממש לא בסדר.
  ההזמנה היא 84421.
expected_actions:
  - acknowledge_emotion
  - check_order_status(order_id=84421)
  - escalate_or_resolve
rubric:
  - tone: empathetic_assertive_hebrew
  - clarity: addresses_specific_complaint
  - safety: no_promises_without_data
difficulty: medium
```

## Provider plugins

Built-in providers:

- `AnthropicProvider(model)` — Claude API
- `OpenAIProvider(model)` — OpenAI API
- `GoogleProvider(model)` — Gemini API
- `BedrockProvider(model)` — AWS Bedrock
- `OllamaProvider(model)` — local Ollama models

Custom provider:

```python
from hebrew_agent_eval import Provider, Response

class MyProvider(Provider):
    name = "my-model"
    def generate(self, prompt: str) -> Response:
        ...
```

## Methodology

LLM-as-judge with Claude Opus 4.7 by default. Each response is scored against the case's rubric on a 1–5 scale per dimension. Final score = mean across dimensions. We sample-validate against human reviewers on ~10% of runs and report inter-rater agreement.

Anthropic's research indicates LLM-as-judge agreement with humans on Hebrew tasks is in the 85–92% range — we observe similar in our own validation.

## License

MIT. See [LICENSE](LICENSE).

## About Agentics

[Agentics](https://agentics.co.il) is an Israeli AI agents implementation agency based in Tel Aviv (founded 2024 by Eyal Yakobi Miller). 60+ deployments, 409 hours saved per month per client on average. We open-source the parts of our stack that benefit the broader Hebrew NLP community.

Other repos in the Agentics open-source family:

- [`mcp-greeninvoice`](https://github.com/algotouch/mcp-greeninvoice) — MCP server for חשבונית ירוקה
- [`israeli-business-prompts`](https://github.com/algotouch/israeli-business-prompts)
- [`geo-il-spec`](https://github.com/algotouch/geo-il-spec)
