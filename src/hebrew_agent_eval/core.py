"""Core evaluation primitives for hebrew-agent-eval."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml
from pydantic import BaseModel, Field


class Case(BaseModel):
    """A single Hebrew test case."""

    id: str
    input: str = Field(description="Hebrew prompt presented to the agent.")
    expected_actions: list[str] = Field(
        default_factory=list,
        description="High-level actions the agent should take (logged or function-called).",
    )
    rubric: list[dict[str, str]] = Field(
        default_factory=list,
        description="Per-dimension scoring criteria. Used by LLM-as-judge.",
    )
    difficulty: str = Field(default="medium", description="easy / medium / hard")
    category: str = Field(default="", description="Filled in by the loader.")


@dataclass
class Response:
    """A single model response, with timing + cost metadata."""

    text: str
    latency_seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class Provider:
    """Base interface for an LLM provider plugin.

    Implement `generate` to return a Response for a given prompt. The harness
    handles judging, scoring, and aggregation.
    """

    name: str = "unnamed"

    def generate(self, prompt: str) -> Response:  # pragma: no cover
        raise NotImplementedError


@dataclass
class CaseResult:
    case_id: str
    provider: str
    response_text: str
    score: float  # 0.0 – 5.0
    score_breakdown: dict[str, float]
    latency_seconds: float
    cost_usd: float
    judge_reasoning: str = ""


@dataclass
class SuiteResult:
    suite_name: str
    case_results: list[CaseResult] = field(default_factory=list)

    def by_provider(self) -> dict[str, list[CaseResult]]:
        out: dict[str, list[CaseResult]] = {}
        for r in self.case_results:
            out.setdefault(r.provider, []).append(r)
        return out

    def print_leaderboard(self) -> None:
        try:
            from rich.console import Console
            from rich.table import Table
        except ImportError:
            self._print_leaderboard_plain()
            return

        console = Console()
        table = Table(title=f"Hebrew Agent Eval — {self.suite_name}")
        table.add_column("provider", style="cyan")
        table.add_column("accuracy", justify="right")
        table.add_column("latency_p50", justify="right")
        table.add_column("cost_per_run", justify="right")

        for provider, results in self.by_provider().items():
            mean_score = sum(r.score for r in results) / max(len(results), 1)
            accuracy = mean_score / 5.0
            latencies = sorted(r.latency_seconds for r in results)
            p50 = latencies[len(latencies) // 2] if latencies else 0.0
            mean_cost = sum(r.cost_usd for r in results) / max(len(results), 1)
            table.add_row(
                provider,
                f"{accuracy:.3f}",
                f"{p50:.1f}s",
                f"${mean_cost:.3f}",
            )

        console.print(table)

    def _print_leaderboard_plain(self) -> None:
        print(f"\n=== Hebrew Agent Eval — {self.suite_name} ===")
        for provider, results in self.by_provider().items():
            mean_score = sum(r.score for r in results) / max(len(results), 1)
            print(f"  {provider}: accuracy={mean_score / 5.0:.3f}, n={len(results)}")

    def save_html(self, path: str | Path) -> None:
        """Write a minimal HTML report (no extra deps)."""
        rows = []
        for provider, results in self.by_provider().items():
            mean = sum(r.score for r in results) / max(len(results), 1)
            rows.append(
                f"<tr><td>{provider}</td>"
                f"<td>{mean / 5.0:.3f}</td>"
                f"<td>{len(results)}</td></tr>"
            )
        html = (
            "<!DOCTYPE html><html dir='rtl' lang='he'>"
            "<head><meta charset='utf-8'><title>Hebrew Agent Eval Report</title></head>"
            "<body><h1>Hebrew Agent Eval — " + self.suite_name + "</h1>"
            "<table border='1' cellpadding='8'>"
            "<thead><tr><th>provider</th><th>accuracy</th><th>n</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            "</table></body></html>"
        )
        Path(path).write_text(html, encoding="utf-8")


class Suite:
    """A collection of Hebrew test cases that can be run against multiple providers."""

    def __init__(self, name: str, cases: list[Case]):
        self.name = name
        self.cases = cases

    @classmethod
    def load(cls, category: str = "all", root: Path | None = None) -> "Suite":
        """Load a category (or `all`) of test cases from the bundled tests folder."""
        if root is None:
            root = Path(__file__).parent.parent.parent / "tests"
        if not root.exists():
            # Fallback for installed package layout.
            import importlib.resources as resources
            root = Path(str(resources.files("hebrew_agent_eval"))) / "tests"

        cases: list[Case] = []
        if category == "all":
            yaml_files = sorted(root.glob("**/*.yaml"))
        else:
            yaml_files = sorted((root / category).glob("*.yaml"))

        for path in yaml_files:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            data["category"] = path.parent.name
            cases.append(Case(**data))

        return cls(name=category, cases=cases)

    def run(
        self,
        providers: Iterable[Provider],
        judge: Any | None = None,
    ) -> SuiteResult:
        """Run every case against every provider; score with LLM-as-judge."""
        result = SuiteResult(suite_name=self.name)
        for case in self.cases:
            for provider in providers:
                response = provider.generate(case.input)
                score, breakdown, reasoning = self._judge(
                    case=case, response_text=response.text, judge=judge
                )
                result.case_results.append(
                    CaseResult(
                        case_id=case.id,
                        provider=provider.name,
                        response_text=response.text,
                        score=score,
                        score_breakdown=breakdown,
                        latency_seconds=response.latency_seconds,
                        cost_usd=response.cost_usd,
                        judge_reasoning=reasoning,
                    )
                )
        return result

    def _judge(
        self,
        case: Case,
        response_text: str,
        judge: Any | None,
    ) -> tuple[float, dict[str, float], str]:
        """Score a response. Defaults to a simple length-and-keyword heuristic
        if no LLM judge is configured — for real eval, pass a judge."""
        if judge is None:
            # Heuristic placeholder so tests can run without API keys in CI.
            score = 3.0 if len(response_text) > 20 else 1.0
            return score, {"baseline": score}, "no judge configured"
        return judge.score(case=case, response_text=response_text)
