"""CLI entry point: `hebrew-agent-eval run --category customer_service`."""

from __future__ import annotations

import typer

from .core import Suite
from .providers import AnthropicProvider, OpenAIProvider

app = typer.Typer(help="Hebrew Agent Eval — measure LLM quality on Israeli business prompts.")


@app.command()
def run(
    category: str = typer.Option("customer_service", help="Test category, or 'all'."),
    anthropic_model: str | None = typer.Option(None, help="Anthropic model id."),
    openai_model: str | None = typer.Option(None, help="OpenAI model id."),
    out: str = typer.Option("report.html", help="Path to write HTML report."),
):
    """Run the eval suite against one or more LLM providers."""
    providers = []
    if anthropic_model:
        providers.append(AnthropicProvider(anthropic_model))
    if openai_model:
        providers.append(OpenAIProvider(openai_model))

    if not providers:
        typer.echo("Pass at least one --anthropic-model or --openai-model.")
        raise typer.Exit(1)

    suite = Suite.load(category)
    typer.echo(f"Loaded {len(suite.cases)} cases from category '{category}'")

    results = suite.run(providers)
    results.print_leaderboard()
    results.save_html(out)
    typer.echo(f"Report written to: {out}")
