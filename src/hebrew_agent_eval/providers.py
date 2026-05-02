"""LLM provider plugins for hebrew-agent-eval."""

from __future__ import annotations

import os
import time

from .core import Provider, Response


class AnthropicProvider(Provider):
    """Provider plugin for Anthropic Claude models.

    Requires ANTHROPIC_API_KEY env var. The default model is sonnet-4-6;
    pass any active model id explicitly.
    """

    def __init__(self, model: str = "claude-sonnet-4-6", system: str | None = None):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "AnthropicProvider requires `anthropic` — install via `pip install anthropic`."
            ) from e
        self.model = model
        self.name = model
        self.system = system or (
            "You are a helpful AI assistant for Israeli small businesses. "
            "Respond in natural Hebrew unless the user writes in English. "
            "Be direct, warm, and avoid corporate filler."
        )
        from anthropic import Anthropic
        self._client = Anthropic()

    def generate(self, prompt: str) -> Response:
        start = time.perf_counter()
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=self.system,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = time.perf_counter() - start
        text = "".join(b.text for b in msg.content if hasattr(b, "text"))
        # Cost per million tokens (Sonnet 4.6 baseline; override per-model if needed)
        in_cost_per_m = 3.0
        out_cost_per_m = 15.0
        cost = (
            msg.usage.input_tokens * in_cost_per_m / 1_000_000
            + msg.usage.output_tokens * out_cost_per_m / 1_000_000
        )
        return Response(
            text=text,
            latency_seconds=latency,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
            cost_usd=cost,
        )


class OpenAIProvider(Provider):
    """Provider plugin for OpenAI / Azure OpenAI models."""

    def __init__(self, model: str = "gpt-4o", system: str | None = None):
        try:
            import openai  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "OpenAIProvider requires `openai` — install via `pip install openai`."
            ) from e
        self.model = model
        self.name = model
        self.system = system or (
            "You are a helpful AI assistant for Israeli small businesses. "
            "Respond in natural Hebrew unless the user writes in English."
        )
        from openai import OpenAI
        self._client = OpenAI()

    def generate(self, prompt: str) -> Response:
        start = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048,
        )
        latency = time.perf_counter() - start
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        # GPT-4o baseline rates; override per-model.
        in_cost_per_m = 2.50
        out_cost_per_m = 10.0
        cost = (
            (usage.prompt_tokens or 0) * in_cost_per_m / 1_000_000
            + (usage.completion_tokens or 0) * out_cost_per_m / 1_000_000
        )
        return Response(
            text=text,
            latency_seconds=latency,
            input_tokens=usage.prompt_tokens or 0,
            output_tokens=usage.completion_tokens or 0,
            cost_usd=cost,
        )
