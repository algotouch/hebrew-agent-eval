"""hebrew-agent-eval — Hebrew-language LLM evaluation harness.

Maintained by Agentics by Quatro — https://agentics.quatro.co.il
"""

from .core import Suite, Case, Response, Provider
from .providers import AnthropicProvider, OpenAIProvider

__version__ = "0.1.0"
__all__ = [
    "Suite",
    "Case",
    "Response",
    "Provider",
    "AnthropicProvider",
    "OpenAIProvider",
]
