# Copyright 2025 Horizon RL Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tool call parsers for different model formats.

This module provides parsers that extract tool calls from model outputs.
Different models use different formats for tool calls in their chat templates.

Design for RL Training:
- Only handle `JSONDecodeError` (can't extract anything from malformed JSON)
- Let Strands validate arguments against tool schemas
- Parse errors become tool calls with error info for model feedback
"""

from typing import Any

from .base import UNKNOWN_TOOL_NAME, ToolCallParser, ToolCallParseResult
from .hermes import HermesToolCallParser

# Parser registry
TOOL_PARSER_REGISTRY: dict[str, type[ToolCallParser]] = {
    "hermes": HermesToolCallParser,
}


def get_tool_parser(name: str, **kwargs: Any) -> ToolCallParser:
    """Get a tool parser by name.

    Args:
        name: Parser name (e.g., "hermes").
        **kwargs: Arguments passed to the parser constructor.

    Returns:
        Instantiated parser.

    Raises:
        KeyError: If parser name is not registered.

    Example:
        >>> parser = get_tool_parser("hermes")
        >>> parser = get_tool_parser("hermes", think_tokens=None)
    """
    if name not in TOOL_PARSER_REGISTRY:
        available = ", ".join(sorted(TOOL_PARSER_REGISTRY.keys()))
        raise KeyError(f"Unknown tool parser: {name!r}. Available: {available}")
    return TOOL_PARSER_REGISTRY[name](**kwargs)


__all__ = [
    # Base
    "ToolCallParseResult",
    "ToolCallParser",
    "UNKNOWN_TOOL_NAME",
    # Parsers
    "HermesToolCallParser",
    # Registry
    "TOOL_PARSER_REGISTRY",
    "get_tool_parser",
]
