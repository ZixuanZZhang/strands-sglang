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

"""Base classes for tool call parsing."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# Fallback tool name when we can't identify which tool the model tried to call
UNKNOWN_TOOL_NAME = "unknown_tool"


@dataclass(frozen=True, slots=True)
class ToolCallParseResult:
    """A parsed tool call request.

    For successful parses: name and input are populated, raw is None.
    For parse errors: name is extracted or UNKNOWN_TOOL_NAME, raw contains the unparseable content.
    """

    id: str
    name: str
    input: dict[str, Any] = field(default_factory=dict)
    raw: str | None = None

    @property
    def is_error(self) -> bool:
        """Check if this represents a parse error."""
        return self.raw is not None

    @property
    def payload(self) -> str:
        """Get the tool call payload string to pass to the tool executor.

        For successful parses, returns JSON-encoded input.
        For errors, returns the raw content (so model sees its mistake).
        """
        if self.is_error:
            return self.raw or ""
        return json.dumps(self.input)


class ToolCallParser(ABC):
    """Base class for tool call parsers.

    Subclasses implement `parse` to extract tool calls from model output.
    Only JSONDecodeError is handled; Strands validates arguments downstream.

    Example:
        >>> from strands_sglang import get_tool_parser
        >>> parser = get_tool_parser("hermes")
        >>> results = parser.parse('<tool_call>{"name": "foo", "arguments": {}}</tool_call>')
        >>> print(results[0].name)
        foo
    """

    @property
    def message_separator(self) -> str:
        """Separator between messages in the chat template.

        Different tokenizers use different separators between messages.
        This is used during incremental tokenization to ensure the TITO
        trajectory matches what `apply_chat_template` would produce.

        Default is no separator.
        """
        return ""

    @abstractmethod
    def parse(self, text: str) -> list[ToolCallParseResult]:
        """Parse tool calls from model output text.

        Args:
            text: Model output text.

        Returns:
            List of parsed tool call results.
        """
        ...

    def __call__(self, text: str) -> list[dict[str, Any]]:
        """Parse tool calls (callable interface for backwards compatibility).

        Args:
            text: Model output text.

        Returns:
            List of successful tool calls as dicts.
        """
        results = self.parse(text)
        return [{"id": tc.id, "name": tc.name, "input": tc.input} for tc in results if not tc.is_error]
