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

"""HTTP client utilities for high-concurrency workloads."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

# OpenAI's default connection limits (from openai/_constants.py)
DEFAULT_MAX_CONNECTIONS = 1000
DEFAULT_MAX_KEEPALIVE_CONNECTIONS = 100


def create_client(
    base_url: str,
    *,
    max_connections: int = DEFAULT_MAX_CONNECTIONS,
    max_keepalive_connections: int | None = None,
    timeout: float = 600.0,
    connect_timeout: float = 5.0,
) -> httpx.AsyncClient:
    """Create an httpx.AsyncClient configured for high-concurrency workloads.

    Uses OpenAI's connection pool defaults (1000 max connections, 100 keepalive)
    which are 10x higher than httpx defaults, preventing PoolTimeout errors.

    Args:
        base_url: Server URL (e.g., "http://localhost:8000").
        max_connections: Maximum concurrent connections (default: 1000).
        max_keepalive_connections: Idle connections kept warm (default: max_connections // 10).
        timeout: Request timeout in seconds (default: 600s, matching OpenAI).
        connect_timeout: TCP connection timeout in seconds (default: 5s).

    Returns:
        Configured httpx.AsyncClient for connection pooling.

    Example:
        >>> client = create_client("http://localhost:8000", max_connections=512)
        >>> model = SGLangModel(tokenizer=tokenizer, client=client)
    """
    if max_keepalive_connections is None:
        max_keepalive_connections = max(DEFAULT_MAX_KEEPALIVE_CONNECTIONS, max_connections // 10)

    logger.info(
        f"Creating httpx client: max_connections={max_connections}, "
        f"max_keepalive_connections={max_keepalive_connections}"
    )

    return httpx.AsyncClient(
        base_url=base_url,
        timeout=httpx.Timeout(timeout, connect=connect_timeout),
        limits=httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
        ),
    )
