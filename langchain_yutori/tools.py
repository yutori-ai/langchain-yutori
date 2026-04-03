from __future__ import annotations

import asyncio
import json
import os
import time
from enum import Enum
from typing import Any, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from yutori import AsyncYutoriClient, YutoriClient
from yutori.auth.credentials import resolve_api_key


_MIN_POLL_INTERVAL = 60.0
_DEFAULT_POLL_INTERVAL = 60.0  # seconds between status polls
_DEFAULT_TIMEOUT = 1200.0  # maximum seconds to wait for task completion (tasks take 5–15 min)


# ---------------------------------------------------------------------------
# Browsing tool
# ---------------------------------------------------------------------------


class _BrowsingInput(BaseModel):
    task: str = Field(description="Natural language description of the browsing task to perform")
    start_url: str = Field(description="URL to start browsing from (e.g. 'https://example.com')")
    max_steps: int | None = Field(default=None, description="Maximum number of agent steps (1-100)")


class YutoriBrowsingTool(BaseTool):
    """Execute a web browsing task on Yutori's cloud browser.

    Creates a browsing task, then polls until it succeeds or fails.
    Returns the full task result as a JSON string.

    Example::

        from langchain_yutori import YutoriBrowsingTool

        tool = YutoriBrowsingTool(api_key="yt-...")
        result = tool.run({
            "task": "Find the price of the first result",
            "start_url": "https://amazon.com/s?k=headphones",
        })
    """

    name: str = "yutori_browsing"
    description: str = (
        "Execute a web browsing task using Yutori's cloud browser agent. "
        "Provide a natural language task description and the starting URL. "
        "The agent will navigate the page and return the result."
    )
    args_schema: Type[BaseModel] = _BrowsingInput

    api_key: str | None = Field(default=None, description="Yutori API key (defaults to SDK credential resolution)")
    base_url: str = Field(default="https://api.yutori.com/v1")
    poll_interval: float = Field(default=_DEFAULT_POLL_INTERVAL)
    timeout: float = Field(default=_DEFAULT_TIMEOUT)

    def model_post_init(self, __context: Any) -> None:
        resolved_key = resolve_api_key(self.api_key or os.environ.get("YUTORI_API_KEY"))
        if not resolved_key:
            raise ValueError(
                "No API key provided. Set YUTORI_API_KEY, run 'yutori auth login', or pass api_key explicitly."
            )
        object.__setattr__(self, "api_key", resolved_key)
        if self.poll_interval < _MIN_POLL_INTERVAL:
            raise ValueError(f"poll_interval must be at least {_MIN_POLL_INTERVAL:.0f} seconds")

    def _run(self, task: str, start_url: str, max_steps: int | None = None) -> str:
        with YutoriClient(api_key=self.api_key, base_url=self.base_url, timeout=30) as client:
            task_data = client.browsing.create(task=task, start_url=start_url, max_steps=max_steps)
            task_id = task_data["task_id"]
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                data = client.browsing.get(task_id)
                status = data.get("status")
                if status == "succeeded":
                    return json.dumps(data)
                if status == "failed":
                    raise RuntimeError(f"Browsing task {task_id} failed: {data.get('error', 'unknown error')}")
                time.sleep(self.poll_interval)

        raise TimeoutError(f"Browsing task {task_id} did not complete within {self.timeout}s")

    async def _arun(self, task: str, start_url: str, max_steps: int | None = None) -> str:
        async with AsyncYutoriClient(api_key=self.api_key, base_url=self.base_url, timeout=30) as client:
            task_data = await client.browsing.create(task=task, start_url=start_url, max_steps=max_steps)
            task_id = task_data["task_id"]
            deadline = asyncio.get_event_loop().time() + self.timeout
            while asyncio.get_event_loop().time() < deadline:
                data = await client.browsing.get(task_id)
                status = data.get("status")
                if status == "succeeded":
                    return json.dumps(data)
                if status == "failed":
                    raise RuntimeError(f"Browsing task {task_id} failed: {data.get('error', 'unknown error')}")
                await asyncio.sleep(self.poll_interval)

        raise TimeoutError(f"Browsing task {task_id} did not complete within {self.timeout}s")


# ---------------------------------------------------------------------------
# Research tool
# ---------------------------------------------------------------------------


class _ResearchInput(BaseModel):
    query: str = Field(description="Natural language research query")
    user_timezone: str | None = Field(default=None, description="User timezone, e.g. 'America/Los_Angeles'")
    user_location: str | None = Field(default=None, description="User location, e.g. 'San Francisco, CA, US'")


class YutoriResearchTool(BaseTool):
    """Perform deep web research using Yutori's research infrastructure.

    Yutori Research uses 100+ MCP tools to answer complex research queries.
    Creates a research task, then polls until it succeeds or fails.
    Returns the full task result (including the research report) as a JSON string.

    Example::

        from langchain_yutori import YutoriResearchTool

        tool = YutoriResearchTool(api_key="yt-...")
        result = tool.run({"query": "What are the latest AI research papers on browser agents?"})
    """

    name: str = "yutori_research"
    description: str = (
        "Perform deep web research using Yutori's research agent (100+ tools). "
        "Provide a natural language research query. Returns a comprehensive research report."
    )
    args_schema: Type[BaseModel] = _ResearchInput

    api_key: str | None = Field(default=None, description="Yutori API key (defaults to SDK credential resolution)")
    base_url: str = Field(default="https://api.yutori.com/v1")
    poll_interval: float = Field(default=_DEFAULT_POLL_INTERVAL)
    timeout: float = Field(default=_DEFAULT_TIMEOUT)

    def model_post_init(self, __context: Any) -> None:
        resolved_key = resolve_api_key(self.api_key or os.environ.get("YUTORI_API_KEY"))
        if not resolved_key:
            raise ValueError(
                "No API key provided. Set YUTORI_API_KEY, run 'yutori auth login', or pass api_key explicitly."
            )
        object.__setattr__(self, "api_key", resolved_key)
        if self.poll_interval < _MIN_POLL_INTERVAL:
            raise ValueError(f"poll_interval must be at least {_MIN_POLL_INTERVAL:.0f} seconds")

    def _run(
        self,
        query: str,
        user_timezone: str | None = None,
        user_location: str | None = None,
    ) -> str:
        with YutoriClient(api_key=self.api_key, base_url=self.base_url, timeout=30) as client:
            task_data = client.research.create(
                query=query,
                user_timezone=user_timezone,
                user_location=user_location,
            )
            task_id = task_data["task_id"]
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                data = client.research.get(task_id)
                status = data.get("status")
                if status == "succeeded":
                    return json.dumps(data)
                if status == "failed":
                    raise RuntimeError(f"Research task {task_id} failed: {data.get('error', 'unknown error')}")
                time.sleep(self.poll_interval)

        raise TimeoutError(f"Research task {task_id} did not complete within {self.timeout}s")

    async def _arun(
        self,
        query: str,
        user_timezone: str | None = None,
        user_location: str | None = None,
    ) -> str:
        async with AsyncYutoriClient(api_key=self.api_key, base_url=self.base_url, timeout=30) as client:
            task_data = await client.research.create(
                query=query,
                user_timezone=user_timezone,
                user_location=user_location,
            )
            task_id = task_data["task_id"]
            deadline = asyncio.get_event_loop().time() + self.timeout
            while asyncio.get_event_loop().time() < deadline:
                data = await client.research.get(task_id)
                status = data.get("status")
                if status == "succeeded":
                    return json.dumps(data)
                if status == "failed":
                    raise RuntimeError(f"Research task {task_id} failed: {data.get('error', 'unknown error')}")
                await asyncio.sleep(self.poll_interval)

        raise TimeoutError(f"Research task {task_id} did not complete within {self.timeout}s")


# ---------------------------------------------------------------------------
# Scouting tool
# ---------------------------------------------------------------------------


class _ScoutAction(str, Enum):
    CREATE = "create"
    LIST = "list"
    GET = "get"
    GET_UPDATES = "get_updates"
    PAUSE = "pause"
    RESUME = "resume"
    DELETE = "delete"


class _ScoutingInput(BaseModel):
    action: _ScoutAction = Field(
        description=(
            "Action to perform: "
            "'create' (start a new recurring monitor), "
            "'list' (list all scouts), "
            "'get' (get details for a specific scout), "
            "'get_updates' (fetch latest findings from a scout), "
            "'pause' (pause a scout), "
            "'resume' (resume a paused scout), "
            "'delete' (delete a scout)."
        )
    )
    query: str | None = Field(
        default=None,
        description="Natural language monitoring query. Required for 'create'.",
    )
    scout_id: str | None = Field(
        default=None,
        description="Scout ID (UUID). Required for 'get', 'get_updates', 'pause', 'resume', 'delete'.",
    )
    output_interval: int | None = Field(
        default=None,
        description="Seconds between scout runs. Minimum 1800 (30 min), default 86400 (daily). Only for 'create'.",
    )
    limit: int | None = Field(
        default=None,
        description="Maximum number of results to return. Applies to 'list' and 'get_updates'.",
    )


class YutoriScoutingTool(BaseTool):
    """Manage Yutori scouts — recurring web monitors that run on a schedule.

    Supports creating, listing, reading, and managing scouts, as well as
    fetching their latest update findings.

    Example::

        from langchain_yutori import YutoriScoutingTool

        tool = YutoriScoutingTool(api_key="yt-...")

        # Create a scout
        tool.run({"action": "create", "query": "Alert me when Tesla announces new models"})

        # List all scouts
        tool.run({"action": "list"})

        # Get updates from a scout
        tool.run({"action": "get_updates", "scout_id": "abc123"})
    """

    name: str = "yutori_scouting"
    description: str = (
        "Manage Yutori scouts — recurring web monitors that run on a schedule and surface findings. "
        "Actions: create (set up a new monitor), list (view all monitors), "
        "get (details for one monitor), get_updates (fetch latest findings), "
        "pause, resume, delete."
    )
    args_schema: Type[BaseModel] = _ScoutingInput

    api_key: str | None = Field(default=None, description="Yutori API key (defaults to SDK credential resolution)")
    base_url: str = Field(default="https://api.yutori.com/v1")

    def model_post_init(self, __context: Any) -> None:
        resolved_key = resolve_api_key(self.api_key or os.environ.get("YUTORI_API_KEY"))
        if not resolved_key:
            raise ValueError(
                "No API key provided. Set YUTORI_API_KEY, run 'yutori auth login', or pass api_key explicitly."
            )
        object.__setattr__(self, "api_key", resolved_key)

    def _get_global_updates(self, client: YutoriClient, limit: int | None = None) -> dict[str, Any]:
        now = time.time()
        params: dict[str, Any] = {
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - 24 * 60 * 60)),
            "end_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        }
        if limit is not None:
            params["page_size"] = limit

        response = client._client.get(
            f"{client._base_url}/scouting/updates",
            headers={"X-API-Key": client._api_key},
            params=params,
        )
        response.raise_for_status()
        return response.json()

    async def _aget_global_updates(self, client: AsyncYutoriClient, limit: int | None = None) -> dict[str, Any]:
        now = time.time()
        params: dict[str, Any] = {
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - 24 * 60 * 60)),
            "end_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        }
        if limit is not None:
            params["page_size"] = limit

        response = await client._client.get(
            f"{client._base_url}/scouting/updates",
            headers={"X-API-Key": client._api_key},
            params=params,
        )
        response.raise_for_status()
        return response.json()

    def _run(
        self,
        action: str,
        query: str | None = None,
        scout_id: str | None = None,
        output_interval: int | None = None,
        limit: int | None = None,
    ) -> str:
        with YutoriClient(api_key=self.api_key, base_url=self.base_url, timeout=30) as client:
            return self._dispatch(client, action, query, scout_id, output_interval, limit)

    async def _arun(
        self,
        action: str,
        query: str | None = None,
        scout_id: str | None = None,
        output_interval: int | None = None,
        limit: int | None = None,
    ) -> str:
        async with AsyncYutoriClient(api_key=self.api_key, base_url=self.base_url, timeout=30) as client:
            return await self._adispatch(client, action, query, scout_id, output_interval, limit)

    def _dispatch(
        self,
        client: YutoriClient,
        action: str,
        query: str | None,
        scout_id: str | None,
        output_interval: int | None,
        limit: int | None,
    ) -> str:
        if action == _ScoutAction.CREATE:
            if not query:
                raise ValueError("'query' is required for action='create'")
            response = client.scouts.create(query=query, output_interval=output_interval or 86400)
            return json.dumps(response)

        elif action == _ScoutAction.LIST:
            return json.dumps(client.scouts.list(limit=limit))

        elif action == _ScoutAction.GET:
            if not scout_id:
                raise ValueError("'scout_id' is required for action='get'")
            return json.dumps(client.scouts.get(scout_id))

        elif action == _ScoutAction.GET_UPDATES:
            if scout_id:
                return json.dumps(client.scouts.get_updates(scout_id, limit=limit))
            return json.dumps(self._get_global_updates(client, limit=limit))

        elif action == _ScoutAction.PAUSE:
            if not scout_id:
                raise ValueError("'scout_id' is required for action='pause'")
            return json.dumps(client.scouts.update(scout_id, status="paused"))

        elif action == _ScoutAction.RESUME:
            if not scout_id:
                raise ValueError("'scout_id' is required for action='resume'")
            return json.dumps(client.scouts.update(scout_id, status="active"))

        elif action == _ScoutAction.DELETE:
            if not scout_id:
                raise ValueError("'scout_id' is required for action='delete'")
            client.scouts.delete(scout_id)
            return json.dumps({"deleted": True, "scout_id": scout_id})

        else:
            raise ValueError(f"Unknown action: {action!r}. Must be one of: {[a.value for a in _ScoutAction]}")

    async def _adispatch(
        self,
        client: AsyncYutoriClient,
        action: str,
        query: str | None,
        scout_id: str | None,
        output_interval: int | None,
        limit: int | None,
    ) -> str:
        if action == _ScoutAction.CREATE:
            if not query:
                raise ValueError("'query' is required for action='create'")
            response = await client.scouts.create(query=query, output_interval=output_interval or 86400)
            return json.dumps(response)

        elif action == _ScoutAction.LIST:
            return json.dumps(await client.scouts.list(limit=limit))

        elif action == _ScoutAction.GET:
            if not scout_id:
                raise ValueError("'scout_id' is required for action='get'")
            return json.dumps(await client.scouts.get(scout_id))

        elif action == _ScoutAction.GET_UPDATES:
            if scout_id:
                return json.dumps(await client.scouts.get_updates(scout_id, limit=limit))
            return json.dumps(await self._aget_global_updates(client, limit=limit))

        elif action == _ScoutAction.PAUSE:
            if not scout_id:
                raise ValueError("'scout_id' is required for action='pause'")
            return json.dumps(await client.scouts.update(scout_id, status="paused"))

        elif action == _ScoutAction.RESUME:
            if not scout_id:
                raise ValueError("'scout_id' is required for action='resume'")
            return json.dumps(await client.scouts.update(scout_id, status="active"))

        elif action == _ScoutAction.DELETE:
            if not scout_id:
                raise ValueError("'scout_id' is required for action='delete'")
            await client.scouts.delete(scout_id)
            return json.dumps({"deleted": True, "scout_id": scout_id})

        else:
            raise ValueError(f"Unknown action: {action!r}. Must be one of: {[a.value for a in _ScoutAction]}")
