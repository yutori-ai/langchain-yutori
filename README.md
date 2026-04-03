# langchain-yutori

LangChain integration for the [Yutori API](https://docs.yutori.com) — n1 browser control, browser automation, deep research, and recurring web monitors.

## Installation

```bash
pip install langchain-yutori
```

This package is implemented as a standalone LangChain integration package. It uses the official `yutori`
Python SDK for Browsing, Research, and Scouts, and wraps n1 as a LangChain chat model.

## Components

| Class | Type | Description |
|---|---|---|
| `ChatYutoriN1` | `ChatModel` | Yutori n1 browser navigation model (OpenAI-compatible) |
| `YutoriBrowsingTool` | `BaseTool` | Execute web browsing tasks on a remote browser |
| `YutoriResearchTool` | `BaseTool` | Perform deep and broad research using 100+ tools |
| `YutoriScoutingTool` | `BaseTool` | Create and manage recurring web monitors with Scouts |

## Authentication

Set your API key via environment variable:

```bash
export YUTORI_API_KEY="yt-..."
```

Or pass it directly to each class.

Get your API key at [platform.yutori.com](https://platform.yutori.com).

## Usage

### ChatYutoriN1

n1 is Yutori's pixels-to-actions LLM for browser navigation. It accepts screenshots and returns browser actions (click, type, scroll, etc.).

```python
from langchain_yutori import ChatYutoriN1

llm = ChatYutoriN1()  # uses YUTORI_API_KEY env var

# Use as a drop-in for ChatOpenAI
response = llm.invoke("Describe what you see on this page.")
print(response.content)
```

With a screenshot (base64 WebP, 1280×800 recommended):

```python
from langchain_core.messages import HumanMessage

message = HumanMessage(content=[
    {"type": "image_url", "image_url": {"url": "data:image/webp;base64,<base64-encoded-screenshot>"}},
    {"type": "text", "text": "What is the next action to complete the task: 'Add item to cart'?"},
])
response = llm.invoke([message])
# Returns tool_calls with browser actions
```

For the full n1 input requirements and action schema, see the Yutori docs: https://docs.yutori.com

### YutoriBrowsingTool

Runs a browser automation agent on Yutori's cloud browser. The tool creates the task and polls until it completes (up to 20 minutes by default; tasks typically take 5–15 minutes).

```python
from langchain_yutori import YutoriBrowsingTool

tool = YutoriBrowsingTool()

result = tool.run({
    "task": "Find the price of the MacBook Pro 14-inch M4",
    "start_url": "https://www.apple.com",
})
print(result)  # JSON string with task result
```

In a LangChain agent:

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_yutori import YutoriBrowsingTool

tools = [YutoriBrowsingTool()]
llm = ChatOpenAI(model="gpt-4o")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with web browsing capabilities."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
executor.invoke({"input": "What's the current price of AAPL on Yahoo Finance?"})
```

### YutoriResearchTool

Performs deep and broad research using Yutori's research agent (100+ MCP tools). Creates the task and polls until complete.

```python
from langchain_yutori import YutoriResearchTool

tool = YutoriResearchTool()

result = tool.run({
    "query": "What are the top 5 AI coding assistants in 2026 and how do their pricing models compare?",
    "user_location": "San Francisco, CA, US",
})
print(result)  # JSON string with research report
```

### YutoriScoutingTool

Manages Yutori Scouts — recurring web monitors that run on a schedule and surface findings.

```python
from langchain_yutori import YutoriScoutingTool

tool = YutoriScoutingTool()

# Create a scout (runs daily by default)
result = tool.run({
    "action": "create",
    "query": "Monitor Hacker News for posts about browser automation agents",
    "output_interval": 3600,  # hourly
})

# List all scouts
scouts = tool.run({"action": "list"})

# Get updates from a specific scout
updates = tool.run({
    "action": "get_updates",
    "scout_id": "abc123...",
    "limit": 10,
})

# Pause / resume / delete
tool.run({"action": "pause", "scout_id": "abc123..."})
tool.run({"action": "resume", "scout_id": "abc123..."})
tool.run({"action": "delete", "scout_id": "abc123..."})
```

## Configuration

Browsing and Research tools accept `poll_interval` (seconds between status checks, default 60, minimum 60) and `timeout` (max wait seconds, default 1200):

```python
tool = YutoriBrowsingTool(
    api_key="yt-...",
    poll_interval=10.0,
    timeout=300.0,
)
```

## Links

- [Yutori documentation](https://docs.yutori.com)
- [API platform](https://platform.yutori.com)
- [PyPI](https://pypi.org/project/langchain-yutori/)
- [Yutori Python SDK](https://github.com/yutori-ai/yutori-sdk-python)
