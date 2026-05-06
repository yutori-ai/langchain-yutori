# langchain-yutori

LangChain integration for the [Yutori API](https://docs.yutori.com) — Navigator browser control, browser automation, deep research, and recurring web monitors.

## Installation

```bash
pip install langchain langchain-yutori
```

This package is implemented as a standalone LangChain integration package. It uses the official `yutori`
Python SDK for Browsing, Research, and Scouts, and wraps Navigator as a LangChain chat model.
Installing `langchain-yutori` also installs the `yutori` Python package, plus the `yutori` CLI.

## Components

| Class | Type | Description |
|---|---|---|
| `ChatYutoriNavigator` | `ChatModel` | Yutori Navigator browser-control model (OpenAI-compatible, defaults to n1.5) |
| `YutoriBrowsingTool` | `BaseTool` | Execute web browsing tasks on a remote browser |
| `YutoriResearchTool` | `BaseTool` | Perform deep and broad research using 100+ tools |
| `YutoriScoutingTool` | `BaseTool` | Create and manage recurring web monitors with Scouts |

## Authentication

Recommended:

```bash
yutori auth login
```

This opens your browser and saves your API key locally for the SDK and this package to use.

Or set your API key via environment variable:

```bash
export YUTORI_API_KEY="yt-..."
```

Or pass it directly to each class.

Get your API key at [platform.yutori.com](https://platform.yutori.com).

## Usage

### ChatYutoriNavigator

Navigator is Yutori's pixels-to-actions LLM for browser navigation. It accepts screenshots and returns browser actions (click, type, scroll, etc.). The current version is **n1.5** (the default); older versions like n1 remain selectable via the `model` argument.

```python
from langchain_yutori import ChatYutoriNavigator
from langchain_core.messages import HumanMessage
from yutori.navigator import aplaywright_screenshot_to_data_url

llm = ChatYutoriNavigator()  # defaults to n1.5-latest, uses YUTORI_API_KEY env var
image_url = await aplaywright_screenshot_to_data_url(page)

message = HumanMessage(content=[
    {"type": "text", "text": "What is the next action to complete the task: 'Add item to cart'?"},
    {"type": "image_url", "image_url": {"url": image_url}},
])
response = llm.invoke([message])
# Returns tool_calls with browser actions
```

To pin to a specific Navigator version:

```python
llm = ChatYutoriNavigator(model="n1-latest")     # older Navigator n1
llm = ChatYutoriNavigator(model="n1.5-latest")   # current Navigator n1.5 (default)
```

With Playwright, use the SDK helper so the image is captured with the SDK's default JPEG capture
settings and encoded to a WebP data URL optimized for Navigator.

`ChatYutoriNavigator` accepts image URLs but does not capture or preprocess screenshots itself, so
if you are using Playwright you should call the SDK helper directly before passing the image into
LangChain.

If you execute returned browser actions yourself, Navigator coordinates are normalized to a
`1000x1000` space. Convert them back into viewport pixels with the SDK helper:

```python
from yutori.navigator import denormalize_coordinates

coords = [500, 250]
x, y = denormalize_coordinates(coords, width=1280, height=800)
await page.mouse.click(x, y)
```

#### Tool sets and request options

Navigator's available action set is server-side and version-tagged. Select it (and optionally
disable specific actions) via first-class constructor params; they are forwarded to the OpenAI
client's `extra_body`:

```python
llm = ChatYutoriNavigator(
    tool_set="browser_tools_expanded-20260403",   # adds extract_elements, find, set_element_value, execute_js
    disable_tools=["hold_key", "drag"],
)
```

Other Navigator-specific request fields (e.g. `json_schema` for structured output) can be passed
through `extra_body` directly — they're merged with the first-class params:

```python
llm = ChatYutoriNavigator(
    tool_set="browser_tools_core-20260403",
    extra_body={"json_schema": {"type": "object", "properties": {...}}},
)
```

#### Multi-turn loop and message-history shape

After Navigator returns `tool_calls`, you execute the action in your browser, capture a fresh
screenshot, and feed the result back as a `ToolMessage` whose `content` is a multimodal list
of `[text, image_url]`. Navigator requires the new screenshot inside the tool message — not in a
separate `HumanMessage`:

```python
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from yutori.navigator import aplaywright_screenshot_to_data_url, denormalize_coordinates

history = [
    HumanMessage(content=[
        {"type": "text", "text": "Search for Yutori on Google."},
        {"type": "image_url", "image_url": {"url": await aplaywright_screenshot_to_data_url(page)}},
    ])
]

while True:
    response: AIMessage = llm.invoke(history)
    history.append(response)
    if not response.tool_calls:
        break  # Navigator finished the task; response.content has the summary

    for call in response.tool_calls:
        result_text = await execute_in_browser(page, call)  # your code; use denormalize_coordinates etc.
        history.append(ToolMessage(
            tool_call_id=call["id"],
            content=[
                {"type": "text", "text": f"{result_text}\nCurrent URL: {page.url}"},
                {"type": "image_url", "image_url": {"url": await aplaywright_screenshot_to_data_url(page)}},
            ],
        ))
```

Notes:
- **Don't add a system message.** Navigator's docs recommend placing extra instructions in the first user message instead.
- **Include `Current URL: ...`** in the tool result text — it improves grounding.
- **Don't trim messages.** For long trajectories, use `yutori.navigator.create_trimmed` / `acreate_trimmed`, which drop only old screenshots while keeping the message structure intact.

For the full Navigator input requirements and action schema, see the Yutori docs:
https://docs.yutori.com/llm-quickstart and https://docs.yutori.com/reference/navigator

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
