from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_openai import ChatOpenAI
from yutori.auth.credentials import resolve_api_key
from yutori.navigator import N1_5_MODEL


def navigator_tool_result(
    *,
    tool_call_id: str,
    result_text: str,
    current_url: str,
    screenshot_data_url: str | None = None,
) -> ToolMessage:
    """Construct a ``ToolMessage`` in the shape Navigator expects for an
    action result.

    Codifies three Navigator conventions in a single typed call so users
    can't drift from them:

    1. The screenshot for the next turn lives **inside** the tool message,
       not in a separate ``HumanMessage``.
    2. The result text is suffixed with ``"Current URL: <url>"`` for
       grounding (Navigator's docs note this is load-bearing for
       attribution).
    3. Content blocks are ordered ``[text, image_url]``, matching the docs
       example and what Navigator was trained on.

    When Navigator returns multiple ``tool_calls`` in a single turn, only
    the **last** tool result needs the post-action screenshot — pass
    ``screenshot_data_url=None`` for intermediate results in a batched
    response (per Navigator's docs).

    Args:
        tool_call_id: ID of the assistant ``tool_call`` this result answers.
        result_text: Short description of what executing the action did,
            e.g. ``"Clicked 1x with left"`` or ``"Typed 'Yutori'"``.
        current_url: The browser's URL after the action completed. Always
            include this; the convention exists for grounding even when
            the URL didn't change.
        screenshot_data_url: WebP data URL of the post-action screenshot.
            Provide for the last (or only) tool result in a turn; omit
            for intermediate results in a batched response.

    Returns:
        A ``ToolMessage`` with content shaped exactly as Navigator's
        reference documents.
    """
    content: list[dict[str, Any]] = [
        {"type": "text", "text": f"{result_text}\nCurrent URL: {current_url}"},
    ]
    if screenshot_data_url is not None:
        content.append({"type": "image_url", "image_url": {"url": screenshot_data_url}})
    return ToolMessage(tool_call_id=tool_call_id, content=content)


def trim_navigator_history(
    messages: list[BaseMessage],
    *,
    keep_recent: int = 2,
) -> list[BaseMessage]:
    """Drop ``image_url`` content blocks from older messages while keeping
    every message and its text intact.

    Navigator's docs explicitly recommend never dropping messages but tolerate
    dropping old screenshots: preserving the full message structure keeps
    attribution accurate while old images consume context the newer
    screenshot already supersedes. Apply this on the LangChain side before
    each ``llm.invoke`` for long-running loops::

        history = trim_navigator_history(history)
        response = llm.invoke(history)

    The input list and original ``BaseMessage`` instances are never mutated;
    messages whose content was unchanged are returned by reference, and only
    messages that lost an ``image_url`` block are reissued as fresh copies.

    Args:
        messages: LangChain message list, typically the running history of a
            Navigator loop.
        keep_recent: Number of most-recent messages whose images are kept
            verbatim. Older messages have any ``image_url`` blocks removed
            from their content. Defaults to ``2`` — usually the last
            assistant turn plus its tool result.

    Raises:
        ValueError: if ``keep_recent`` is negative.
    """
    if keep_recent < 0:
        raise ValueError("keep_recent must be >= 0")
    cutoff = max(0, len(messages) - keep_recent)
    out: list[BaseMessage] = []
    for i, msg in enumerate(messages):
        if i >= cutoff or not isinstance(msg.content, list):
            out.append(msg)
            continue
        trimmed = [
            block
            for block in msg.content
            if not (isinstance(block, dict) and block.get("type") == "image_url")
        ]
        if len(trimmed) == len(msg.content):
            out.append(msg)
        else:
            out.append(msg.model_copy(update={"content": trimmed}))
    return out


class ChatYutoriNavigator(ChatOpenAI):
    """LangChain ChatModel wrapping Yutori's Navigator browser-control model.

    Navigator is Yutori's pixels-to-actions LLM for browser navigation. It
    accepts screenshots and returns ``tool_calls`` describing the next browser
    action (click, type, scroll, etc.). The current version is n1.5; older
    versions like n1 remain selectable via the ``model`` argument.

    Executing the returned actions is the application's responsibility — this
    class returns them as ``AIMessage.tool_calls`` and points users at the
    Yutori SDK's execution helpers (``denormalize_coordinates``,
    ``map_key_to_playwright``, etc.). Use ``YutoriBrowsingTool`` for the
    turnkey hosted-browser path instead.

    Navigator uses the OpenAI Chat Completions interface, so it plugs into
    LangChain's OpenAI-compatible chat model stack. Navigator-specific request
    knobs (``tool_set``, ``disable_tools``) are first-class constructor
    parameters; they are forwarded into the OpenAI client's ``extra_body``.
    Any other Navigator-only fields can still be passed via ``extra_body``
    directly.

    Authentication uses the ``YUTORI_API_KEY`` environment variable, the
    credentials saved by ``yutori auth login``, or the ``api_key`` constructor
    argument.

    Example::

        from langchain_yutori import ChatYutoriNavigator
        from langchain_core.messages import HumanMessage
        from yutori.navigator import aplaywright_screenshot_to_data_url

        llm = ChatYutoriNavigator(
            tool_set="browser_tools_expanded-20260403",
            disable_tools=["hold_key", "drag"],
        )
        image_url = await aplaywright_screenshot_to_data_url(page)

        message = HumanMessage(content=[
            {"type": "text", "text": "What action should I take next?"},
            {"type": "image_url", "image_url": {"url": image_url}},
        ])
        response = llm.invoke([message])

    To pin to a specific Navigator version (e.g. the older n1)::

        llm = ChatYutoriNavigator(model="n1-latest")
    """

    @classmethod
    def is_lc_serializable(cls) -> bool:
        # This wrapper intentionally does not promise a stable serialized config format.
        return False

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = N1_5_MODEL,
        base_url: str = "https://api.yutori.com/v1",
        tool_set: str | None = None,
        disable_tools: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        resolved_key = resolve_api_key(api_key or os.environ.get("YUTORI_API_KEY"))
        if not resolved_key:
            raise ValueError(
                "No API key provided. Set YUTORI_API_KEY, run 'yutori auth login', or pass api_key explicitly."
            )

        extra_body = dict(kwargs.pop("extra_body", None) or {})
        if tool_set is not None:
            extra_body["tool_set"] = tool_set
        if disable_tools is not None:
            extra_body["disable_tools"] = list(disable_tools)
        if extra_body:
            kwargs["extra_body"] = extra_body

        super().__init__(
            model=model,
            openai_api_key=resolved_key,
            openai_api_base=base_url,
            **kwargs,
        )

    def _create_chat_result(
        self,
        response: Any,
        generation_info: dict[str, Any] | None = None,
    ):
        result = super()._create_chat_result(response, generation_info)
        response_dict = response if isinstance(response, dict) else response.model_dump()

        if request_id := response_dict.get("request_id"):
            result.llm_output = {**(result.llm_output or {}), "request_id": request_id}
            for generation in result.generations:
                generation.message.response_metadata["request_id"] = request_id
                generation.message.additional_kwargs.setdefault("request_id", request_id)

        return result
