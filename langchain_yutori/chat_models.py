from __future__ import annotations

import os
from typing import Any

from langchain_openai import ChatOpenAI
from yutori.auth.credentials import resolve_api_key
from yutori.navigator import N1_5_MODEL


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
