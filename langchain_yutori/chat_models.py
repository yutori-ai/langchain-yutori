from __future__ import annotations

import os
from typing import Any

from langchain_openai import ChatOpenAI
from yutori.auth.credentials import resolve_api_key


class ChatYutoriN1(ChatOpenAI):
    """LangChain ChatModel wrapping the Yutori n1 browser navigation model.

    n1 is a pixels-to-actions LLM that processes screenshots and predicts
    browser actions (click, type, scroll, etc.). It uses the OpenAI Chat
    Completions interface, so it plugs into LangChain's OpenAI-compatible
    chat model stack.

    Authentication uses the ``YUTORI_API_KEY`` environment variable or the
    ``api_key`` constructor argument.

    Example::

        from langchain_yutori import ChatYutoriN1
        from langchain_core.messages import HumanMessage
        from yutori.n1 import aplaywright_screenshot_to_data_url

        llm = ChatYutoriN1(api_key="yt-...")
        image_url = await aplaywright_screenshot_to_data_url(page)

        message = HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": "What action should I take next?"},
        ])
        response = llm.invoke([message])
    """

    @classmethod
    def is_lc_serializable(cls) -> bool:
        # This wrapper intentionally does not promise a stable serialized config format.
        return False

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "n1-latest",
        base_url: str = "https://api.yutori.com/v1",
        **kwargs: Any,
    ) -> None:
        resolved_key = resolve_api_key(api_key or os.environ.get("YUTORI_API_KEY"))
        if not resolved_key:
            raise ValueError(
                "No API key provided. Set YUTORI_API_KEY, run 'yutori auth login', or pass api_key explicitly."
            )
        super().__init__(
            model=model,
            openai_api_key=resolved_key,
            openai_api_base=base_url,
            default_headers={"X-API-Key": resolved_key},
            **kwargs,
        )
