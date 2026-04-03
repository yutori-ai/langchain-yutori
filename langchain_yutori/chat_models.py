from __future__ import annotations

import os
from typing import Any

from langchain_openai import ChatOpenAI
from yutori.auth.credentials import resolve_api_key


class ChatYutoriN1(ChatOpenAI):
    """LangChain ChatModel wrapping the Yutori n1 browser navigation model.

    n1 is a pixels-to-actions LLM that processes screenshots and predicts
    browser actions (click, type, scroll, etc.). It uses the OpenAI Chat
    Completions interface, making it a drop-in replacement for any LangChain
    workflow that uses ``ChatOpenAI``.

    Authentication uses the ``YUTORI_API_KEY`` environment variable or the
    ``api_key`` constructor argument.

    Example::

        from langchain_yutori import ChatYutoriN1

        llm = ChatYutoriN1(api_key="yt-...")
        response = llm.invoke("What is on this page?")

    Or with screenshots (base64 image content)::

        from langchain_core.messages import HumanMessage

        message = HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": "data:image/webp;base64,..."}},
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
