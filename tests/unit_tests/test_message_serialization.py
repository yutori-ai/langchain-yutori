"""Verify that langchain-openai's message serialization produces the exact
shape Navigator's API requires (per https://docs.yutori.com/reference/navigator).

These tests guard against future regressions in langchain-openai's message
conversion. Any drift would silently break Navigator request payloads.
"""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_yutori import ChatYutoriNavigator


def _payload(messages):
    model = ChatYutoriNavigator(api_key="yt_test")
    return model._get_request_payload(messages, stop=None)


def test_first_turn_user_message_with_screenshot_matches_navigator_shape():
    msg = HumanMessage(
        content=[
            {"type": "text", "text": "Search for Yutori on Google."},
            {"type": "image_url", "image_url": {"url": "data:image/webp;base64,AAAA"}},
        ]
    )
    [serialized] = _payload([msg])["messages"]
    assert serialized["role"] == "user"
    assert serialized["content"] == [
        {"type": "text", "text": "Search for Yutori on Google."},
        {"type": "image_url", "image_url": {"url": "data:image/webp;base64,AAAA"}},
    ]


def test_assistant_tool_calls_serialize_with_function_wrapper_and_string_args():
    # Navigator (like OpenAI) expects {"id","type":"function","function":{"name","arguments":"<JSON string>"}}.
    ai_msg = AIMessage(
        content="thinking",
        tool_calls=[{"id": "call_1", "name": "left_click", "args": {"coordinate": [500, 250]}}],
    )
    [serialized] = _payload([ai_msg])["messages"]
    assert serialized["role"] == "assistant"
    [tool_call] = serialized["tool_calls"]
    assert tool_call["id"] == "call_1"
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "left_click"
    # arguments must be a JSON string per the API reference, not a dict.
    assert isinstance(tool_call["function"]["arguments"], str)
    assert json.loads(tool_call["function"]["arguments"]) == {"coordinate": [500, 250]}


def test_tool_message_with_text_and_screenshot_lands_inside_tool_role():
    # Navigator requires the next-turn screenshot to live inside the tool message,
    # not in a separate user message.
    tool_msg = ToolMessage(
        content=[
            {"type": "text", "text": "Clicked 1x with left\nCurrent URL: https://example.com/"},
            {"type": "image_url", "image_url": {"url": "data:image/webp;base64,BBBB"}},
        ],
        tool_call_id="call_1",
    )
    [serialized] = _payload([tool_msg])["messages"]
    assert serialized["role"] == "tool"
    assert serialized["tool_call_id"] == "call_1"
    assert serialized["content"] == [
        {"type": "text", "text": "Clicked 1x with left\nCurrent URL: https://example.com/"},
        {"type": "image_url", "image_url": {"url": "data:image/webp;base64,BBBB"}},
    ]


def test_full_loop_history_preserves_navigator_shape():
    # Smoke that a complete user → assistant(tool_calls) → tool(text+screenshot) trio
    # arrives at the API in the exact ordering and shape Navigator documents.
    history = [
        HumanMessage(
            content=[
                {"type": "text", "text": "Search for Yutori on Google."},
                {"type": "image_url", "image_url": {"url": "data:image/webp;base64,AAAA"}},
            ]
        ),
        AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "left_click", "args": {"coordinate": [500, 250]}}],
        ),
        ToolMessage(
            content=[
                {"type": "text", "text": "Clicked 1x with left\nCurrent URL: https://google.com/"},
                {"type": "image_url", "image_url": {"url": "data:image/webp;base64,BBBB"}},
            ],
            tool_call_id="call_1",
        ),
    ]
    serialized = _payload(history)["messages"]
    assert [m["role"] for m in serialized] == ["user", "assistant", "tool"]
    assert serialized[2]["tool_call_id"] == "call_1"
    assert any(part.get("type") == "image_url" for part in serialized[2]["content"])
