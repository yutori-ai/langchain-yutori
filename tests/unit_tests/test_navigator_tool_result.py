"""Tests for navigator_tool_result.

Covers the three conventions the helper codifies (screenshot inside the tool
message, "Current URL:" suffix, [text, image_url] ordering), the batched-
tool-call case (last result has screenshot, intermediates don't), and an
end-to-end serialization check that the produced ToolMessage round-trips
into Navigator's documented wire shape.
"""

import pytest
from langchain_core.messages import ToolMessage

from langchain_yutori import ChatYutoriNavigator, navigator_tool_result

# --- shape ---------------------------------------------------------------------


def test_returns_tool_message_with_correct_id():
    msg = navigator_tool_result(
        tool_call_id="call_xyz",
        result_text="Clicked 1x with left",
        current_url="https://example.com/",
        screenshot_data_url="data:image/webp;base64,AAAA",
    )
    assert isinstance(msg, ToolMessage)
    assert msg.tool_call_id == "call_xyz"


def test_content_order_is_text_then_image():
    """Navigator was trained on [text, image_url] ordering — must not flip."""
    msg = navigator_tool_result(
        tool_call_id="c1",
        result_text="Scrolled",
        current_url="https://example.com/",
        screenshot_data_url="data:image/webp;base64,XXX",
    )
    assert isinstance(msg.content, list)
    [first, second] = msg.content
    assert first["type"] == "text"
    assert second["type"] == "image_url"


def test_text_has_current_url_suffix_with_single_newline():
    msg = navigator_tool_result(
        tool_call_id="c1",
        result_text="Typed 'Yutori'",
        current_url="https://google.com/search?q=Yutori",
        screenshot_data_url="data:image/webp;base64,X",
    )
    text_block = msg.content[0]
    assert text_block["text"] == "Typed 'Yutori'\nCurrent URL: https://google.com/search?q=Yutori"


def test_screenshot_url_threaded_through_image_block():
    msg = navigator_tool_result(
        tool_call_id="c1",
        result_text="r",
        current_url="https://example.com/",
        screenshot_data_url="data:image/webp;base64,UNIQUE_PAYLOAD",
    )
    image_block = msg.content[1]
    assert image_block == {
        "type": "image_url",
        "image_url": {"url": "data:image/webp;base64,UNIQUE_PAYLOAD"},
    }


# --- batched-tool-call case ----------------------------------------------------


def test_screenshot_omitted_yields_text_only_content():
    """Intermediate results in a batched tool_call response carry no screenshot."""
    msg = navigator_tool_result(
        tool_call_id="c1",
        result_text="Clicked",
        current_url="https://example.com/",
    )
    assert len(msg.content) == 1
    assert msg.content[0] == {"type": "text", "text": "Clicked\nCurrent URL: https://example.com/"}


def test_explicit_none_screenshot_yields_text_only():
    msg = navigator_tool_result(
        tool_call_id="c1",
        result_text="Clicked",
        current_url="https://example.com/",
        screenshot_data_url=None,
    )
    assert len(msg.content) == 1


# --- API enforcement -----------------------------------------------------------


def test_signature_is_keyword_only():
    """Positional ordering of four similar strings is a footgun — enforce kwargs."""
    with pytest.raises(TypeError):
        navigator_tool_result(  # type: ignore[misc]
            "c1",
            "result",
            "https://example.com/",
            "data:image/webp;base64,X",
        )


# --- end-to-end: round-trip through Navigator wire format ----------------------


def test_helper_output_round_trips_to_navigator_wire_shape():
    """ToolMessage from the helper → _get_request_payload → dict shape matching
    Navigator's documented tool message format."""
    msg = navigator_tool_result(
        tool_call_id="c1",
        result_text="Clicked 1x with left",
        current_url="https://google.com/",
        screenshot_data_url="data:image/webp;base64,SCREENSHOT",
    )
    model = ChatYutoriNavigator(api_key="yt_test")
    [serialized] = model._get_request_payload([msg], stop=None)["messages"]
    assert serialized == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": [
            {"type": "text", "text": "Clicked 1x with left\nCurrent URL: https://google.com/"},
            {"type": "image_url", "image_url": {"url": "data:image/webp;base64,SCREENSHOT"}},
        ],
    }


def test_helper_output_for_batched_intermediate_round_trips():
    """No-screenshot variant must also serialize to a valid Navigator tool message
    (text-only content list)."""
    msg = navigator_tool_result(
        tool_call_id="c1",
        result_text="Clicked",
        current_url="https://google.com/",
    )
    model = ChatYutoriNavigator(api_key="yt_test")
    [serialized] = model._get_request_payload([msg], stop=None)["messages"]
    assert serialized == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": [{"type": "text", "text": "Clicked\nCurrent URL: https://google.com/"}],
    }
