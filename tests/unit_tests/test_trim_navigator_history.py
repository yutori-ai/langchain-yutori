"""Thorough tests for trim_navigator_history.

Covers: default keep_recent value, boundary cases (history shorter/equal/longer
than window, keep_recent=0), preservation invariants (no mutation, text kept,
tool_calls kept, role preserved), and end-to-end serialization back to the
Navigator wire shape after trimming.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_yutori import ChatYutoriNavigator, trim_navigator_history


def _user_with_image(text: str, screenshot: str = "data:image/webp;base64,USER") -> HumanMessage:
    return HumanMessage(
        content=[
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": screenshot}},
        ]
    )


def _tool_with_image(call_id: str, text: str, screenshot: str = "data:image/webp;base64,TOOL") -> ToolMessage:
    return ToolMessage(
        tool_call_id=call_id,
        content=[
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": screenshot}},
        ],
    )


def _has_image(msg) -> bool:
    return isinstance(msg.content, list) and any(
        isinstance(b, dict) and b.get("type") == "image_url" for b in msg.content
    )


def _has_text(msg, text: str) -> bool:
    return isinstance(msg.content, list) and any(
        isinstance(b, dict) and b.get("type") == "text" and b.get("text") == text for b in msg.content
    )


# --- default behavior ----------------------------------------------------------


def test_default_keep_recent_is_2():
    history = [_user_with_image(f"turn {i}") for i in range(5)]
    out = trim_navigator_history(history)
    # first 3 lose images
    for i in range(3):
        assert not _has_image(out[i]), f"msg {i} should have no images"
        assert _has_text(out[i], f"turn {i}"), f"msg {i} should still have its text"
    # last 2 keep images
    for i in (3, 4):
        assert _has_image(out[i]), f"msg {i} should keep images"
        assert _has_text(out[i], f"turn {i}")


def test_explicit_keep_recent_overrides_default():
    history = [_user_with_image(f"t{i}") for i in range(4)]
    out = trim_navigator_history(history, keep_recent=1)
    assert not _has_image(out[0])
    assert not _has_image(out[1])
    assert not _has_image(out[2])
    assert _has_image(out[3])


# --- boundary cases ------------------------------------------------------------


def test_history_shorter_than_keep_recent_returned_as_is():
    history = [_user_with_image("only")]
    out = trim_navigator_history(history, keep_recent=5)
    assert out == history
    assert out[0] is history[0]  # same instance — no copy made


def test_history_equal_to_keep_recent_returned_as_is():
    history = [_user_with_image("a"), _user_with_image("b")]
    out = trim_navigator_history(history, keep_recent=2)
    assert out[0] is history[0]
    assert out[1] is history[1]


def test_keep_recent_zero_strips_all_images():
    history = [_user_with_image("a"), _user_with_image("b"), _user_with_image("c")]
    out = trim_navigator_history(history, keep_recent=0)
    assert all(not _has_image(m) for m in out)
    # text preserved everywhere
    for i, msg in enumerate(out):
        assert _has_text(msg, "abc"[i])


def test_empty_history_returns_empty_list():
    assert trim_navigator_history([]) == []
    assert trim_navigator_history([], keep_recent=10) == []


def test_negative_keep_recent_raises():
    with pytest.raises(ValueError, match="keep_recent must be >= 0"):
        trim_navigator_history([_user_with_image("a")], keep_recent=-1)


# --- non-mutation guarantees --------------------------------------------------


def test_input_list_not_mutated():
    history = [_user_with_image(f"t{i}") for i in range(3)]
    snapshot = list(history)
    snapshot_first_content = history[0].content

    trim_navigator_history(history, keep_recent=1)

    assert history == snapshot, "outer list should not be mutated"
    assert history[0].content is snapshot_first_content, "original message content not mutated"
    assert _has_image(history[0]), "original message still has its image"


def test_unchanged_messages_returned_by_reference_for_perf():
    """Messages that didn't need trimming should not be needlessly copied."""
    untrimmable = HumanMessage(content=[{"type": "text", "text": "no image here"}])
    recent = _user_with_image("recent")
    history = [untrimmable, recent]

    out = trim_navigator_history(history, keep_recent=1)
    assert out[0] is untrimmable, "text-only content list shouldn't be copied"
    assert out[1] is recent, "in-window message shouldn't be copied"


def test_trimmed_messages_are_fresh_copies_not_aliases():
    history = [_user_with_image("old"), _user_with_image("recent")]
    out = trim_navigator_history(history, keep_recent=1)
    assert out[0] is not history[0], "trimmed msg should be a new instance"
    # original is untouched
    assert _has_image(history[0])
    # copy lost the image
    assert not _has_image(out[0])


# --- mixed message types ------------------------------------------------------


def test_string_content_messages_pass_through():
    """LangChain messages whose content is a plain string aren't multimodal — leave alone."""
    history = [
        AIMessage(content="just thinking"),
        ToolMessage(content="just text", tool_call_id="x"),
        HumanMessage(content="ask"),
        _user_with_image("recent"),
    ]
    out = trim_navigator_history(history, keep_recent=1)
    for orig, trimmed in zip(history, out):
        assert orig is trimmed


def test_aimessage_tool_calls_preserved_under_trim():
    """tool_calls live on the message field, not in content — must survive trimming."""
    ai_msg = AIMessage(
        content="",  # AIMessage often has empty content when tool_calls fire
        tool_calls=[{"id": "c1", "name": "left_click", "args": {"coordinate": [500, 250]}}],
    )
    history = [ai_msg] + [_user_with_image(f"t{i}") for i in range(3)]
    out = trim_navigator_history(history, keep_recent=2)
    # AIMessage is past the cutoff (not in last 2) but has string content → passed through by reference.
    assert out[0] is ai_msg
    # tool_calls should round-trip with the structurally relevant fields intact.
    [tc] = out[0].tool_calls
    assert tc["id"] == "c1"
    assert tc["name"] == "left_click"
    assert tc["args"] == {"coordinate": [500, 250]}


def test_aimessage_tool_calls_preserved_when_msg_is_in_trim_window():
    """If the AIMessage with tool_calls falls inside the cutoff and has list-shaped
    content, the trim path runs over it. tool_calls (a separate field) must survive
    the model_copy."""
    ai_msg = AIMessage(
        content=[
            {"type": "text", "text": "thinking..."},
            {"type": "image_url", "image_url": {"url": "data:image/webp;base64,X"}},
        ],
        tool_calls=[{"id": "c2", "name": "scroll", "args": {"direction": "down", "amount": 3}}],
    )
    history = [ai_msg] + [_user_with_image(f"t{i}") for i in range(3)]
    out = trim_navigator_history(history, keep_recent=1)
    # ai_msg is past the cutoff: content list got trimmed → fresh instance.
    assert out[0] is not ai_msg
    assert not _has_image(out[0])
    assert _has_text(out[0], "thinking...")
    # tool_calls survive the model_copy.
    [tc] = out[0].tool_calls
    assert tc["id"] == "c2"
    assert tc["name"] == "scroll"
    assert tc["args"] == {"direction": "down", "amount": 3}


def test_tool_message_image_dropped_text_and_tool_call_id_preserved():
    tool_msg = _tool_with_image("call_1", "Clicked\nCurrent URL: https://example.com/")
    history = [tool_msg, _user_with_image("recent_a"), _user_with_image("recent_b")]
    out = trim_navigator_history(history, keep_recent=2)
    trimmed = out[0]
    # text survives
    assert _has_text(trimmed, "Clicked\nCurrent URL: https://example.com/")
    # image is gone
    assert not _has_image(trimmed)
    # tool_call_id survives the model_copy
    assert trimmed.tool_call_id == "call_1"


def test_non_image_url_blocks_in_content_list_preserved():
    """Custom content blocks (anything other than image_url) should not be touched."""
    msg = HumanMessage(
        content=[
            {"type": "text", "text": "explain"},
            {"type": "tool_use", "id": "t1", "name": "x", "input": {}},  # arbitrary non-image block
            {"type": "image_url", "image_url": {"url": "data:image/webp;base64,A"}},
        ]
    )
    history = [msg, _user_with_image("r")]
    out = trim_navigator_history(history, keep_recent=1)
    [text_block, tool_use_block] = out[0].content
    assert text_block["type"] == "text"
    assert tool_use_block["type"] == "tool_use"


def test_string_blocks_inside_content_list_preserved():
    """LangChain occasionally emits raw strings inside a content list — should survive."""
    msg = HumanMessage(
        content=[
            "raw string block",  # not a dict
            {"type": "image_url", "image_url": {"url": "data:image/webp;base64,A"}},
        ]
    )
    history = [msg, _user_with_image("r")]
    out = trim_navigator_history(history, keep_recent=1)
    assert out[0].content == ["raw string block"]


# --- end-to-end: trimmed history still serializes to Navigator's wire shape ----


def test_trimmed_history_round_trips_to_navigator_payload():
    """After trimming, the wire payload must still match Navigator's documented shape:
    roles preserved, tool_call_id intact on tool messages, only the right images dropped.
    """
    history = [
        _user_with_image("turn 0", "data:image/webp;base64,A"),
        AIMessage(content="", tool_calls=[{"id": "c1", "name": "left_click", "args": {"coordinate": [500, 250]}}]),
        _tool_with_image("c1", "Clicked\nCurrent URL: https://google.com/", "data:image/webp;base64,B"),
        _user_with_image("turn 3", "data:image/webp;base64,C"),
        _tool_with_image("c1", "final\nCurrent URL: https://google.com/", "data:image/webp;base64,D"),
    ]
    trimmed = trim_navigator_history(history, keep_recent=2)

    model = ChatYutoriNavigator(api_key="yt_test")
    serialized = model._get_request_payload(trimmed, stop=None)["messages"]

    # roles still in order
    assert [m["role"] for m in serialized] == ["user", "assistant", "tool", "user", "tool"]

    # first three messages have no image_url
    for m in serialized[:3]:
        content = m.get("content")
        if isinstance(content, list):
            assert all(b.get("type") != "image_url" for b in content), f"unexpected image in {m['role']}"

    # last two keep their images
    assert any(b.get("type") == "image_url" for b in serialized[3]["content"])
    assert any(b.get("type") == "image_url" for b in serialized[4]["content"])

    # tool_call_id round-tripped on the trimmed tool message
    assert serialized[2]["tool_call_id"] == "c1"
    # text preserved on the trimmed tool message
    assert any(
        b.get("type") == "text" and "Clicked" in b.get("text", "")
        for b in serialized[2]["content"]
    )

    # assistant tool_calls round-tripped through the trim
    [tc] = serialized[1]["tool_calls"]
    assert tc["function"]["name"] == "left_click"
