import json
import os

import pytest
from langchain_core.messages import HumanMessage
from yutori.exceptions import APIError

from langchain_yutori import ChatYutoriN1, YutoriBrowsingTool, YutoriResearchTool, YutoriScoutingTool


pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def check_api_key():
    if not os.environ.get("YUTORI_API_KEY"):
        pytest.skip("YUTORI_API_KEY not set")


def test_chat_model_smoke():
    message = ChatYutoriN1().invoke("Say hello in one word.")
    assert message.content


def test_chat_model_image_action_smoke():
    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": "https://image.thum.io/get/width/1280/noanimate/https://www.google.com"},
            },
            {
                "type": "text",
                "text": (
                    "You are controlling a browser. The user goal is: search for Yutori on Google. "
                    "Based on this screenshot, what single next browser action should you take?"
                ),
            },
        ]
    )
    response = ChatYutoriN1().invoke([message])
    assert response.tool_calls
    assert response.tool_calls[0]["name"] in {
        "left_click",
        "double_click",
        "triple_click",
        "right_click",
        "scroll",
        "type",
        "key_press",
        "hover",
        "drag",
        "wait",
        "goto_url",
        "go_back",
        "refresh",
    }


def test_browsing_tool_smoke():
    result = YutoriBrowsingTool(timeout=600).invoke(
        {"task": "What is the page title?", "start_url": "https://yutori.com"}
    )
    data = json.loads(result)
    assert data["status"] == "succeeded"


def test_research_tool_smoke():
    result = YutoriResearchTool(timeout=900).invoke({"query": "What does Yutori do?"})
    data = json.loads(result)
    assert data["status"] == "succeeded"


def test_scouting_tool_lifecycle_smoke():
    tool = YutoriScoutingTool()
    create_result = json.loads(
        tool.invoke(
            {
                "action": "create",
                "query": "Monitor Hacker News for mentions of Yutori",
                "output_interval": 1800,
            }
        )
    )
    scout_id = create_result["id"]

    try:
        fetched = json.loads(tool.invoke({"action": "get", "scout_id": scout_id}))
        assert fetched["id"] == scout_id

        tool.invoke({"action": "pause", "scout_id": scout_id})
        paused = json.loads(tool.invoke({"action": "get", "scout_id": scout_id}))
        assert paused["status"] == "paused"

        tool.invoke({"action": "resume", "scout_id": scout_id})
        resumed = json.loads(tool.invoke({"action": "get", "scout_id": scout_id}))
        assert resumed["status"] == "active"
    finally:
        deleted = json.loads(tool.invoke({"action": "delete", "scout_id": scout_id}))
        assert deleted["deleted"] is True
        with pytest.raises(APIError):
            tool.invoke({"action": "get", "scout_id": scout_id})
