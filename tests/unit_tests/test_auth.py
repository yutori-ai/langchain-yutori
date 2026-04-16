import pytest

from langchain_yutori import ChatYutoriN1, YutoriBrowsingTool, YutoriResearchTool, YutoriScoutingTool


def test_chat_model_uses_sdk_auth_resolution(monkeypatch):
    monkeypatch.setattr("langchain_yutori.chat_models.resolve_api_key", lambda _: "yt_from_config")
    model = ChatYutoriN1()
    assert model.openai_api_key.get_secret_value() == "yt_from_config"


def test_tools_use_sdk_auth_resolution(monkeypatch):
    monkeypatch.setattr("langchain_yutori.tools.resolve_api_key", lambda _: "yt_from_config")

    browsing = YutoriBrowsingTool()
    research = YutoriResearchTool()
    scouting = YutoriScoutingTool()

    assert browsing.api_key == "yt_from_config"
    assert research.api_key == "yt_from_config"
    assert scouting.api_key == "yt_from_config"


def test_async_tools_enforce_minimum_poll_interval(monkeypatch):
    monkeypatch.setattr("langchain_yutori.tools.resolve_api_key", lambda _: "yt_from_config")

    with pytest.raises(ValueError, match="poll_interval must be at least 60 seconds"):
        YutoriBrowsingTool(poll_interval=30)

    with pytest.raises(ValueError, match="poll_interval must be at least 60 seconds"):
        YutoriResearchTool(poll_interval=30)
