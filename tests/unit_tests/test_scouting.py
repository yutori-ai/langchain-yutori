from types import SimpleNamespace

from langchain_yutori import YutoriScoutingTool


def test_global_updates_fallback_uses_24h_window(monkeypatch):
    monkeypatch.setattr("langchain_yutori.tools.resolve_api_key", lambda _: "yt_from_config")
    tool = YutoriScoutingTool()

    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"items": []}

    class FakeHttpClient:
        def get(self, url, headers, params):
            captured["url"] = url
            captured["headers"] = headers
            captured["params"] = params
            return FakeResponse()

    fake_client = SimpleNamespace(
        _client=FakeHttpClient(),
        _base_url="https://api.yutori.com/v1",
        _api_key="yt_from_config",
    )

    result = tool._get_global_updates(fake_client, limit=5)

    assert result == {"items": []}
    assert captured["url"] == "https://api.yutori.com/v1/scouting/updates"
    assert captured["headers"] == {"X-API-Key": "yt_from_config"}
    assert captured["params"]["page_size"] == 5
    assert "start_time" in captured["params"]
    assert "end_time" in captured["params"]
