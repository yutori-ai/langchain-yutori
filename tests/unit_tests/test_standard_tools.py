import pytest

from langchain_yutori import YutoriBrowsingTool, YutoriResearchTool, YutoriScoutingTool

try:
    from langchain_tests.unit_tests import ToolsUnitTests
except ImportError:
    pytest.skip("langchain-tests is not installed in this environment", allow_module_level=True)


class TestYutoriBrowsingToolStandard(ToolsUnitTests):
    @property
    def tool_constructor(self):
        return YutoriBrowsingTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"api_key": "yt_test"}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"task": "Read the page title", "start_url": "https://example.com"}

    @property
    def init_from_env_params(self):
        return ({"YUTORI_API_KEY": "yt_env"}, {}, {"api_key": "yt_env"})


class TestYutoriResearchToolStandard(ToolsUnitTests):
    @property
    def tool_constructor(self):
        return YutoriResearchTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"api_key": "yt_test"}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"query": "Summarize Yutori's homepage"}

    @property
    def init_from_env_params(self):
        return ({"YUTORI_API_KEY": "yt_env"}, {}, {"api_key": "yt_env"})


class TestYutoriScoutingToolStandard(ToolsUnitTests):
    @property
    def tool_constructor(self):
        return YutoriScoutingTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"api_key": "yt_test"}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"action": "list"}

    @property
    def init_from_env_params(self):
        return ({"YUTORI_API_KEY": "yt_env"}, {}, {"api_key": "yt_env"})
