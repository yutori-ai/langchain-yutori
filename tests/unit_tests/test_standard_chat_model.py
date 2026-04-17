import pytest

from langchain_yutori import ChatYutoriN1

try:
    from langchain_tests.unit_tests import ChatModelUnitTests
except ImportError:
    pytest.skip("langchain-tests is not installed in this environment", allow_module_level=True)


class TestChatYutoriN1Standard(ChatModelUnitTests):
    @property
    def chat_model_class(self):
        return ChatYutoriN1

    @property
    def chat_model_params(self) -> dict:
        return {"api_key": "yt_test", "model": "n1-latest"}

    @property
    def init_from_env_params(self):
        return ({"YUTORI_API_KEY": "yt_env"}, {}, {"openai_api_key": "yt_env"})

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def supports_image_urls(self) -> bool:
        return True

    @property
    def has_tool_choice(self) -> bool:
        return False

    @property
    def has_structured_output(self) -> bool:
        return False
