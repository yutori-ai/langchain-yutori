from openai.types.chat import ChatCompletion

from langchain_yutori import ChatYutoriN1


def test_chat_model_exposes_request_id_in_response_metadata():
    model = ChatYutoriN1(api_key="yt_test")
    response = ChatCompletion.model_validate(
        {
            "id": "cmpl_123",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                }
            ],
            "created": 0,
            "model": "n1-latest",
            "object": "chat.completion",
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
            "request_id": "req_123",
        }
    )

    result = model._create_chat_result(response)
    message = result.generations[0].message

    assert result.llm_output["request_id"] == "req_123"
    assert message.response_metadata["request_id"] == "req_123"
    assert message.additional_kwargs["request_id"] == "req_123"
