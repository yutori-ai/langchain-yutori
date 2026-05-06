from langchain_yutori import ChatYutoriNavigator


def test_tool_set_and_disable_tools_are_promoted_to_extra_body():
    model = ChatYutoriNavigator(
        api_key="yt_test",
        tool_set="browser_tools_expanded-20260403",
        disable_tools=["hold_key", "drag"],
    )
    payload = model._get_request_payload([], stop=None)
    assert payload["extra_body"] == {
        "tool_set": "browser_tools_expanded-20260403",
        "disable_tools": ["hold_key", "drag"],
    }


def test_first_class_params_merge_with_user_supplied_extra_body():
    # Lets users still pass other Navigator-specific fields (e.g. json_schema)
    # via extra_body without losing their first-class params.
    model = ChatYutoriNavigator(
        api_key="yt_test",
        tool_set="browser_tools_core-20260403",
        extra_body={"json_schema": {"type": "object"}},
    )
    payload = model._get_request_payload([], stop=None)
    assert payload["extra_body"] == {
        "tool_set": "browser_tools_core-20260403",
        "json_schema": {"type": "object"},
    }


def test_no_extra_body_when_no_navigator_options_passed():
    model = ChatYutoriNavigator(api_key="yt_test")
    payload = model._get_request_payload([], stop=None)
    assert "extra_body" not in payload
