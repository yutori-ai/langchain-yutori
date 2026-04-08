from __future__ import annotations

import argparse
import json

from langchain_core.messages import HumanMessage

from langchain_yutori import ChatYutoriN1, YutoriBrowsingTool, YutoriResearchTool, YutoriScoutingTool


def run_chat_smoke() -> None:
    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": "https://docs.yutori.com/assets/google_homepage_2024.jpg"},
            },
            {"type": "text", "text": "Briefly describe what page this is."},
        ]
    )
    response = ChatYutoriN1().invoke([message])
    print("chat:", response.content)


def run_browsing_smoke() -> None:
    result = YutoriBrowsingTool(timeout=600).invoke(
        {"task": "What is the page title?", "start_url": "https://yutori.com"}
    )
    print("browsing:", json.dumps(json.loads(result), indent=2)[:1000])


def run_research_smoke() -> None:
    result = YutoriResearchTool(timeout=900).invoke({"query": "What does Yutori do?"})
    print("research:", json.dumps(json.loads(result), indent=2)[:1000])


def run_scouting_smoke() -> None:
    tool = YutoriScoutingTool()
    created = json.loads(
        tool.invoke(
            {
                "action": "create",
                "query": "Monitor Hacker News for mentions of Yutori",
                "output_interval": 1800,
            }
        )
    )
    scout_id = created["id"]
    print("scout_create:", scout_id)

    try:
        print("scout_get:", tool.invoke({"action": "get", "scout_id": scout_id}))
        print("scout_pause:", tool.invoke({"action": "pause", "scout_id": scout_id}))
        print("scout_resume:", tool.invoke({"action": "resume", "scout_id": scout_id}))
        print("scout_updates:", tool.invoke({"action": "get_updates", "scout_id": scout_id, "limit": 5}))
    finally:
        print("scout_delete:", tool.invoke({"action": "delete", "scout_id": scout_id}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test langchain-yutori against real Yutori APIs.")
    parser.add_argument(
        "--scope",
        choices=["all", "chat", "browsing", "research", "scouting"],
        default="all",
        help="Which surface to exercise",
    )
    args = parser.parse_args()

    if args.scope in {"all", "chat"}:
        run_chat_smoke()
    if args.scope in {"all", "browsing"}:
        run_browsing_smoke()
    if args.scope in {"all", "research"}:
        run_research_smoke()
    if args.scope in {"all", "scouting"}:
        run_scouting_smoke()


if __name__ == "__main__":
    main()
