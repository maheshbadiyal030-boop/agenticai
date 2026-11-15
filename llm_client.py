from __future__ import annotations

from typing import Any

from config import OPENAI_MODEL


def get_chat_model(api_key: str) -> Any | None:
    if not api_key:
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=OPENAI_MODEL, api_key=api_key, temperature=0.3)
