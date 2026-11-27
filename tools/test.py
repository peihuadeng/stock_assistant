import asyncio
import time
from typing import Annotated

from langchain_core.callbacks import adispatch_custom_event
from langchain_core.tools import tool


@tool
def search(key: Annotated[str, "检索关键字"]) -> str:
    """联网检索工具：依据检索关键字，联网检索，并把检索结果返回"""
    start_event_data = {"task": "search", "progress": 0.3}
    asyncio.run(adispatch_custom_event("tool_progress", start_event_data))
    time.sleep(1)
    start_event_data = {"task": "search", "progress": 0.6}
    asyncio.run(adispatch_custom_event("tool_progress", start_event_data))
    time.sleep(1)
    result = "人工智能（Artificial Intelligence，简称 AI）是计算机科学的一个重要分支"
    start_event_data = {"task": "search", "progress": 0.9}
    asyncio.run(adispatch_custom_event("tool_progress", start_event_data))
    time.sleep(1)
    return result


if __name__ == "__main__":
    print(search.args_schema.model_json_schema())
    result = search.invoke("人工智能")
    print(result)
