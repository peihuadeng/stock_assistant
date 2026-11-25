from typing import Annotated

from langchain_core.tools import tool
from langgraph.config import get_stream_writer


@tool
def search(key: Annotated[str, "检索关键字"]) -> str:
    """联网检索工具：依据检索关键字，联网检索，并把检索结果返回"""
    writer = get_stream_writer()
    writer({"task": "search", "state": "start", "args": key})
    result = "人工智能（Artificial Intelligence，简称 AI）是计算机科学的一个重要分支"
    writer({"task": "search", "state": "end", "result": result})
    return result


if __name__ == "__main__":
    print(search.args_schema.model_json_schema())
    result = search.invoke("人工智能")
    print(result)
