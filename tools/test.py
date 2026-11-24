from typing import Annotated

from langchain_core.tools import tool


@tool
def search(key: Annotated[str, "检索关键字"]) -> str:
    """联网检索工具：依据检索关键字，联网检索，并把检索结果返回"""
    return "人工智能（Artificial Intelligence，简称 AI）是计算机科学的一个重要分支"


if __name__ == "__main__":
    print(search.args_schema.model_json_schema())
    result = search.invoke("人工智能")
    print(result)
