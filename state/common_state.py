from typing import TypedDict, Annotated

from langgraph.graph import add_messages


# 工具调用图的状态
class CommonState(TypedDict):
    messages: Annotated[list, add_messages]
