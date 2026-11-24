from dotenv import load_dotenv

load_dotenv()

import os
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool

from state.common_state import CommonState


def create_agent(system_prompt: None | SystemMessage, tool_list: None | list[BaseTool]):
    from llm.sync_llm import SyncLLM
    llm = SyncLLM(model=os.getenv("MODEL_NAME"), temperature=0, system_prompt=system_prompt, tool_list=tool_list)
    tool_executor = ToolNode(tool_list)

    graph = StateGraph(CommonState)
    graph.add_node("llm", llm)
    graph.add_node("tools", tool_executor)

    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", tools_condition)
    graph.add_edge("tools", "llm")

    return graph


def init_app():
    from tools import search
    tool_list = [search]

    from prompt_list import system_prompt
    graph = create_agent(system_prompt=SystemMessage(content=system_prompt), tool_list=tool_list)
    memory = InMemorySaver()
    return graph.compile(checkpointer=memory)


# 测试代码
if __name__ == "__main__":
    app = init_app()
    os.makedirs("img", exist_ok=True)
    app.get_graph().draw_png(output_file_path="img/graph_tool.png")

    from prompt_list import human_prompt

    content = "你好！帮我联网检索以下关键字：人工智能"
    human_message = HumanMessagePromptTemplate.from_template(human_prompt).format(message=content)

    config = {"configurable": {"thread_id": "1"}}
    response = app.invoke({"messages": [human_message]}, config=config)
    for message in response["messages"]:
        print("*" * 50)
        print(f"{message.__class__.__name__}:\n{message.content}")
