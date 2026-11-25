from dotenv import load_dotenv

load_dotenv()

import os
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
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
    from langchain_core.messages import AIMessageChunk, ToolMessage
    from langchain_core.prompts import HumanMessagePromptTemplate
    from prompt_list import human_prompt

    app = init_app()
    os.makedirs("img", exist_ok=True)
    app.get_graph().draw_png(output_file_path="img/graph_tool.png")

    content = "你好！帮我联网检索以下关键字：人工智能"
    human_message = HumanMessagePromptTemplate.from_template(human_prompt).format(message=content)

    config = {"configurable": {"thread_id": "1"}}
    # for event in app.stream_events({"messages": [human_message]}, config=config, version='v3'):
    #     print("**")
    #     print(event)
    # 流式输出模式：stream_mode="messages"。实时输出模型文本
    current_node = None
    for event in app.stream({"messages": [human_message]}, config=config,
                            stream_mode=["messages", "updates", "custom"], version='v2'):
        event_type = event["type"]
        event_data = event["data"]
        if event_type == "messages":
            chunk, metadata = event_data
            if current_node != metadata.get("langgraph_node"):
                current_node = metadata.get("langgraph_node")
                print("*" * 100, flush=True)
                print(f"{current_node}>>> \n", end="", flush=True)
            if isinstance(chunk, AIMessageChunk):
                if chunk.tool_calls is not None and len(chunk.tool_calls) > 0:
                    for tool_call in chunk.tool_calls:
                        if tool_call['name']:
                            print(f"工具名：{tool_call['name']}", flush=True)
                        if tool_call["args"]:  # 有缺陷，有时候不打印args。完整的工具调用信息，请使用updates模式
                            print(f"工具入参：{tool_call['args']}", flush=True)
                if chunk.content:
                    print(chunk.content, end="", flush=True)
            if isinstance(chunk, ToolMessage):
                print(f"工具名：{chunk.name}\n状态：{chunk.status}\n返回值：{chunk.content}", flush=True)
        elif event_type == "updates":
            for node, update in event_data.items():
                print("\n>>>节点：", node)
                if not isinstance(update, dict):
                    continue
                for msg in update.get("messages", []):
                    if msg.type == "ai" and msg.tool_calls:
                        print("工具请求：", [(tool["name"], tool["args"]) for tool in msg.tool_calls])
                    elif msg.type == "tool":
                        print("工具结果：", msg.name, msg.content)
                    elif msg.type == "ai" and msg.text:
                        print("AI回答：", msg.text)
        elif event_type == "custom":
            print(event_data, flush=True)
