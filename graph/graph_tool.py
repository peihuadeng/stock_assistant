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


def create_agent(system_prompt: None | SystemMessage, tool_list: None | list[BaseTool], is_async=True) -> StateGraph:
    if is_async:
        from llm.async_llm import AsyncLLM as LLM
    else:
        from llm.sync_llm import SyncLLM as LLM
    llm = LLM(model=os.getenv("MODEL_NAME"), temperature=0, system_prompt=system_prompt, tool_list=tool_list)
    tool_executor = ToolNode(tool_list)

    graph = StateGraph(CommonState)
    graph.add_node("llm", llm)
    graph.add_node("tools", tool_executor)

    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", tools_condition)
    graph.add_edge("tools", "llm")

    return graph


def init_app(is_async=True):
    from tools import search
    tool_list = [search]

    from prompt_list import system_prompt
    graph = create_agent(system_prompt=SystemMessage(content=system_prompt), tool_list=tool_list, is_async=is_async)
    memory = InMemorySaver()
    return graph.compile(checkpointer=memory)


# 测试代码
if __name__ == "__main__":
    import asyncio
    from langchain_core.messages import SystemMessage, AIMessageChunk, ToolMessage
    from langchain_core.prompts import HumanMessagePromptTemplate
    from prompt_list import human_prompt

    # 输出可视化图
    app = init_app(False)
    os.makedirs("img", exist_ok=True)
    app.get_graph().draw_png(output_file_path="img/graph_tool.png")

    content = "你好！帮我联网检索以下关键字：人工智能"
    human_message = HumanMessagePromptTemplate.from_template(human_prompt).format(message=content)

    config = {"configurable": {"thread_id": "1"}}

    # 同步流式输出模式，messages：实时输出模型文本；updates：节点更新；custom：自定义流输出
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

    # 异步流式输出模式，基于事件模型
    app = init_app(True)


    async def async_main():
        async for event in app.astream_events({"messages": [human_message]}, config=config, version='v2'):
            kind = event["event"]
            name = event["name"]
            data = event["data"]
            metadata = event["metadata"]
            # print(kind, name, data, metadata)
            # 大模型事件
            if kind == "on_chat_model_start":
                print("*" * 100, flush=True)
                for message in data["input"]["messages"][0]:
                    print(f"{message.__class__.__name__}：\n{message.content}", flush=True)
                print(f"{metadata.get('langgraph_node')}>>>", flush=True)
            if kind == "on_chat_model_stream" and data['chunk'].content:
                print(f"{data['chunk'].content}", end="", flush=True)
            if kind == "on_chat_model_end":
                if data["output"].tool_calls:
                    print("请求调用以下工具：")
                    for tool in data["output"].tool_calls:
                        print(f"工具：{tool['name']}，入参：{tool['args']}", flush=True)
                print("", flush=True)
            # 工具调用事件
            elif kind == "on_tool_start":
                print("*" * 100, flush=True)
                print(f"{metadata.get('langgraph_node')}>>>", flush=True)
                print(f"工具调用开始，工具名：{name}，入参：{data['input']}", flush=True)
            elif kind == "on_tool_end":
                print(f"工具调用结束，工具名：{name}，入参：{data['output'].content}", flush=True)
            # 自定义事件
            if kind == "on_custom_event":
                print(f"自定义事件：{data}", flush=True)


    asyncio.run(async_main())
