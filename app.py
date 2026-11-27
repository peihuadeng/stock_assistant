import time
from typing import Any, Dict, List

import gradio as gr
from gradio.components.chatbot import ChatMessage
from langchain_core.prompts import HumanMessagePromptTemplate

from graph.graph_tool import init_app
from prompt_list import human_prompt

app = init_app()
config = {"configurable": {"thread_id": "1"}}  # 暂不支持多用户
thinking_message = "正在思考..."


async def process_message(user_input: str, history: List[Dict[str, Any]]):
    human_message = HumanMessagePromptTemplate.from_template(human_prompt).format(message=user_input)
    # 用于存放本轮对话中产生的多条消息（工具调用面板 + 最终回复）
    message_list: List[ChatMessage] = []
    # 记录运行中的大模型的思考开始时间和对应的ChatMessage对象：key = run_id, value = (start_time, ChatMessage)
    active_llm_runs: dict[str, tuple[(float, ChatMessage)]] = {}
    # 记录每个工具调用的开始时间和对应的ChatMessage对象：key = run_id, value = (start_time, ChatMessage)
    tool_tracker: dict[str, tuple[float, ChatMessage]] = {}
    async for event in app.astream_events({"messages": [human_message]}, config=config, version='v2'):
        kind = event["event"]
        run_id = event["run_id"]
        name = event["name"]
        data = event.get("data", {})
        metadata = event.get("metadata", {})

        # 大模型事件
        if kind == "on_chat_model_start":
            message = ChatMessage(content=thinking_message)
            active_llm_runs[run_id] = (time.time(), message)
            message_list.append(message)
            yield message_list
        if kind == "on_chat_model_stream" and data['chunk'].content:
            if run_id not in active_llm_runs:
                continue
            _, message = active_llm_runs[run_id]
            if message.content == thinking_message:
                message.content = data["chunk"].content
            else:
                message.content += data["chunk"].content
            yield message_list
        if kind == "on_chat_model_end":
            if run_id not in active_llm_runs:
                continue
            start_time, message = active_llm_runs[run_id]
            time_consuming = round(time.time() - start_time, 3)
            thinking_time = f"[思考耗时：{time_consuming}s]"
            if message.content == thinking_message:
                message.content = thinking_time
            else:
                message.content += thinking_time
            yield message_list
        # 工具调用事件
        elif kind == "on_tool_start":
            tool_msg = ChatMessage(content=f"**入参:**\n```json\n{data['input']}\n```",
                                   metadata={"id": run_id,
                                             "title": f"⏳ 正在运行：{name}",
                                             "status": "pending"
                                             })
            # 如果当前工具是由另一个正在运行的工具触发的，自动建立嵌套子折叠面板
            for pid in reversed(event.get("parent_id", [])):
                if pid in tool_tracker:
                    tool_msg.metadata["parent_id"] = pid
                    break
            message_list.append(tool_msg)
            tool_tracker[run_id] = (time.time(), tool_msg)
            yield message_list
        elif kind == "on_tool_end":
            if run_id not in tool_tracker:
                continue
            start_time, tool_msg = tool_tracker[run_id]
            time_consuming = round(time.time() - start_time, 3)
            tool_msg.metadata["title"] = f"✅ 已完成：{name}"
            tool_msg.metadata["duration"] = time_consuming
            tool_msg.metadata["status"] = "done"
            tool_msg.content += f"\n\n**返回结果:**\n```text\n{data['output'].content}\n```"
            yield message_list
        elif kind == "on_custom_event" and name == "tool_progress":
            if run_id not in tool_tracker:
                continue
            start_time, tool_msg = tool_tracker[run_id]
            time_consuming = round(time.time() - start_time, 3)
            tool_msg.metadata["title"] = f"⏳ 正在运行({data['progress']:.0%})：{name}"
            tool_msg.metadata["duration"] = time_consuming
            yield message_list


if __name__ == "__main__":
    gr.ChatInterface(title="AI股票分析助手",
                     fn=process_message).launch()
