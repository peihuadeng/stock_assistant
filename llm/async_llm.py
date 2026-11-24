import asyncio
import os

from langchain_core.messages import HumanMessage

from llm.common_llm import Agent
from state.common_state import CommonState


# 异步调用大模型
class AsyncLLM(Agent):
    async def __call__(self, state: CommonState):
        prompt = self._build_prompt(state)
        response = await self.llm.ainvoke(prompt)
        print(f"提交给大模型如下提示词：\n{prompt}\n大模型回答如下：\n{response}")
        return {"messages": [response]}


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    # 无工具调用
    llm = AsyncLLM(model=os.getenv("MODEL_NAME"), temperature=0, system_prompt=None, tool_list=None)
    state = CommonState(messages=[HumanMessage(content="你好")])
    result = asyncio.run(llm(state))
    print(result)
    print("*" * 200)
    # 带工具调用
    from tools import *

    tool_list = [search]
    llm = AsyncLLM(model=os.getenv("MODEL_NAME"), temperature=0, system_prompt=None, tool_list=tool_list)
    state = CommonState(messages=[HumanMessage(content="你好！帮我联网检索以下关键字：人工智能")])
    result = asyncio.run(llm(state))
    print(result)
