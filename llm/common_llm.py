import os

from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from state.common_state import CommonState


class Agent:
    def __init__(self, model: str | None, temperature: float | None, system_prompt: SystemMessage | None,
                 tool_list: list[BaseTool] | None):
        if model is None or temperature is None:
            raise ValueError("model and temperature are required")

        self.llm = ChatOpenAI(model=model, temperature=temperature, streaming=True)
        print(
            f"成功连接大模型服务，model[{model}], temperature[{temperature}], base_url[{os.getenv('OPENAI_BASE_URL')}]")
        if tool_list is not None and len(tool_list) > 0:
            self.llm = self.llm.bind_tools(tool_list)
            print(f"大模型成功绑定以下工具：")
            for tool in tool_list:
                print(f"名称: {tool.name}, 描述：{tool.description}")
        self.system_prompt = system_prompt

    def _build_prompt(self, state: CommonState):
        return [self.system_prompt] + state["messages"] if self.system_prompt else state["messages"]

    def _parse_output(self, state: CommonState, response: AIMessage):
        pass

    def __call__(self, state: CommonState):
        pass
