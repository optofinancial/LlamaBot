from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv
from functools import partial
load_dotenv()

from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode, InjectedState

import requests
import json
from typing import Annotated

# Warning: Brittle - None type will break this when it's injected into the state for the tool call, and it silently fails. So if it doesn't map state types properly from the frontend, it will break. (must be exactly what's defined here).
class LlamaPressState(MessagesState): 
    api_token: str
    agent_instructions: str

# Tools

# Global tools list
# tools = [run_rails_console_command]
tools = []

# Node
def llamapress(state: LlamaPressState):
   
   additional_instructions = state.get("agent_instructions")

   # System message
   sys_msg = SystemMessage(content=f"""You are LlamaPress, a helpful AI assistant.
                        In normal chat conversations, feel free to implement markdown formatting to make your responses more readable, if it's appropriate.
                        Here are additional instructions provided by the user: <USER_INSTRUCTIONS> {additional_instructions} </USER_INSTRUCTIONS>""")

   llm = ChatOpenAI(model="gpt-4o-mini")
   llm_with_tools = llm.bind_tools(tools)
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

def build_workflow(checkpointer=None):
    # Graph
    builder = StateGraph(LlamaPressState)

    # Define nodes: these do the work
    builder.add_node("llamapress", llamapress)
    builder.add_node("tools", ToolNode(tools))

    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "llamapress")
    builder.add_conditional_edges(
        "llamapress",
        # If the latest message (result) from llamapress is a tool call -> tools_condition routes to tools
        # If the latest message (result) from llamapress is a not a tool call -> tools_condition routes to END
        tools_condition,
    )
    builder.add_edge("tools", "llamapress")

    react_graph = builder.compile(checkpointer=checkpointer)

    return react_graph