from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv
from functools import partial
from typing import Optional
import os
import logging
import requests
import json
from typing import Annotated
from datetime import datetime
import httpx

from .helpers import reassemble_fragments

load_dotenv()

from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode, InjectedState


logger = logging.getLogger(__name__)

# Warning: Brittle - None type will break this when it's injected into the state for the tool call, and it silently fails. So if it doesn't map state types properly from the frontend, it will break. (must be exactly what's defined here).
class LlamaPressState(MessagesState):
    api_token: str
    agent_prompt: str
    page_id: str
    current_page_html: str
    selected_element: Optional[str]
    javascript_console_errors: Optional[str]
    created_at: Optional[datetime] = datetime.now()

# Node
def router_node(state: LlamaPressState):
    last_message = state.get("messages")[-1]
    if "deep clone" in last_message.content.lower():
        return {"next": "url_clone_agent"}
    elif "clone" in last_message.content.lower():
        return {"next": "image_clone_agent"}
    else:
        return {"next": "html_agent"}

# Node
def url_clone_agent(state: LlamaPressState):
    instructions = state.get("agent_prompt", "")
    system_content = (
        f"Simply respond with, I am the url clone agent!"
    )

    model = ChatOpenAI(model="gpt-4.1-2025-04-14")
    llm_with_tools = model.bind_tools([])
    llm_response_message = llm_with_tools.invoke([SystemMessage(content=system_content)] + state["messages"])
    llm_response_message.response_metadata["created_at"] = str(datetime.now())

    return {"messages": [llm_response_message]}

# Node
def image_clone_agent(state: LlamaPressState):
    # instructions = state.get("agent_prompt", "")
    system_content = (
        f"Simply respond with, I am the image clone agent!"
    )

    model = ChatOpenAI(model="gpt-4.1-2025-04-14")
    llm_with_tools = model.bind_tools([])
    llm_response_message = llm_with_tools.invoke([SystemMessage(content=system_content)] + state["messages"])
    llm_response_message.response_metadata["created_at"] = str(datetime.now())

    return {"messages": [llm_response_message]}

# Global tools list
tools = []

def build_workflow(checkpointer=None):
    # Graph
    builder = StateGraph(LlamaPressState)

    # Define nodes: these do the work
    builder.add_node("router", router_node)
    builder.add_node("url_clone_agent", url_clone_agent)
    builder.add_node("image_clone_agent", image_clone_agent)
    builder.add_node("tools", ToolNode(tools))

    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "router")

    # Router condition
    builder.add_conditional_edges(
        "router",
         lambda x: x["next"], 
         {
             "url_clone_agent": "url_clone_agent",
             "image_clone_agent": "image_clone_agent",
         }
    )

    builder.add_conditional_edges(
        "url_clone_agent",
        # If the latest message (result) from software_developer_assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from software_developer_assistant is a not a tool call -> tools_condition routes to END
        tools_condition,
    )

    builder.add_conditional_edges(
        "image_clone_agent",
        # If the latest message (result) from software_developer_assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from software_developer_assistant is a not a tool call -> tools_condition routes to END
        tools_condition,
    )

    builder.add_edge("tools", "url_clone_agent")
    builder.add_edge("tools", "image_clone_agent")

    builder.add_edge("url_clone_agent", END)
    builder.add_edge("image_clone_agent", END)

    clone_agent = builder.compile(checkpointer=checkpointer, name="clone_agent")

    return clone_agent