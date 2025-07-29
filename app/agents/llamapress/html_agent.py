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

# Tools
@tool
async def write_html_page(
    full_html_document: str,
    message_to_user: str,
    internal_thoughts: str,
    state: Annotated[dict, InjectedState],
) -> str:
    """
    Write an HTML page to the filesystem.
    full_html_document is the full HTML document to write to the filesystem, including CSS and JavaScript.
    message_to_user is a string to tell the user what you're doing.
    internal_thoughts are your thoughts about the command.
    """
    # Debug logging
    logger.info(f"API TOKEN: {state.get('api_token')}")
    logger.info(f"Page ID: {state.get('page_id')}")
    logger.info(
        f"State keys: {list(state.keys()) if isinstance(state, dict) else 'Not a dict'}"
    )

    # Configuration
    LLAMAPRESS_API_URL = os.getenv("LLAMAPRESS_API_URL")

    logger.info(f"Writing HTML page to filesystem! to {LLAMAPRESS_API_URL}")

    # Get page_id from state, with fallback
    page_id = state.get("page_id")
    if not page_id:
        return "Error: page_id is required but not provided in state"

    API_ENDPOINT = f"{LLAMAPRESS_API_URL}/pages/{page_id}.json"
    try:
        # Get API token from state
        api_token = state.get("api_token")
        if not api_token:
            return "Error: api_token is required but not provided in state"

        async with httpx.AsyncClient() as client:
            response = await client.put(
                API_ENDPOINT,
                json={"content": full_html_document},
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"LlamaBot {api_token}",
                },
                timeout=30,  # 30 second timeout
            )

        # Parse the response
        if response.status_code == 200:
            data = response.json()
            return json.dumps(data, ensure_ascii=False, indent=2)
        else:
            return f"HTTP Error {response.status_code}: {response.text}"

    except httpx.ConnectError:
        return "Error: Could not connect to Rails server. Make sure your Rails app is running."

    except httpx.TimeoutException:
        return "Error: Request timed out. The Rails request may be taking too long to execute."

    except httpx.RequestError as e:
        return f"Request Error: {str(e)}"

    except json.JSONDecodeError:
        return f"Error: Invalid JSON response from server. Raw response: {response.text}"

    except Exception as e:
        return f"Unexpected Error: {str(e)}"

@tool
async def overwrite_html_snippet(new_html_code: str, state: LlamaPressState) -> str:
    """Writes new HTML code to the current page.

    This tool is used to modify an existing HTML element on the page.

    Args:
        new_html_code (str): The new HTML code to write. This will replace the selected element.
        state (LlamaBotState): The current state of the LlamaBot.

    Returns:
        str: A success or error message.
    """
    logger.info(f"HTML Snippet code written for the selected element: {new_html_code}")
    reassembled_html = reassemble_fragments(
        new_html_code, state.get("current_page_html")
    )
    api_token = state.get("api_token")
    LLAMAPRESS_API_URL = os.getenv("LLAMAPRESS_API_URL")
    POST_API_ENDPOINT = f"{LLAMAPRESS_API_URL}/pages/{state.get("page_id")}.json"

    async with httpx.AsyncClient() as client:
        response = await client.put(
            POST_API_ENDPOINT,
            json={"content": reassembled_html},
            headers={
                "Content-Type": "application/json",
                "Authorization": f"LlamaBot {api_token}",
            },
            timeout=30,  # 30 second timeout
        )

    if response.status_code == 200:
        data = response.json()
        return json.dumps(data, ensure_ascii=False, indent=2)
    else:
        return f"HTTP Error {response.status_code}: {response.text}"

# Node
def router_node(state: LlamaPressState):
    additional_instructions = state.get("agent_prompt")
    user_has_selected_element = state.get("selected_element") is not None
    if user_has_selected_element:
        next_node = "selected_element_agent"
    else:
        next_node = "write_html_page_agent"
    return {"next": next_node}

# Node
def selected_element_agent(state: LlamaPressState):
    instructions = state.get("agent_prompt", "")
    system_content = (
        f"You are given an HTML and Tailwind snippet of code to inspect. Here it is: {state.get('selected_element')}"
        "You are able to modify the HTML and Tailwind snippet of code, if the user asks you to by using the tool/function `overwrite_html_snippet`"
        ""
        "You are able to write the new HTML and Tailwind snippet of code to the filesystem, if the user asks you to."
    )

    model = ChatOpenAI(model="gpt-4.1-2025-04-14")
    llm_with_tools = model.bind_tools([overwrite_html_snippet])
    llm_response_message = llm_with_tools.invoke([SystemMessage(content=system_content)] + state["messages"])
    llm_response_message.response_metadata["created_at"] = str(datetime.now())

    return {"messages": [llm_response_message]}

# Node
def write_html_page_agent(state: LlamaPressState):
    # instructions = state.get("agent_prompt", "")
    system_content = (
        f"You are currently viewing an HTML Page and Tailwind CSS full page."
        "The user needs you to respond to their message. If the user so desires, you are able to modify the HTML and Tailwind snippet of code,"
        "if the user asks you to by using the tool/function `write_html_page`"
        "IF you using the `write_html_page` function/tool, and are generating HTML/CSS/JavaScript code, include comments that explain what you're doing for each logical block of code you're about to generate."
            "For every logical block you generate (HTML section, CSS rule set, JS function):"
            "1. Precede it with exactly **one** comment line that starts with 'CODE_EXPLANATION: <code_explanation> writing a section that ... </code_explanation>'"
            "2. Keep the code_explanation ≤ 15 words."
            "3. Never include other text on that line."
            "4. Examples of how to do this:"
                "EXAMPLE_HTML_COMMENT <!-- <code_explanation>Adding a section about the weather</code_explanation> -->"
                "EXAMPLE_TAILWIND_CSS_COMMENT /* <code_explanation>Setting the page background color to blue with Tailwind CSS</code_explanation> */"
                "EXAMPLE_JAVASCRIPT_COMMENT // <code_explanation>Making the weather section interactive and animated with JavaScript  </code_explanation>"
            "5. You are able to write the new HTML and Tailwind snippet of code to the filesystem, if the user asks you to."
                "Any HTML pages generated MUST include tailwind CDN and viewport meta helper tags in the header: "
                "<EXAMPLE> <head data-llama-editable='true' data-llama-id='0'>"
                "<meta content='width=device-width, initial-scale=1.0' name='viewport'>"
                "<script src='https://cdn.tailwindcss.com'></script> </EXAMPLE>"
        "You DONT HAVE to write HTML code, and in fact sometimes it is inappropriate depending on what the user is asking you to do."
        "You can also just respond and answer questions, or even ask clarifying questions, etc. Parse the user's intent and make a decision."
    )

    model = ChatOpenAI(model="gpt-4.1-2025-04-14")
    llm_with_tools = model.bind_tools([write_html_page])
    llm_response_message = llm_with_tools.invoke([SystemMessage(content=system_content)] + state["messages"] + [SystemMessage(content="<CURRENT_PAGE_HTML>" + state.get("current_page_html") + "</CURRENT_PAGE_HTML>")])
    llm_response_message.response_metadata["created_at"] = str(datetime.now())
    # breakpoint()

    return {"messages": [llm_response_message]}

# Global tools list
tools = [write_html_page, overwrite_html_snippet]

def build_workflow(checkpointer=None):
    # Graph
    builder = StateGraph(LlamaPressState)

    # Define nodes: these do the work
    builder.add_node("router", router_node)
    builder.add_node("selected_element_agent", selected_element_agent)
    builder.add_node("write_html_page_agent", write_html_page_agent)
    builder.add_node("tools", ToolNode(tools))

    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "router")

    # Router condition
    builder.add_conditional_edges(
        "router",
         lambda x: x["next"], 
         {
             "selected_element_agent": "selected_element_agent",
             "write_html_page_agent": "write_html_page_agent",
         }
    )

    builder.add_conditional_edges(
        "write_html_page_agent",
        # If the latest message (result) from software_developer_assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from software_developer_assistant is a not a tool call -> tools_condition routes to END
        tools_condition,
    )

    builder.add_conditional_edges(
        "selected_element_agent",
        # If the latest message (result) from software_developer_assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from software_developer_assistant is a not a tool call -> tools_condition routes to END
        tools_condition,
    )

    builder.add_edge("tools", "selected_element_agent")
    builder.add_edge("tools", "write_html_page_agent")

    builder.add_edge("selected_element_agent", END)
    builder.add_edge("write_html_page_agent", END)

    html_agent = builder.compile(checkpointer=checkpointer, name="html_agent")

    return html_agent