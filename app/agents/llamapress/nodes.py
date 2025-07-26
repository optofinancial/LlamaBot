from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv
from functools import partial
from typing import Optional
import os
import logging

load_dotenv()

from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, AnyMessage

from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode, InjectedState

from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

import requests
import json
from typing import Annotated
from datetime import datetime

logger = logging.getLogger(__name__)

# Warning: Brittle - None type will break this when it's injected into the state for the tool call, and it silently fails. So if it doesn't map state types properly from the frontend, it will break. (must be exactly what's defined here).
class LlamaPressState(AgentState): 
    api_token: str
    agent_prompt: str
    page_id: str
    current_page_html: str
    selected_element: Optional[str]
    javascript_console_errors: Optional[str]
    created_at: Optional[datetime] = datetime.now()

# Tools
@tool
def write_html_page(full_html_document: str, message_to_user: str, internal_thoughts: str, state: Annotated[dict, InjectedState]) -> str:
   """
   Write an HTML page to the filesystem.
   full_html_document is the full HTML document to write to the filesystem, including CSS and JavaScript.
   message_to_user is a string to tell the user what you're doing.
   internal_thoughts are your thoughts about the command.
   """
   # Debug logging
   logger.info(f"API TOKEN: {state.get('api_token')}")
   logger.info(f"Page ID: {state.get('page_id')}")
   logger.info(f"State keys: {list(state.keys()) if isinstance(state, dict) else 'Not a dict'}")
   
   # Configuration
   LLAMAPRESS_API_URL = os.getenv("LLAMAPRESS_API_URL")

   # Get page_id from state, with fallback
   page_id = state.get('page_id')
   if not page_id:
       return "Error: page_id is required but not provided in state"
   
   API_ENDPOINT = f"{LLAMAPRESS_API_URL}/pages/{page_id}.json"
   try:
       # Get API token from state
       api_token = state.get('api_token')
       if not api_token:
           return "Error: api_token is required but not provided in state"
       
       print(f"API TOKEN: LlamaBot {api_token}")

       # Make HTTP request to Rails API
       response = requests.put(
           API_ENDPOINT,
           json={'content': full_html_document},
           headers={'Content-Type': 'application/json', 'Authorization': f'LlamaBot {api_token}'},
           timeout=30  # 30 second timeout
       )

       # Parse the response
       if response.status_code == 200:
           data = response.json()
           return json.dumps(data, ensure_ascii=False, indent=2)
       else:
           return f"HTTP Error {response.status_code}: {response.text}"

   except requests.exceptions.ConnectionError:
       return "Error: Could not connect to Rails server. Make sure your Rails app is running."

   except requests.exceptions.Timeout:
       return "Error: Request timed out. The Rails request may be taking too long to execute."

   except requests.exceptions.RequestException as e:
       return f"Request Error: {str(e)}"

   except json.JSONDecodeError:
       return f"Error: Invalid JSON response from server. Raw response: {response.text}"

   except Exception as e:
       return f"Unexpected Error: {str(e)}"

   print("Write to filesystem!")
   return "HTML page written to filesystem!"

# Global tools list
tools = [write_html_page]

model = ChatOpenAI(model="gpt-4o")

def system_prompt(state: LlamaPressState) -> list[AnyMessage]:
    # Use whatever state fields you need to build the system prompt
    instructions = state.get("agent_prompt", "")
    system_content = (
        "You are Leonardo the Llama, a helpful AI assistant. "
        "You live within LlamaPress, a web application that allows you "
        "to write full HTML pages with Tailwind CSS to the filesystem. "
        "Any HTML pages generated MUST include Tailwind CDN and viewport meta tags. "
        "Here is additional state context: {state}"
        "And here are additional system instructions provided by the user: {instructions}"
    )
    return [SystemMessage(content=system_content)] + state["messages"]

def write_html_prompt(state: LlamaPressState) -> list[AnyMessage]:
    instructions = state.get("agent_prompt", "")
    system_content = (
        "You are Leonardo the Llama, a helpful AI assistant. "
        "You live within LlamaPress, a web application that allows you "
        "to write full HTML pages with Tailwind CSS to the filesystem. "
        "Any HTML pages generated MUST include tailwind CDN and viewport meta helper tags in the header: "
        "<EXAMPLE> <head data-llama-editable='true' data-llama-id='0'>"
        "<meta content='width=device-width, initial-scale=1.0' name='viewport'>"
        "<script src='https://cdn.tailwindcss.com'></script> </EXAMPLE>"
        "Here is additional state context: <ADDITIONAL_STATE_AND_CONTEXT> {state} </ADDITIONAL_STATE_AND_CONTEXT>"
        "And here are additional system instructions provided by the user: <USER_INSTRUCTIONS> {instructions} </USER_INSTRUCTIONS>"
    )
    return [SystemMessage(content=system_content)] + state["messages"]

def build_workflow(checkpointer=None):
    write_html_page_agent = create_react_agent(
        model=model,
        tools=[write_html_page],
        name="write_html_page_agent",
        prompt=write_html_prompt,
        state_schema=LlamaPressState,
        checkpointer=checkpointer
    )

    # Create supervisor workflow
    workflow = create_supervisor(
        [write_html_page_agent],
        model=model,
        prompt=system_prompt,
        state_schema=LlamaPressState,
    )

    # Compile and run
    return workflow.compile(checkpointer=checkpointer)