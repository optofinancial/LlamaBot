from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv
from functools import partial
import os
from typing import Optional

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
class LlamaBotState(MessagesState): 
    api_token: str
    agent_prompt: str
    available_routes: Optional[str] = None

@tool
def rails_https_request(route: Optional[str], method: Optional[str], params: Optional[dict], state: Annotated[dict, InjectedState]) -> str:
    """
    Make an HTTP request to the Rails server with robust error handling.
    Returns a JSON string with structured information about the request and response.
    """
    import traceback
    from urllib.parse import urljoin
    
    RAILS_SERVER_URL = os.getenv("LLAMAPRESS_API_URL")
    if not RAILS_SERVER_URL:
        return json.dumps({
            "success": False,
            "error": "LLAMAPRESS_API_URL environment variable not set",
            "error_type": "configuration_error"
        }, indent=2)
    
    # Build the API endpoint
    if route:
        API_ENDPOINT = urljoin(RAILS_SERVER_URL.rstrip('/') + '/', f"{route.lstrip('/')}.json")
    else:
        API_ENDPOINT = RAILS_SERVER_URL
    
    # Prepare request info for debugging
    request_info = {
        "method": method or "GET",
        "url": API_ENDPOINT,
        "has_params": bool(params),
        "params_preview": str(params)[:200] + "..." if params and len(str(params)) > 200 else params
    }
    
    try:
        # Make the HTTP request
        response = requests.request(
            method=method or "GET",
            url=API_ENDPOINT,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'LlamaBot {state.get("api_token")}',
                'Accept': 'application/json, text/html, text/plain, */*'
            },
            json=params if params and method and method.upper() in ['POST', 'PUT', 'PATCH'] else None,
            params=params if params and method and method.upper() == 'GET' else None,
            timeout=30,
            allow_redirects=True
        )
        
        # Get response metadata
        content_type = response.headers.get('content-type', '').lower()
        response_size = len(response.content)
        
        # Handle different HTTP status codes
        if response.status_code >= 200 and response.status_code < 300:
            # Success responses
            try:
                # Try to parse as JSON first
                json_data = response.json()
                return json.dumps({
                    "success": True,
                    "status_code": response.status_code,
                    "content_type": content_type,
                    "data": json_data,
                    "request_info": request_info
                }, indent=2, default=str)
                
            except json.JSONDecodeError:
                # Not valid JSON, handle based on content type
                response_text = response.text
                
                if 'html' in content_type:
                    # HTML response - extract useful info
                    title_match = response_text.find('<title>')
                    if title_match != -1:
                        title_end = response_text.find('</title>', title_match)
                        title = response_text[title_match + 7:title_end] if title_end != -1 else "Unknown"
                    else:
                        title = "No title found"
                    
                    return json.dumps({
                        "success": True,
                        "status_code": response.status_code,
                        "content_type": content_type,
                        "data_type": "html",
                        "html_title": title,
                        "html_content": response_text[:1000] + "..." if len(response_text) > 1000 else response_text,
                        "full_html_length": len(response_text),
                        "request_info": request_info
                    }, indent=2)
                    
                elif 'text' in content_type or 'plain' in content_type:
                    # Plain text response
                    return json.dumps({
                        "success": True,
                        "status_code": response.status_code,
                        "content_type": content_type,
                        "data_type": "text",
                        "text_content": response_text,
                        "request_info": request_info
                    }, indent=2)
                    
                else:
                    # Unknown content type
                    return json.dumps({
                        "success": True,
                        "status_code": response.status_code,
                        "content_type": content_type,
                        "data_type": "unknown",
                        "raw_content": response_text[:500] + "..." if len(response_text) > 500 else response_text,
                        "content_length": len(response_text),
                        "request_info": request_info
                    }, indent=2)
        
        elif response.status_code >= 300 and response.status_code < 400:
            # Redirection responses
            location = response.headers.get('location', 'Not specified')
            return json.dumps({
                "success": False,
                "status_code": response.status_code,
                "error": f"Redirection to: {location}",
                "error_type": "redirection",
                "location": location,
                "content_type": content_type,
                "response_text": response.text[:500] + "..." if len(response.text) > 500 else response.text,
                "request_info": request_info
            }, indent=2)
            
        elif response.status_code >= 400 and response.status_code < 500:
            # Client error responses
            try:
                error_data = response.json()
                return json.dumps({
                    "success": False,
                    "status_code": response.status_code,
                    "error": error_data.get('error', f'Client error {response.status_code}'),
                    "error_type": "client_error",
                    "error_data": error_data,
                    "request_info": request_info
                }, indent=2)
            except json.JSONDecodeError:
                return json.dumps({
                    "success": False,
                    "status_code": response.status_code,
                    "error": f"Client error {response.status_code}: {response.text[:200]}",
                    "error_type": "client_error",
                    "content_type": content_type,
                    "response_text": response.text,
                    "request_info": request_info
                }, indent=2)
                
        elif response.status_code >= 500:
            # Server error responses
            try:
                error_data = response.json()
                return json.dumps({
                    "success": False,
                    "status_code": response.status_code,
                    "error": error_data.get('error', f'Server error {response.status_code}'),
                    "error_type": "server_error",
                    "error_data": error_data,
                    "request_info": request_info
                }, indent=2)
            except json.JSONDecodeError:
                return json.dumps({
                    "success": False,
                    "status_code": response.status_code,
                    "error": f"Server error {response.status_code}: {response.text[:200]}",
                    "error_type": "server_error",
                    "content_type": content_type,
                    "response_text": response.text,
                    "request_info": request_info
                }, indent=2)
        
    except requests.exceptions.ConnectionError as e:
        return json.dumps({
            "success": False,
            "error": f"Could not connect to Rails server at {API_ENDPOINT}",
            "error_type": "connection_error",
            "error_details": str(e),
            "stack_trace": traceback.format_exc(),
            "request_info": request_info
        }, indent=2)
        
    except requests.exceptions.Timeout as e:
        return json.dumps({
            "success": False,
            "error": "Request timed out after 30 seconds",
            "error_type": "timeout_error",
            "error_details": str(e),
            "stack_trace": traceback.format_exc(),
            "request_info": request_info
        }, indent=2)
        
    except requests.exceptions.TooManyRedirects as e:
        return json.dumps({
            "success": False,
            "error": "Too many redirects",
            "error_type": "redirect_error",
            "error_details": str(e),
            "stack_trace": traceback.format_exc(),
            "request_info": request_info
        }, indent=2)
        
    except requests.exceptions.RequestException as e:
        return json.dumps({
            "success": False,
            "error": f"HTTP request failed: {str(e)}",
            "error_type": "request_error",
            "error_details": str(e),
            "stack_trace": traceback.format_exc(),
            "request_info": request_info
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "error_type": "unexpected_error",
            "error_details": str(e),
            "stack_trace": traceback.format_exc(),
            "request_info": request_info
        }, indent=2)

# Tools
@tool
def run_rails_console_command(rails_console_command: str, message_to_user: str, internal_thoughts: str, state: Annotated[LlamaBotState, InjectedState]) -> str:
    """
    Run a Rails console command.
    Message to user is a string to tell the user what you're doing.
    Internal thoughts are your thoughts about the command.
    State is the state of the agent.
    """
    print ("API TOKEN", state.get("api_token")) # empty. only messages is getting passed through.
    
    # Configuration
    RAILS_SERVER_URL = os.getenv("LLAMAPRESS_API_URL")

    API_ENDPOINT = f"{RAILS_SERVER_URL}/llama_bot/agent/command"
    
    try:
        # Make HTTP request to Rails AP
        response = requests.post(
            API_ENDPOINT,
            json={'command': rails_console_command},
            headers={'Content-Type': 'application/json', 'Authorization': f'LlamaBot {state.get("api_token")}'},
            timeout=30  # 30 second timeout
        )
        
        # Parse the response
        if response.status_code == 200:
            data = response.json()
            result = data.get('result')
            result_type = data.get('type')
            
            # Format the output nicely
            if isinstance(result, (list, dict)):
                formatted_result = json.dumps(result, indent=2, default=str)
            else:
                formatted_result = str(result)
            
            # Create a JSON-serializable dictionary
            result_data = {
                "command": rails_console_command,
                "result": formatted_result,
                "type": result_type
            }
            
            # Serialize to JSON string for safe transmission
            return json.dumps(result_data, ensure_ascii=False, indent=2)
        elif response.status_code == 403:
            error_data = response.json()
            return f"Error: {error_data.get('error', 'Command not allowed')}"
            
        elif response.status_code == 500:
            error_data = response.json()
            return f"Rails Error: {error_data.get('error', 'Unknown error')}\nType: {error_data.get('type', 'Unknown')}"
            
        else:
            return f"HTTP Error {response.status_code}: {response.text}"
            
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Rails server. Make sure your Rails app is running on http://localhost:3000"
        
    except requests.exceptions.Timeout:
        return "Error: Request timed out. The Rails command may be taking too long to execute."
        
    except requests.exceptions.RequestException as e:
        return f"Request Error: {str(e)}"
        
    except json.JSONDecodeError:
        return f"Error: Invalid JSON response from server. Raw response: {response.text}"
        
    except Exception as e:
        return f"Unexpected Error: {str(e)}"

# Global tools list
# tools = [run_rails_console_command]
tools = [rails_https_request]

# Node
def llamabot(state: LlamaBotState):
   additional_instructions = state.get("agent_prompt")
#    breakpoint()


   # System message
   sys_msg = SystemMessage(content=f"""You are LlamaBot, a helpful AI assistant.
                        In normal chat conversations, feel free to implement markdown formatting to make your responses more readable, if it's appropriate.
                        Here are additional instructions provided by the user: <USER_INSTRUCTIONS> {additional_instructions} </USER_INSTRUCTIONS> 
                        You can do HTTP requests to the Rails server using the rails_https_request tool and the following routes: <RAILS_ROUTES> {state.get("available_routes")} </RAILS_ROUTES>""")

#    llm = ChatOpenAI(model="o4-mini")
   llm = ChatOpenAI(model="gpt-4o")

#    breakpoint()

   llm_with_tools = llm.bind_tools(tools)
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

def build_workflow(checkpointer=None):
    # Graph
    builder = StateGraph(LlamaBotState)

    # Define nodes: these do the work
    builder.add_node("llamabot", llamabot)
    builder.add_node("tools", ToolNode(tools))

    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "llamabot")
    builder.add_conditional_edges(
        "llamabot",
        # If the latest message (result) from llamabot is a tool call -> tools_condition routes to tools
        # If the latest message (result) from llamabot is a not a tool call -> tools_condition routes to END
        tools_condition,
    )
    builder.add_edge("tools", "llamabot")

    react_graph = builder.compile(checkpointer=checkpointer)

    return react_graph