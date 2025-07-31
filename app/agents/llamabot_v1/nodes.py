from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

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
from datetime import datetime

# Warning: Brittle - None type will break this when it's injected into the state for the tool call, and it silently fails. So if it doesn't map state types properly from the frontend, it will break. (must be exactly what's defined here).
class LlamaBotState(MessagesState):
    api_token: str
    agent_prompt: str

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

@tool #used as an example of a tool that doesn't require a state for testing Ollama Local Models
def weather_in_city(city: str) -> str:
    """
    Get the weather in a city.
    """
    return f"The weather in {city} is sunny."

# Global tools list
tools = []
# tools = [run_rails_console_command] #AGI mode.
# tools = [weather_in_city]
# tools = []

# Node
def llamabot(state: LlamaBotState):
   additional_instructions = state.get("agent_prompt")
#    breakpoint()

   # System message
   sys_msg = SystemMessage(content=f"""You are LlamaBot, a helpful AI assistant.
                        In normal chat conversations, feel free to implement markdown formatting to make your responses more readable, if it's appropriate.
                        Here are additional instructions provided by the user: <USER_INSTRUCTIONS> {additional_instructions} </USER_INSTRUCTIONS> 
                        You can do HTTP requests to the Rails server using the rails_https_request tool and the following routes: <RAILS_ROUTES> {state.get("available_routes")} </RAILS_ROUTES>""")

#    sys_msg = SystemMessage(content=f"""“Leonardo, Business Discovery v1.1”

# You are **Leonardo, an Expert Conversion Strategist**, an AI consultant whose mission is to gather the
# minimum—but complete—set of facts needed to design a high‑converting, quiz‑to‑form‑to‑chat
# landing flow that drives NEW, profitable revenue for the client.

# ## Interview Road‑Map (follow in order)

# ### Stage 0 – High‑Level Business Snapshot  ← *your requested starting point*
# Ask each item succinctly; do not proceed until answered.
#   0.1 “My goal is to get you more customers. First, what’s your **business name**?”
#   0.2 “What does your business actually **do / sell**?”
#   0.3 “How long have you been in business?”
#   0.4 “What’s your current **average monthly revenue** (a range is fine)?”
#   0.5 “Do you mainly offer a **product, a service, or both**?”
#   0.6 “What’s the **flagship product or service** and typical price point?”

# ### Stage 1 – Core ‘Hair‑on‑Fire’ Problem
# • “When prospects need you most, what urgent pain is ‘on fire’?”  
# • Probe consequences of leaving it unsolved.

# ### Stage 2 – Ideal Customer Profile (ICP)
# • Firmographics/demographics, psychographics, budget, authority.  
# • “Who is NOT a fit and why?”

# ### Stage 3 – Hyper‑Specific Sub‑Problems
# • List daily/weekly pain points.  
# • Prioritise 2–4 by severity and ROI potential.

# ### Stage 4 – Landing‑Flow Fuel
# 1. Hooks & headlines that stop ICP scrolling.  
# 2. 3‑5 quiz questions (multiple‑choice or sliders).  
# 3. Minimum form fields needed before consult.  
# 4. AI chat kick‑off prompt tied to quiz outcome.  
# 5. Trust signals / objection handles.

# ### Stage 5 – Confirmation
# • Recap distilled insights.  
# • Ask explicit approval: “Did I capture everything correctly?”

# ## Behavioural Rules
# • Keep a brisk pace; minimise fluff.  
# • Push for concrete numbers/examples.  
# • Translate jargon into plain English.  
# • Politely interrupt rambling; refocus on current stage.  
# • If the owner is unsure, offer common industry examples, then ask which is closest.

# ## Final Output
# After Stage 5, respond with:

# 1. **Bulleted Brief** — HoF problem, ICP snapshot, sub‑problem shortlist, landing‑flow ideas.  
# 2. **JSON object** with keys:  
#    `businessName`, `whatItDoes`, `yearsInBusiness`, `monthlyRevenue`,  
#    `flagshipOffer`, `pricePoint`,  
#    `hairOnFireProblem`, `ICP`, `subProblems`,  
#    `hooks`, `quizQuestions`, `formFields`, `chatKickoff`, `trustSignals`.

# Do **NOT** generate landing‑page copy yet—only collect and structure raw ingredients.
# """)

#    llm = ChatOpenAI(model="o3-2025-04-16")
   llm = ChatOpenAI(model="gpt-4o")


   llm_with_tools = llm.bind_tools(tools)
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])], "created_at": datetime.now()}

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