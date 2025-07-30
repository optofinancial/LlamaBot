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
import asyncio
import json

from app.agents.utils.playwright_screenshot import capture_page_and_img_src
from app.agents.utils.images import encode_image


from .helpers import reassemble_fragments

load_dotenv()

from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

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
            return {'tool_name': 'write_html_page', 'tool_args': {'full_html_document': full_html_document}}
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
def get_screenshot_and_html_content_using_playwright(url: str, state: Annotated[dict, InjectedState]) -> tuple[str, list[str]]:
    """
    Get the screenshot and HTML content of a webpage using Playwright. Then, generate the HTML as a clone, and save it to the file system. 
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = f"tmp/url-screenshot-{timestamp}.png"  
    trimmed_html_content, image_sources = asyncio.run(capture_page_and_img_src(url, image_path))
    
    if not trimmed_html_content:
        return "Screenshot functionality is not available. Please install Playwright to enable this feature."

    # This returns a ToolMessage
    # From here, we're going to return back to the url_clone_agent node, and then we'll force the model to call the write_html_page tool.
    return {'tool_name': 'get_screenshot_and_html_content_using_playwright', 'tool_args': {'url': url, 'image_path': image_path}, 'tool_data': {'trimmed_html_content': trimmed_html_content}}

# Global tools list
url_clone_tools = [get_screenshot_and_html_content_using_playwright, write_html_page]
image_clone_tools = []

# Node
def url_clone_agent(state: MessagesState):
   last_message = state.get("messages")[-1]
   if type(last_message) == ToolMessage:
        data = json.loads(last_message.content)
        if data.get("tool_name") == "get_screenshot_and_html_content_using_playwright":
            # force a tool call to the LLM with write_html_page
            image_path = data.get("tool_args").get("image_path")
            base64_image = encode_image(image_path)
            llm_forced_tool_call = ChatOpenAI(model="gpt-4o").bind_tools([write_html_page], tool_choice="write_html_page")
            
            print(f"Making our call to o3 vision right now")
    
            breakpoint()
            response = llm_forced_tool_call.invoke([
                SystemMessage(content="""
                    ### SYSTEM
        You are "Pixel-Perfect Front-End", a senior web-platform engineer who specialises in
        * redesigning bloated, auto-generated pages into clean, semantic, WCAG-conformant HTML/CSS
        * matching the *visual* layout of the reference screenshot to within ±2 px for all major breakpoints

        When you reply you MUST:
        1. **Think step-by-step silently** ("internal reasoning"), then **output nothing but the final HTML inside a single fenced code block**.
        2. **Inline zero commentary** – the code block is the entire answer.
        3. Use **only system fonts** (font-stack: `Roboto, Arial, Helvetica, sans-serif`) and a single `<style>` block in the `<head>`.
        4. Avoid JavaScript unless explicitly asked; replicate all interactions with pure HTML/CSS where feasible.
        5. Preserve all outbound links exactly as provided in the RAW_HTML input.
        7. Ensure the layout is mobile-first responsive (Flexbox/Grid) and maintains the same visual hierarchy:  
        e.g) **header ➔ main (logo, search box, buttons, promo) ➔ footer**.

        ### USER CONTEXT
        You will receive two payloads:

        **SCREENSHOT** – a screenshot of the webpage.  
        **RAW_HTML** – the stripped, uglified DOM dump (may include redundant tags, hidden dialogs, etc.).

        ### TASK
        1. **Infer the essential visual / UX structure** of the page from SCREENSHOT.  
        2. **Cross-reference** with RAW_HTML only to copy:
        * anchor `href`s & visible anchor text
        * any aria-labels, alt text, or titles that improve accessibility.
        3. **Discard** every element not visible in the screenshot (menus, dialogs, split-tests, inline JS blobs).
        4. Re-create the page as a **single HTML document** following best practices described above.

        You using the `write_html_page` function/tool, and are generating a brand new vanilla HTML/Tailwind CSS/JavaScript code, include comments that explain what you're doing for each logical block of code you're about to generate."
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

        ### OUTPUT FORMAT
        Return one fenced code block starting with <!DOCTYPE html> and ending with </html>
        No extra markdown, no explanations, no leading or trailing whitespace outside the code block.
                    
                    Here is the trimmed down HTML:
                    {trimmed_html_content}
                """),
                HumanMessage(content=f"Here is the trimmed down HTML: {data.get('tool_data').get('trimmed_html_content')}"),
                HumanMessage(content=[
                    {"type": "text", "text": "Please use our write_html_page tool to clone this webpage based on the screenshot and HTML content provided."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ])
            ])

            return {"messages": [response]}
        
        else:
            return {}
   
   # In the default case force it to call the get_screenshot_and_html_content_using_playwright tool
   # System message
   sys_msg = SystemMessage(content="You are an agent that can 'deep clone' by using playwright to navigate to a URL, take a screenshot of the page, look at the HTML structure, and clone the HTML page out. You have access to the tool `get_screenshot_and_html_content_using_playwright` to do this. If the user requests a deep clone, you should use this tool.")
   llm = ChatOpenAI(model="o4-mini")
   llm_with_tools = llm.bind_tools(url_clone_tools, tool_choice="get_screenshot_and_html_content_using_playwright")
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Node
def router_node(state: LlamaPressState):
    last_message = state.get("messages")[-1]
    # if "deep clone" in last_message.content.lower():
    return {"next": "url_clone_agent"}
    # elif "clone" in last_message.content.lower():
    #     return {"next": "image_clone_agent"}
    # else:
    #     return {"next": "html_agent"}

# Node
def image_clone_agent(state: LlamaPressState):
    # instructions = state.get("agent_prompt", "")
    system_content = (
        f"Simply respond with, I am the image clone agent!"
    )

    model = ChatOpenAI(model="gpt-4.1-2025-04-14")
    llm_with_tools = model.bind_tools(image_clone_tools)
    llm_response_message = llm_with_tools.invoke([SystemMessage(content=system_content)] + state["messages"])
    llm_response_message.response_metadata["created_at"] = str(datetime.now())

    return {"messages": [llm_response_message]}

def build_workflow(checkpointer=None):
    # Graph
    builder = StateGraph(LlamaPressState)

    # Define nodes: these do the work
    builder.add_node("router", router_node)
    builder.add_node("url_clone_agent", url_clone_agent)
    builder.add_node("image_clone_agent", image_clone_agent)
    builder.add_node("url_clone_tools", ToolNode(url_clone_tools))
    # builder.add_node("url_clone_tools", ToolNode(url_clone_tools))
    # builder.add_node("image_clone_tools", ToolNode(image_clone_tools))

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
        # If the latest message (result) from url_clone_agent is a tool call -> route to url_clone_tools
        # If the latest message (result) from url_clone_agent is not a tool call -> route to END
        tools_condition,
        {"tools": "url_clone_tools"}
    )

    # builder.add_conditional_edges(
    #     "image_clone_agent",
    #     # If the latest message (result) from image_clone_agent is a tool call -> route to image_clone_tools
    #     # If the latest message (result) from image_clone_agent is not a tool call -> route to END
    #     tools_condition,
    #     {"tools": "image_clone_tools"} #we have to map this since we're not using the default tools_condition
    # )

    builder.add_edge("url_clone_tools", "url_clone_agent")
    # builder.add_edge("image_clone_tools", "image_clone_agent")

    builder.add_edge("url_clone_agent", END)
    builder.add_edge("image_clone_agent", END)

    clone_agent = builder.compile(checkpointer=checkpointer, name="clone_agent")

    return clone_agent