from asyncio import Lock, CancelledError
from fastapi import WebSocket

from websocket.web_socket_request_context import WebSocketRequestContext
from typing import Dict, Optional

from langchain.schema import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

import json
import importlib
import os
import logging

#This is an example of a custom state object for a custom agent.
from agents.llamapress_legacy.state import LlamaPressMessage

# from llm.websocket.websocket_helper import send_json_through_websocket
# from llm.workflows.nodes import build_workflow, build_workflow_saas

logger = logging.getLogger(__name__)

def get_or_create_checkpointer():
    """Get persistent checkpointer, creating once if needed"""
    db_uri = os.getenv("POSTGRES_URI_CUSTOM")
    checkpointer = MemorySaver() # save in RAM if postgres is not available
    if db_uri:
        try:
            # Create connection pool and PostgresSaver directly
            pool = AsyncConnectionPool(db_uri)
            checkpointer = AsyncPostgresSaver(pool)
            checkpointer.setup()
            logger.info("✅✅✅ Using PostgreSQL persistence!")
        except Exception as e:
            logger.warning(f"Failed to connect to PostgreSQL: {e}. Using MemorySaver.")
    else:
        logger.info("❌❌❌ No DB_URI found. Using MemorySaver for session-based persistence.")
    
    return checkpointer

class RequestHandler:
    def __init__(self):
        self.locks: Dict[int, Lock] = {}
    
    def _get_lock(self, websocket: WebSocket) -> Lock:
        """Get or create a lock for a specific websocket connection"""
        ws_id = id(websocket)
        if ws_id not in self.locks:
            self.locks[ws_id] = Lock()
        return self.locks[ws_id]

    async def handle_request(self, websocket: WebSocket):
        """Handle incoming WebSocket requests with proper locking and cancellation"""
        if not message:
            logger.error("Received null message in handle_request")
            await websocket.send_json({
                "type": "error",
                "content": "Invalid message received"
            })
            return
        
        lock = self._get_lock(websocket)
        
        async with lock:
            try:
                context = WebSocketRequestContext(websocket, langgraph_checkpointer=get_or_create_checkpointer())

                ### The following section allows LlamaPress Users to use other Agents Defined in our langgraph.json file, as configured
                ### In their Site Settings.

                app, state, use_saas_workflow = await self.get_langgraph_app_and_state(message, context)

                #########################################################################################
                #########################################################################################

                # Define the config dictionary
                config = {
                    "configurable": {
                        "thread_id": f"{message.web_page_id}", #lol, this is laughably bad. It must be something reliable and unique, that we can save specific to the page.
                        "page_id": message.web_page_id,
                        "page_url": f"https://llamapress.ai/pages/{message.web_page_id}",
                    },
                    "context": context,
                }
                # breakpoint()

                #If we are using our v1 workflow, the nodes send messages through the websocket in a format and way that LlamaPress can understand.
                if use_saas_workflow:
                    async for output in app.astream(state, config=config):
                        logger.info(f"Workflow output: {output}")
                #########################################################################################
                #########################################################################################

                else: # This is our LlamaBot v2 workflow, which relies on langgraph streaming to send messages through the websocket. We do not use the websocket directly in LangGraph nodes in this version.
                    async for chunk in app.astream(state, config=config, stream_mode=["updates", "messages"]):
                        is_this_chunk_an_llm_message = isinstance(chunk, tuple) and len(chunk) == 2 and chunk[0] == 'messages'
                        is_this_chunk_an_update_stream_type = isinstance(chunk, tuple) and len(chunk) == 2 and chunk[0] == 'updates'
                        if is_this_chunk_an_llm_message:
                            message_chunk_from_llm = chunk[1][0] #AIMessageChunk object -> https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.AIMessageChunk.html
                        elif is_this_chunk_an_update_stream_type:
                            # breakpoint()
                            state_object = chunk[1]
                            
                            # Handle dynamic agent key - look for messages in any nested dict
                            messages = None
                            for agent_key, agent_data in state_object.items():
                                if isinstance(agent_data, dict) and 'messages' in agent_data:
                                    messages = agent_data['messages']
                                    break
                            
                            logger.info(f"Workflow output: {chunk}")
                        else:
                            logger.info(f"Workflow output: {chunk}")

            except CancelledError as e:
                logger.info("handle_request was cancelled")
                await websocket.send_json({
                    "type": "error",
                    "content": f"Cancelled!"
                })
                raise e
            except Exception as e:
                logger.error(f"Error handling request: {str(e)}", exc_info=True)
                await websocket.send_json({
                    "type": "error",
                    "content": f"Error processing request: {str(e)}"
                })
                raise e

    async def get_chat_history(self, thread_id: str):
        websocket_context = WebSocketRequestContext(None, langgraph_checkpointer=get_or_create_checkpointer())
        app, _, _ = await self.get_langgraph_app_and_state(None, websocket_context)
        config = {"configurable": {"thread_id": thread_id}}
        state_history = await app.aget_state(config=config)
        return state_history[0] #gets the actual state.

    def cleanup_connection(self, websocket: WebSocket):
        """Clean up resources when a connection is closed"""
        ws_id = id(websocket)
        if ws_id in self.locks:
            del self.locks[ws_id]

    def get_workflow_from_langgraph_json(self, message: LlamaPressMessage):
        langgraph_json = json.load(open("langgraph.json"))
        langgraph_workflow = langgraph_json.get("graphs").get(message.llamabot_agent_name)
        return langgraph_workflow
    
    async def get_langgraph_app_and_state(self, message: LlamaPressMessage, context: WebSocketRequestContext):
        # Initialize LangGraph & the state
        if message is None:
            return await build_workflow_saas(context), None, True #default to using the SaaS workflow.

        app = None
        state = None
        use_saas_workflow = True
        user_message = f"<system> {message.system_prompt} </system> <user> {message.user_message} </user>"
        
        # Initialize LangGraph & the state
        if message.llamabot_agent_name is not None: # Default to using the SaaS workflow.
            # First, Map from langgraph.json to the correct workflow
            langgraph_workflow = self.get_workflow_from_langgraph_json(message)
            if langgraph_workflow is not None:
                app = await self.get_app_from_workflow_string(langgraph_workflow, context)
                use_saas_workflow = False
            else:
                raise ValueError(f"Unknown workflow: {message.llamabot_agent_name}")
        
        
        if use_saas_workflow:
            app = await build_workflow_saas(context)

            #Convert LlamaPressMessage to the SaaS Workflow State (LlamaPressWorkflowState object):
            state = {
                "messages": [HumanMessage(content=message.user_message)],
                "user_message": user_message,
                "web_page_id": message.web_page_id,
                "selected_element": message.selected_element,
                "context": message.context,
                "initial_html_content": message.file_contents,
                "final_html_content": message.file_contents, #by default, the final html content is the same as the initial html content.
                "wordpress_api_encoded_token": message.wordpress_api_encoded_token,
                "wordpress_page_id": None,
                "decision": "",
                "user_llamapress_api_token": message.user_llamapress_api_token,
                "design_requirements": "",
                "javascript_console_errors": message.javascript_console_errors,
                "end": False
            }
        else: #use non-saas coding agent workflow (legacy)
            # app = await build_workflow(context)
            state = { #TODO:We have to figure out the correct state for the workflow.
                "messages": [HumanMessage(content=message.user_message)],
                "requirements": "",
                "plan": "",
                "coding_commands": [],
                "requirements_tested_and_passed": False,
                "rails_project_path": "/Users/kodykendall/SoftEngineering/LLMPress/LlamaPress",
                "end": False
            }
        
        return app, state, use_saas_workflow

    
    def get_app_from_workflow_string(self, workflow_string: str, context: WebSocketRequestContext):
        # Split the path into module path and function name
        module_path, function_name = workflow_string.split(':')
        # Remove './' if present and convert path to module format
        if module_path.startswith('./'):
            module_path = module_path[2:]
        module_path = module_path.replace('/', '.').replace('.py', '')

        # Dynamically import the module and get the function
        module = importlib.import_module(module_path)
        workflow_builder = getattr(module, function_name)

        # Build the workflow using the imported function
        return workflow_builder(context)