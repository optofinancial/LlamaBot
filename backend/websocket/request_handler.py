from asyncio import Lock, CancelledError

from fastapi import FastAPI, WebSocket

from websocket.web_socket_request_context import WebSocketRequestContext
from typing import Dict, Optional

from langchain.schema import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from dotenv import load_dotenv
import json
import importlib
import os
import logging

#This is an example of a custom state object for a custom agent.
from agents.llamapress_legacy.state import LlamaPressMessage

# from llm.websocket.websocket_helper import send_json_through_websocket
# from llm.workflows.nodes import build_workflow, build_workflow_saas
logger = logging.getLogger(__name__)

load_dotenv()

class RequestHandler:
    def __init__(self, app: FastAPI):
        self.locks: Dict[int, Lock] = {}
        self.app = app
    
    def _get_lock(self, websocket: WebSocket) -> Lock:
        """Get or create a lock for a specific websocket connection"""
        ws_id = id(websocket)
        if ws_id not in self.locks:
            self.locks[ws_id] = Lock()
        return self.locks[ws_id]

    async def handle_request(self, message: dict, websocket: WebSocket):
        """Handle incoming WebSocket requests with proper locking and cancellation"""
        # breakpoint()

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

                ### The following section allows LlamaPress Users to use other Agents Defined in our langgraph.json file, as configured
                ### In their Site Settings.

                # langgraph_checkpointer = get_or_create_checkpointer()
                app, state = self.get_langgraph_app_and_state(message)
                # breakpoint()

                #########################################################################################
                #########################################################################################

                ## LEGACY CONFIG:
                # # # Define the config dictionary
                # config = {
                #     "configurable": {
                #         "thread_id": f"{message.web_page_id}", #lol, this is laughably bad. It must be something reliable and unique, that we can save specific to the page.
                #         "page_id": message.web_page_id,
                #         "page_url": f"https://llamapress.ai/pages/{message.web_page_id}",
                #     },
                #     "context": context,
                # }

                # breakpoint()
                config = {
                    "configurable": {
                        "thread_id": f"{message.get('thread_id')}"
                    }
                }

                # #If we are using our v1 workflow, the nodes send messages through the websocket in a format and way that LlamaPress can understand.
                # if use_saas_workflow:
                #     async for output in app.astream(state, config=config):
                #         logger.info(f"Workflow output: {output}")
                # #########################################################################################
                # #########################################################################################

                # else: # This is our LlamaBot v2 workflow, which relies on langgraph streaming to send messages through the websocket. We do not use the websocket directly in LangGraph nodes in this version.
                async for chunk in app.astream(state, config=config, stream_mode=["updates", "messages"]):
                    is_this_chunk_an_llm_message = isinstance(chunk, tuple) and len(chunk) == 2 and chunk[0] == 'messages'
                    is_this_chunk_an_update_stream_type = isinstance(chunk, tuple) and len(chunk) == 2 and chunk[0] == 'updates'
                    if is_this_chunk_an_llm_message:
                        message_chunk_from_llm = chunk[1][0] #AIMessageChunk object -> https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.AIMessageChunk.html
                    
                    
                    elif is_this_chunk_an_update_stream_type: # This means that LangGraph has given us a state update. This will often include a new message from the AI.
                        state_object = chunk[1]
                    
                        # Handle dynamic agent key - look for messages in any nested dict
                        messages = None
                        for agent_key, agent_data in state_object.items():
                            
                            did_agent_have_a_message_for_us = isinstance(agent_data, dict) and 'messages' in agent_data
                            if did_agent_have_a_message_for_us:
                                messages = agent_data['messages'] #Question: is this ALL messages coming through, or just the latest AI message?

                                # AIMessage is not serializable to JSON, so we need to convert it to a string.
                                messages_as_string = [message.content for message in messages]
                                # breakpoint()

                                await websocket.send_json({
                                    "type": "ai", #matches our langgraph streaming type.
                                    "content": messages_as_string[0]
                                })
                                break
                        
                        logger.info(f"LangGraph Output (State Update): {chunk}")

                        # chunk will look like this:
                        # {'llamabot': {'messages': [AIMessage(content='Hello! I hear you loud and clear. I’m LlamaBot, your full-stack Rails developer assistant. How can I help you today?', additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'model_name': 'o4-mini-2025-04-16', 'service_tier': 'default'}, id='run--ce385bc4-fecb-4127-81d2-1da5814874f8')]}}

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
        # websocket_context = WebSocketRequestContext(None, langgraph_checkpointer=get_or_create_checkpointer())
        app, _ = await self.get_langgraph_app_and_state(None)
        config = {"configurable": {"thread_id": thread_id}}
        state_history = await app.aget_state(config=config)
        return state_history[0] #gets the actual state.

    def get_or_create_checkpointer(self):
        """Get persistent checkpointer, creating once if needed"""
        # breakpoint()
        if self.app.state.checkpointer is not None:
            return self.app.state.checkpointer
        
        db_uri = os.getenv("DB_URI")
        self.app.state.checkpointer = MemorySaver() # save in RAM if postgres is not available
        if db_uri:
            try:
                # Create connection pool and PostgresSaver directly
                pool = AsyncConnectionPool(db_uri)
                self.app.state.checkpointer = AsyncPostgresSaver(pool)
                self.app.state.checkpointer.setup()
                logger.info("✅✅✅ Using PostgreSQL persistence!")
            except Exception as e:
                logger.warning(f"Failed to connect to PostgreSQL: {e}. Using MemorySaver.")
        else:
            logger.info("❌❌❌ No DB_URI found. Using MemorySaver for session-based persistence.")
        
        return self.app.state.checkpointer

    def cleanup_connection(self, websocket: WebSocket):
        """Clean up resources when a connection is closed"""
        ws_id = id(websocket)
        if ws_id in self.locks:
            del self.locks[ws_id]

    def get_workflow_from_langgraph_json(self, message: dict):
        langgraph_json = json.load(open("../langgraph.json"))
        langgraph_workflow = langgraph_json.get("graphs").get(message.get("agent_name"))
        return langgraph_workflow
    
    def get_langgraph_app_and_state(self, message: dict):
        # Initialize LangGraph & the state
        # if message is None:
        #     return await build_workflow_saas(context), None, True #default to using the SaaS workflow.

        app = None
        state = None #this would map to the corresponding AgentState object. So this needs to mirror what is in the client state config.

        
        # Initialize LangGraph & the state
        if message.get("agent_name") is not None: # This is so that we can configure different agents & workflows in our Rails app.
            # First, Map from langgraph.json to the correct workflow
            langgraph_workflow = self.get_workflow_from_langgraph_json(message)
            if langgraph_workflow is not None:
                app = self.get_app_from_workflow_string(langgraph_workflow)
                state = {
                    "messages": [HumanMessage(content=message.get("user_message"))]
                }

            else:
                raise ValueError(f"Unknown workflow: {message.get('agent_name')}")
        
        return app, state
    
    def get_app_from_workflow_string(self, workflow_string: str):
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
        return workflow_builder(checkpointer=self.get_or_create_checkpointer())