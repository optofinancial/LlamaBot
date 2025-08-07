from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode

import asyncio

from app.agents.utils.playwright_screenshot import capture_page_and_img_src

from openai import OpenAI
from app.agents.utils.images import encode_image
import os

@tool
def get_user_goals(user_id: str) -> str:
    """
    Get the user's goals.
    """
    #TODO: Get the user's goals from the database.
    #TODO: Return the results as a string.
    return "user goals"

    #TODO: Connect to the database using the DB_URI in the .env file.
    #TODO: Execute the SQL query and return the results.
    #TODO: Return the results as a string.

@tool
def write_sql_query(sql_query: str) -> str:
    """
    Write a SQL query to a file.
    """
    #TODO: Connect to the database using the DB_URI in the .env file.
    #TODO: Execute the SQL query and return the results.
    #TODO: Return the results as a string.
    # use psycopg to connect to the database and execute the query, then return the results as a string.

    DB_URI = os.getenv("DB_URI")
    import psycopg
    conn = psycopg.connect(DB_URI)
    cursor = conn.cursor()
    cursor.execute(sql_query)
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    # breakpoint()

    return f"query: {sql_query}\nresults: {results}"

# Global tools list
tools = [write_sql_query]

# System message
sys_msg = SystemMessage(content="""
You are opto, a helpful assistnat that is able to write SQL queries to access Transaction data and other 
financial account information from the user. Here is the stucture of their database:

the table name is transactions.

interface transaction {
  id: string;
  amount: number;
  description: string;
  category: string;
  category_tag: string;
  date: string;
  merchant_name?: string;
  type: string;
  account_id: string;
  pending: boolean;
  income_flag: boolean;
}""")

# Node
def software_developer_assistant(state: MessagesState):
   llm = ChatOpenAI(model="o4-mini")
   llm_with_tools = llm.bind_tools(tools)
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


# //TODO: This is where you'll implement opto logic
def build_workflow(checkpointer=None):
    # Graph
    builder = StateGraph(MessagesState)

    # Define nodes: these do the work
    builder.add_node("software_developer_assistant", software_developer_assistant)
    builder.add_node("tools", ToolNode(tools))

    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "software_developer_assistant")
    builder.add_conditional_edges(
        "software_developer_assistant",
        # If the latest message (result) from software_developer_assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from software_developer_assistant is a not a tool call -> tools_condition routes to END
        tools_condition,
    )
    builder.add_edge("tools", "software_developer_assistant")
    react_graph = builder.compile(checkpointer=checkpointer)

    return react_graph