# init_pg_checkpointer.py
import os   
from langgraph.checkpoint.postgres import PostgresSaver, ConnectionPool
from dotenv import load_dotenv
from psycopg import Connection

#https://github.com/langchain-ai/langgraph/issues/2887

load_dotenv()

db_uri = os.getenv("POSTGRES_URI_CUSTOM")

if db_uri is None:
    print("POSTGRES_URI_CUSTOM is not set")
    exit(1)

# Create connection pool
# pool = ConnectionPool(db_uri)
conn = Connection.connect(db_uri, autocommit=True)

# Create the saver
checkpointer = PostgresSaver(conn)

# This runs DDL like CREATE TABLE and CREATE INDEX
# including CREATE INDEX CONCURRENTLY, which must be run outside a transaction
checkpointer.setup()

print("✅ Checkpointer tables & indexes initialized.")