FROM nikolaik/python-nodejs:python3.11-nodejs18

WORKDIR /app

# Install dependencies (cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# COPY ../assets ./assets

# Copy the rest of the app
COPY . .

# Environment variables (all optional)
# DB_URI: PostgreSQL connection string (falls back to MemorySaver if not provided)
# Example: postgresql://user:password@host:port/database
ENV DB_URI=""

# Expose port
EXPOSE 8000

# CMD ["bash", "-c", "python init_pg_checkpointer.py --uri $POSTGRES_URI_CUSTOM && uvicorn app:app --host 0.0.0.0 --port 8000"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
