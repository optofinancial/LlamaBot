# LlamaBot Backend

FastAPI backend for LlamaBot with optional PostgreSQL persistence.

## Docker Usage

### Basic Usage (MemorySaver)
```bash
docker run -e OPENAI_API_KEY=your_openai_key -p 8000:8000 llamabot-backend
```

### With PostgreSQL Persistence (DB_URI is optional Postgres connection string)
```bash
docker run \
  -e OPENAI_API_KEY=your_openai_key \
  -e DB_URI="postgresql://user:password@host:5432/database" \
  -p 8000:8000 \
  llamabot-backend
```

## Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for LLM access | - |
| `DB_URI` | No | PostgreSQL connection string | "" (uses MemorySaver) |
| `LANGSMITH_API_KEY` | No | LangSmith API key for tracing | - |

## Database Behavior

- **If `DB_URI` is provided and valid**: Uses PostgreSQL for persistent conversation storage
- **If `DB_URI` is not provided or invalid**: Gracefully falls back to MemorySaver (in-memory storage)
- **No connection spam**: Failed PostgreSQL connections are handled elegantly with a single warning message

## Examples

### Development (no persistence needed)
```bash
docker run -e OPENAI_API_KEY=sk-... -p 8000:8000 llamabot-backend
```

### Production with PostgreSQL
```bash
docker run \
  -e OPENAI_API_KEY=sk-... \
  -e DB_URI="postgresql://llamabot:secure_password@db.example.com:5432/llamabot_prod" \
  -p 8000:8000 \
  llamabot-backend
```

### With Docker Compose and PostgreSQL
```yaml
version: '3.8'
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: llamabot
      POSTGRES_USER: llamabot
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  backend:
    image: llamabot-backend
    environment:
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      DB_URI: "postgresql://llamabot:secure_password@postgres:5432/llamabot"
    ports:
      - "8000:8000"
    depends_on:
      - postgres

volumes:
  postgres_data:
``` 