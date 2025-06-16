<!-- README.md – modern & user-friendly -->
<div align="center">

<!-- Logo (scaled) -->
<img src="https://service-jobs-images.s3.us-east-2.amazonaws.com/7rl98t1weu387r43il97h6ipk1l7" width="160" alt="LlamaBot logo">

# **LlamaBot**

_The open-source AI coding agent that chats, writes, and live-previews your web apps._

[![Live Site](https://img.shields.io/badge/Visit-LlamaPress.ai-brightgreen?style=for-the-badge&logo=safari)](https://llamapress.ai)
[![LLM Prompts](https://img.shields.io/badge/LangSmith-Prompts-blue?style=for-the-badge&logo=langchain)](https://smith.langchain.com/hub/llamabot)
[![MIT License](https://img.shields.io/github/license/KodyKendall/LlamaBot?style=for-the-badge)](LICENSE)
[![Discord](https://img.shields.io/badge/Join-Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/HtVVSxrK)

<!-- Demo GIF -->
<br>
<img src="https://llamapress-ai-image-uploads.s3.us-west-2.amazonaws.com/d7zial72abpkblr9n6lf8cov8lp4" width="600" alt="LlamaBot live demo">

</div>

---

## ✨ What is LlamaBot?

**LlamaBot** is an AI agent (built on **LangGraph + FastAPI**) that _converses_ with you while writing and modifying **HTML / CSS / JavaScript** in real time. Available in two versions:

- 🚀 **Modern Version**: React + TypeScript frontend with FastAPI backend
- 🎯 **Simple Version**: Pure HTML/CSS/JS with FastAPI backend

Perfect for:
- 🎮 Mini-games (Canvas-based)  
- 🖥️ Portfolio & static sites  
- 💼 Business landing pages  
- 🧮 Interactive calculators & quote tools  
- …and a growing list of web-dev use cases!

---

## 🚀 Key Features

| Feature                | Description                                   |
|------------------------|-----------------------------------------------|
| **Chat-driven coding** | Type plain English; get instant code          |
| **Live preview**       | See changes render side-by-side as you chat   |
| **Hot-reload**         | Edits appear without refresh                  |
| **Multi-file support** | Handles HTML, CSS, and JS seamlessly          |
| **Modern Architecture**| React + TypeScript frontend (optional)        |
| **Streaming responses**| Real-time AI responses with WebSocket/SSE     |
| **Thread management**  | Persistent conversation history               |
| **Zero-config start**  | Clone → `pip install` → `uvicorn` → create    |

---

## 🏗️ Project Structure

```
LlamaBot/
├── README.md                       # Project documentation
├── LICENSE                         # License
├── requirements.txt                # Python dependencies
├── langgraph.json                  # LangGraph configuration
├── chat_app.log                    # Application logs (root level)
├── page.html                       # Generated content display (root level)
│
├── backend/                        # Backend application directory
│   ├── app.py                      # FastAPI application with streaming endpoints
│   ├── chat_app.log                # Backend-specific application logs
│   ├── chat.html                   # Simple chat interface UI
│   ├── home.html                   # Landing page
│   ├── page.html                   # Generated content display
│   ├── conversations.html          # Conversation history interface
│   │
│   └── agents/                     # AI agent implementations
│       ├── __init__.py
│       ├── base_agent.py           # Base agent functionality
│       ├── react_agent/
│       │   └── nodes.py            # ReACT workflow implementation
│       ├── write_html_agent/       # Archived for educational purposes. v1.
│       │   ├── nodes.py            # Main workflow orchestration
│       │   ├── state.py            # Agent state definition
│       │   ├── design_and_plan.py  # Planning and design logic
│       │   ├── write_html_code.py  # HTML/CSS/JS generation in 1 file.
│       │   ├── respond_naturally.py # Natural language responses
│       │   └── route_initial_user_message.py  # Initial message routing
│       └── utils/                  # Shared utilities
│
├── frontend/                       # React + TypeScript frontend (Optional)
│   ├── src/
│   │   ├── components/             # React components
│   │   ├── stores/                 # Zustand state management
│   │   ├── services/               # API service layer
│   │   ├── types/                  # TypeScript type definitions
│   │   ├── App.tsx                 # Main App component
│   │   └── main.tsx                # React entry point
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── tailwind.config.js
│
├── assets/                         # Static assets (CSS, JS, images)
├── docs/                           # Documentation
├── examples/                       # Example files and demos
├── experiments/                    # Experimental features
├── mcp/                           # MCP (Model Context Protocol) integration
└── venv/                          # Python virtual environment
```

---

## ⚡ Quick Start

### Option 1: Simple Version (HTML + FastAPI)

```bash
# 1 — Clone
git clone https://github.com/KodyKendall/LlamaBot.git
cd LlamaBot

# 2 — Create & activate virtual env (recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3 — Install deps
pip install -r requirements.txt

# 4 — Add your secrets
cat <<EOF > .env
OPENAI_API_KEY=sk-...
LANGSMITH_API_KEY=your_langsmith_api_key
EOF

# 5 (optional) - Set up PostgreSQL database tables & connection string.
psql -c "
CREATE DATABASE langgraph_dev;
CREATE USER langgraph_user WITH PASSWORD 'langgraph_pass';
GRANT ALL PRIVILEGES ON DATABASE langgraph_dev TO langgraph_user;
\c langgraph_dev;
GRANT USAGE ON SCHEMA public TO langgraph_user;
GRANT CREATE ON SCHEMA public TO langgraph_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO langgraph_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO langgraph_user;
"

# 6 - (Optional): if you want to use Playwright in your workflows:, npm install playwright
npm init playwright@latest

# 7 — Navigate to backend and run it
cd backend
uvicorn app:app --reload
```

Browse to http://localhost:8000/chat and start building!

### Option 2: Modern Version (React + TypeScript + FastAPI)

**Prerequisites:**
- Node.js 18+ and npm
- Python 3.8+
- PostgreSQL (optional, for persistent storage)

**Backend Setup:**
```bash
# Follow steps 1-5 from Option 1 above, then:
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Frontend Setup:**
```bash
# In a new terminal, navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Create environment variables (optional)
cat <<EOF > .env
VITE_API_URL=http://localhost:8000
EOF

# Start the development server
npm run dev
```

Open your browser to `http://localhost:3001` for the modern React interface!

---

## 🛠️ How It Works

1. **Chat input** → parsed by LlamaBot
2. **Intent detection** → reply naturally or generate code
3. **LangGraph workflow** produces / edits code snippets
4. **WebSocket/SSE push** updates the preview pane instantly
5. All orchestration logic lives in `agents/` folder

## 🧠 Agent Architecture
<div align="center">
<img src="https://www.kodykendall.com/wp-content/uploads/2025/06/Screenshot-2025-06-01-at-1.32.14%E2%80%AFPM.png" width="400" alt="LlamaBot Agent Architecture">
</div>

---

## 🛠️ Technology Stack

### Backend (Core)
- **FastAPI** - Modern Python web framework
- **LangGraph** - AI agent orchestration
- **LangChain** - LLM framework
- **OpenAI** - Language models
- **PostgreSQL** - Optional persistent storage
- **WebSocket/SSE** - Real-time streaming

### Frontend Options

**Simple Version:**
- Pure HTML/CSS/JavaScript
- Minimal dependencies
- Direct FastAPI integration

**Modern Version:**
- **React 18** - Modern React with hooks
- **TypeScript** - Type safety and better developer experience
- **Vite** - Fast build tool and development server
- **Tailwind CSS** - Utility-first CSS framework
- **Zustand** - Lightweight state management
- **Lucide React** - Beautiful icons
- **Axios** - HTTP client for API calls

---

## 🎯 Key Features by Version

### Simple Version
- **Zero-config setup** - Just HTML files and Python
- **Lightweight** - Minimal dependencies
- **Educational** - Easy to understand and modify
- **Fast deployment** - Single FastAPI server

### Modern Version
- **Component-based design** with proper separation of concerns
- **TypeScript** for type safety and better IDE support
- **State management** with Zustand for predictable state updates
- **Real-time streaming** with proper error handling
- **Responsive design** with Tailwind CSS
- **Hot reload** for both frontend and backend
- **Production-ready** architecture

---

## 📡 API Endpoints

- `GET /` - Serve home page
- `GET /chat` - Serve chat interface (simple version)
- `GET /page` - Get generated HTML page
- `GET /assets/*` - Serve static assets
- `GET /threads` - Get all conversation threads (modern version)
- `GET /chat-history/{thread_id}` - Get specific conversation history
- `POST /chat-message` - Send a message (streaming response)

---

## 💡 Example Prompts to Try

| Goal | Prompt |
|------|--------|
| Build Snake game | "Create a simple Snake game with arrow-key controls." |
| Personal portfolio | "Generate a minimal personal site with About, Projects, and Contact sections." |
| SaaS landing page | "Design a SaaS landing page hero with CTA and pricing table." |
| Mortgage calculator | "Make a responsive mortgage calculator that outputs monthly payment." |

---

## 🔧 Development

### Simple Version Development
```bash
cd backend
uvicorn app:app --reload  # Start with auto-reload
```

### Modern Version Development

**Frontend:**
```bash
cd frontend
npm run dev        # Start development server
npm run build      # Build for production
npm run lint       # Run ESLint
npm run preview    # Preview production build
```

**Backend:**
```bash
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

---

## 🔒 Environment Variables

### Backend (.env in root directory)
```env
OPENAI_API_KEY=your_openai_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
DB_URI=postgresql://user:password@localhost:5432/llamabot  # Optional
```

### Modern Frontend (.env in frontend/)
```env
VITE_API_URL=http://localhost:8000
```

---

## 🚢 Deployment

### Simple Version
Deploy the FastAPI backend to any Python hosting service:
- Railway, Render, or Heroku
- AWS EC2 or Google Cloud Run
- DigitalOcean App Platform

### Modern Version

**Using Docker (Recommended):**

1. **Backend Dockerfile:**
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

2. **Frontend build:**
   ```bash
   cd frontend
   npm run build
   # Serve the dist/ folder with a web server
   ```

**Cloud Platforms:**
- **Backend**: Railway, Render, or Heroku
- **Frontend**: Vercel, Netlify, or AWS S3/CloudFront

---

## 🎨 Customization

### Simple Version
- Modify `chat.html`, `home.html`, and `page.html` directly
- Update styles in the HTML files or create separate CSS files
- Customize the FastAPI routes in `app.py`

### Modern Version
- **Styling**: Customize in `frontend/tailwind.config.js` and `frontend/src/index.css`
- **Components**: Modify React components in `frontend/src/components/`
- **State Management**: Update Zustand store in `frontend/src/stores/chatStore.ts`

---

## 🔗 Key Resources

- **View the key prompts this agent uses**: [![LLM Prompts](https://img.shields.io/badge/LangSmith-Prompts-blue?style=for-the-badge&logo=langchain)](https://smith.langchain.com/hub/llamabot)
- **Live Demo**: [LlamaPress.ai](https://llamapress.ai)
- **Discord Community**: [Join Discord](https://discord.gg/HtVVSxrK)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/awesome`
3. Make your changes
4. Add tests if applicable
5. Commit, push, and open a PR 🎉

---

## 📜 License

The LlamaBot backend (LangGraph orchestration layer) is licensed under the [GNU Affero General Public License v3 (AGPLv3)](./LICENSE).

This ensures that anyone who uses this code to serve a network-accessible product must also make their source code available under the same license.

If you’d like to use LlamaBot in a closed-source, commercial, or proprietary product, contact us for a commercial license:

📧 kody@llamapress.ai

---

<div align="center">
Made with ❤️ in San Francisco, by Kody Kendall — inspired by the next wave of AI code-gen tools.
</div>