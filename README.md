<div align="center">

<img src="https://service-jobs-images.s3.us-east-2.amazonaws.com/7rl98t1weu387r43il97h6ipk1l7" width="160" alt="LlamaBot logo">

# **LlamaBot**

*The open-source AI agent that writes, automates, and operates inside real software.*

[![Live Site](https://img.shields.io/badge/Visit-LlamaPress.ai-brightgreen?style=for-the-badge\&logo=safari)](https://llamapress.ai)
[![LLM Prompts](https://img.shields.io/badge/LangSmith-Prompts-blue?style=for-the-badge\&logo=langchain)](https://smith.langchain.com/hub/llamabot)
[![MIT License](https://img.shields.io/github/license/KodyKendall/LlamaBot?style=for-the-badge)](LICENSE)
[![Discord](https://img.shields.io/badge/Join-Discord-7289DA?style=for-the-badge\&logo=discord\&logoColor=white)](https://discord.gg/HtVVSxrK)

<img src="https://llamapress-ai-image-uploads.s3.us-west-2.amazonaws.com/d7zial72abpkblr9n6lf8cov8lp4" width="600" alt="LlamaBot live demo">

</div>

---

## ✨ What is LlamaBot?

LlamaBot is an open-source AI agent built on **LangGraph** and **FastAPI**. It helps you:

* 💬 Chat to write HTML/CSS/JS
* ⚙️ Operate directly inside your real software (like Rails apps)
* 🔁 Automate business logic using existing models, services, and routes

There are two primary ways to use LlamaBot:

1. **🧪 Try the interactive HTML/JS agent** – see the magic in your browser
2. **⚙️ Embed it in your backend app (e.g. Rails)** – let it run real workflows

---

## 🚀 Option 1: Try the HTML/JS Agent (No setup)

Perfect for:

* Mini-games (Canvas-based)
* Static websites
* Marketing pages
* Interactive calculators

```bash
# Run via Docker
# (only requirement: Docker and your OpenAI key)
docker run -e OPENAI_API_KEY=sk-... -p 8000:8000 kody06/llamabot-backend
```

Then open:

```
http://localhost:8000/chat
```

---

## 🧩 Option 2: Embed the Agent in Your App (Rails, Django, Laravel, etc.) 
Rails is our primary use case right now, others coming soon!

LlamaBot also works as a **CodeAct-style embedded agent**. That means it can:

* Operate inside your existing web app
* Query your real ActiveRecord/ORM models
* Trigger existing services and codepaths
* Send emails, enqueue jobs, and more
* Learn about your app and remember key workflows to trigger

**Demo use cases:**

* Refund a user and notify them by SMS
* Generate a weekly revenue report
* Queue 100 Sidekiq jobs from natural language

**Add Rails adapter to your app:**

* [`llama_bot_rails`](https://github.com/kodykendall/llama_bot_rails) gem
* Point to the same backend (via Docker)
* Whitelist safe routes and tools using your existing RBAC (Devise, Pundit, etc.)

```rb
# config/initializers/llama_bot.rb
LlamaBotRails.configure do |config|
  config.api_base_url = ENV["LLAMABOT_BACKEND_URL"]
  config.allowed_routes = {
    "send_sms" => { verb: :post, path: "/agent/users/:id/send_sms" },
    "refund_order" => { verb: :post, path: "/agent/orders/:id/refund" }
  }
end
```

---

## 🧠 Agent Architecture

* Built on **LangGraph** (multi-step agent workflows)
* Streaming responses via **FastAPI + WebSocket**
* Memory saved and scoped to conversation / session
* Can call external tools via APIs or internal app methods (via HTTP calls)

---

## 📦 Project Structure

```
LlamaBot/
├── backend/
│   ├── app.py             # FastAPI app with WebSocket + API routes
│   ├── chat.html          # Chat interface UI
│   ├── page.html          # Rendered result display
│   ├── agents/            # LangGraph agent logic
│   └── ...                # Utility code, workflows, memory, etc.
├── Dockerfile             # To run the backend agent anywhere
├── requirements.txt       # Python deps
└── README.md
```

---

## 🔧 Development (Local)

```bash
# Clone & start backend locally
git clone https://github.com/KodyKendall/LlamaBot.git
cd LlamaBot/backend
python -m venv venv && source venv/bin/activate
pip install -r ../requirements.txt
uvicorn app:app --reload
```

Browse to: [http://localhost:8000/chat](http://localhost:8000/chat)

---

## 🤝 Contributing

We welcome PRs, feedback, and ideas! Open an issue or drop into our [Discord](https://discord.gg/HtVVSxrK).

---

## 📜 License

LlamaBot is AGPLv3 open-source. For commercial licensing, contact: **[kody@llamapress.ai](mailto:kody@llamapress.ai)**

<div align="center">
Made with ❤️ in San Francisco — inspired by the next wave of AI code-gen tools.
</div>
