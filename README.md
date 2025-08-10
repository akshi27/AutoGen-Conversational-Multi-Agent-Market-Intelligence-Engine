# 🧠 Multi-Agent Conversational AI for Market Intelligence

This project implements an **AutoGen-powered conversational multi-agent architecture** for scraping, analyzing, and reporting competitive market intelligence on brands/products such as **Atomberg smart fans**.

The system is **conversational**, each specialized AI agent communicates with others, passing tasks in sequence, and the **Orchestrator Agent** manages the workflow.

---

## ⚙️ Installation

### 1️⃣ Clone this repository

`git clone https://github.com/akshi27/Marketing-Analyzer-AI-Agent.git`
`cd Marketing-Analyzer-AI-Agent`

### 2️⃣ Create a virtual environment

`python -m venv venv`
`source venv/bin/activate`  # Linux/Mac
`venv\Scripts\activate`    # Windows

### 3️⃣ Install dependencies

`pip install -r requirements.txt`

---

## 🔑 Environment Variables
Create a .env file in the project root with:

### API Keys

`SERP_API_KEY=your_serpapi_key`
`REDDIT_CLIENT_ID=your_reddit_client_id`
`REDDIT_CLIENT_SECRET=your_reddit_client_secret`
`REDDIT_USER_AGENT=your_user_agent`
`X_BEARER_TOKEN=your_x_bearer_token`
`GROQ_API_KEY=your_groq_api_key`

---

## 🛠 How the Agents Work

1. `UserProxyAgent`
Acts as the interface for you.

You can ask something like "Generate a competitive market analysis of Atomberg smart fans."

2. `PlanningAgent`
Breaks the request into scraping sub-tasks:

Google + YouTube

E-commerce + Reddit

X (Twitter)

3. `Scraper Agents`
scraper_agent_google → Calls tools/google_youtube_scraper_tool.py

scraper_agent_ecommerce → Calls tools/ecommerce_scraper_tool.py

scraper_agent_social → Calls tools/x_scraper_tool.py

All results are saved to /data/raw/ as JSON.

4. `InsightAgent`
Reads all raw JSON files.

Runs generate_insights() from insight_impl.py.

Produces metrics: Share of Voice, Share of Positive Voice, sentiment scores, etc.

Saves `/data/processed/analysis_report.json`.

5. `Orchestration`
OrchestratorAgent passes control between agents until the final report is ready.

---

## ▶️ Running the System
From the project root:

`python main.py`

Example flow:

`UserProxyAgent` sends your mission to the `OrchestratorAgent`.

Orchestrator assigns scraping jobs in parallel.

`Scraper agents` save JSON files under `/data/raw/`.

`InsightAgent` analyzes everything.

The final report is saved at:

`data/processed/analysis_report.json`
