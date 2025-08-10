"""
Orchestrator wrapper that wires ConversableAgents together and executes the pipeline.
This file creates ConversableAgent instances for Orchestrator, Scraper agents, and Insight agent,
and contains OrchestratorWrapper.run(plan) which coordinates conversation and actual tool execution.
"""
from __future__ import annotations
import os, json, time
from typing import Dict, Any
from autogen.agentchat import ConversableAgent
from .scraper_agent_google import google_agent_handle_task
from .scraper_agent_ecommerce import ecommerce_agent_handle_task
from .scraper_agent_social import social_agent_handle_task
from .insight_agent import insight_agent_handle_task

LLM_CONFIG = {"config_list": [{"model": "llama3", "api_key": os.getenv("GROQ_API_KEY","")}], "api_key_env": "GROQ_API_KEY"}

# Create a Conversable Orchestrator — this agent will *initiate chats* with other agents (so they "talk")
orchestrator_agent = ConversableAgent(
    name="Orchestrator_Agent",
    system_message="""You are the Orchestrator Agent. Your role is to manage the workflow:
1) Break the mission into three parallel scraping tasks (ecommerce+reddit, google+youtube, x).
2) Assign tasks to the specialist scraping agents (they will run tools and reply with structured results).
3) After all scrapers returned results, instruct the Insight Agent to analyze and create the final report.""",
    llm_config=LLM_CONFIG,
    human_input_mode="NEVER"
)

class OrchestratorWrapper:
    def __init__(self, llm_config: dict | None = None):
        # We already created orchestrator_agent above using LLM_CONFIG; keep that instance.
        self.agent = orchestrator_agent

    def run(self, plan: Dict[str, Any], run_insights: bool = True) -> Dict[str, Any]:
        print("[OrchestratorWrapper] Starting orchestration...")
        # 1) Have the orchestrator message (LLM) optionally summarize the plan (non-blocking).
        try:
            summary_prompt = f"Received plan: {json.dumps(plan, indent=2)}. Prepare a short plan summary in 1-2 sentences."
            # Kick off an internal chat to get a friendly summary (this is optional; graceful fallback)
            chat_res = self.agent.initiate_chat(
                recipient=None,  # no recipient just ask the agent to self-reflect
                message=summary_prompt,
                max_turns=1
            )
            print("[OrchestratorWrapper] Orchestrator summary (LLM):")
            print(chat_res)
        except Exception as e:
            print(f"[OrchestratorWrapper] Orchestrator LLM summary failed: {e}")

        # 2) Run scrapers — we'll both call the local tool (deterministic) and ask each ConversableAgent to 'comment' via LLM.
        results = {}

        # Ecommerce + Reddit
        ecommerce_cfg = plan["platforms"]["ecommerce"]
        print("[OrchestratorWrapper] Dispatching ecommerce task...")
        # LLM-generated task message — just for record
        try:
            msg = f"Task: Scrape ecommerce domains {ecommerce_cfg['domains']} for keywords/brands."
            _ = self.agent.initiate_chat(recipient=None, message=msg, max_turns=1)
        except Exception:
            pass
        ecommerce_res = ecommerce_agent_handle_task(ecommerce_cfg)
        results["ecommerce"] = ecommerce_res
        print("[OrchestratorWrapper] ecommerce done:", ecommerce_res.get("status"))

        # Google + YouTube
        google_cfg = plan["platforms"]["google_youtube"]
        print("[OrchestratorWrapper] Dispatching google_youtube task...")
        try:
            msg = f"Task: Scrape Google + YouTube for keywords: {google_cfg['keywords'][:5]} (truncated)."
            _ = self.agent.initiate_chat(recipient=None, message=msg, max_turns=1)
        except Exception:
            pass
        google_res = google_agent_handle_task(google_cfg)
        results["google_youtube"] = google_res
        print("[OrchestratorWrapper] google_youtube done:", google_res.get("status"))

        # Social (X)
        social_cfg = plan["platforms"]["social"]
        print("[OrchestratorWrapper] Dispatching social task...")
        try:
            msg = f"Task: Scrape X for brands and product types."
            _ = self.agent.initiate_chat(recipient=None, message=msg, max_turns=1)
        except Exception:
            pass
        social_res = social_agent_handle_task(social_cfg)
        results["social"] = social_res
        print("[OrchestratorWrapper] social done:", social_res.get("status"))

        # 3) Aggregate and call insight agent
        aggregated = {"plan": plan, "results": results}
        print("[OrchestratorWrapper] Calling Insight Agent for analysis...")
        # Ask orchestrator LLM to announce analysis
        try:
            ann = "All scraping complete. Please analyze results and produce a summary and recommendations."
            _ = self.agent.initiate_chat(recipient=None, message=ann, max_turns=1)
        except Exception:
            pass

        insight_res = None
        if run_insights:
            insight_res = insight_agent_handle_task(aggregated)
            results["insight"] = insight_res
            print("[OrchestratorWrapper] Insight generated.")

        # Save a conversation summary (structured)
        out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "output")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"pipeline_summary_{int(time.time())}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[OrchestratorWrapper] Pipeline summary saved to {out_path}")

        return results
