"""
Ecommerce + Reddit scraping agent.
Exposes ecommerce_agent (Conversable) and ecommerce_agent_handle_task(cfg) which calls the tool.
Tool expectation:
tools/ecommerce_scraper_tool.py -> run_all_scrapers(brands, product_types, reddit_subs)
"""
from __future__ import annotations
import os, traceback, time
from autogen.agentchat import ConversableAgent
from typing import Dict, Any
from ..tools import ecommerce_scraper_tool as ecs

LLM_CONFIG = {"config_list": [{"model": "llama3", "api_key": os.getenv("GROQ_API_KEY","")}], "api_key_env": "GROQ_API_KEY"}

ecommerce_agent = ConversableAgent(
    name="Ecommerce_Scraper_Agent",
    system_message="You are the Ecommerce+Reddit scraper agent. You will scrape product listings, star ratings and reddit discussions. This agent will call the local ecommerce_scraper_tool to execute scrapes.",
    llm_config=LLM_CONFIG,
    human_input_mode="NEVER"
)

def ecommerce_agent_handle_task(cfg: Dict[str, Any]) -> Dict[str, Any]:
    brands = cfg.get("brands", [])
    product_types = cfg.get("product_types", [])
    reddit_subs = cfg.get("reddit_subs", [])
    try:
        t0 = time.time()
        data = ecs.run_all_scrapers(brands, product_types, reddit_subs)
        return {"status": "ok", "count": len(data) if isinstance(data, list) else 0, "data": data, "meta": {"elapsed_s": time.time() - t0}}
    except Exception as e:
        traceback.print_exc()
        return {"status": "fail", "error": str(e)}
