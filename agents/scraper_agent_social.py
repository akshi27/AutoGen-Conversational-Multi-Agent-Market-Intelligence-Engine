"""
Social (X) scraping agent.
Exposes social_agent (Conversable) and social_agent_handle_task(cfg).
Tool expectation:
tools/x_scraper_tool.py -> scrape_x_tweets(brands, product_types, max_posts_to_scrape, max_replies_per_post, save)
"""
from __future__ import annotations
import os, time, traceback
from autogen.agentchat import ConversableAgent
from typing import Dict, Any
from ..tools import x_scraper_tool as xs

LLM_CONFIG = {"config_list": [{"model": "llama3", "api_key": os.getenv("GROQ_API_KEY","")}], "api_key_env": "GROQ_API_KEY"}

social_agent = ConversableAgent(
    name="Social_Scraper_Agent",
    system_message="You are the X(Twitter) scraping specialist. You will fetch posts and replies relevant to brands/product types. This agent will call the x_scraper_tool to run scrapes.",
    llm_config=LLM_CONFIG,
    human_input_mode="NEVER"
)

def social_agent_handle_task(cfg: Dict[str, Any]) -> Dict[str, Any]:
    brands = cfg.get("brands", [])
    product_types = cfg.get("product_types", [])
    max_posts = cfg.get("max_posts_to_scrape", 40)
    max_replies = cfg.get("max_replies_per_post", 4)
    try:
        t0 = time.time()
        data = xs.scrape_x_tweets(brands, product_types, max_posts_to_scrape=max_posts, max_replies_per_post=max_replies, save=True)
        return {"status": "ok", "count": len(data) if isinstance(data, list) else 0, "data": data, "meta": {"elapsed_s": time.time() - t0}}
    except Exception as e:
        traceback.print_exc()
        return {"status": "fail", "error": str(e)}
