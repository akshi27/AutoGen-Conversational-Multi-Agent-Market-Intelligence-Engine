"""
Google+YouTube scraper agent. This file exposes:
- google_agent: a ConversableAgent used for conversation/LLM comments (optional)
- google_agent_handle_task(cfg): deterministic wrapper that calls your tools and returns a structured dict.

Tool expectation:
tools/google_youtube_scraper_tool.py -> scrape_keywords(keywords, num_results, save=True)
"""
from __future__ import annotations
import os, traceback
from autogen.agentchat import ConversableAgent
from typing import Dict, Any
from ..tools import google_youtube_scraper_tool as gys
import time

LLM_CONFIG = {"config_list": [{"model": "llama3", "api_key": os.getenv("GROQ_API_KEY","")}], "api_key_env": "GROQ_API_KEY"}

google_agent = ConversableAgent(
    name="GoogleYt_Scraper_Agent",
    system_message="You are the Google+YouTube scraping specialist. When given a config you will fetch SERP and transcripts (this agent also runs the real scraping tool).",
    llm_config=LLM_CONFIG,
    human_input_mode="NEVER"
)

def google_agent_handle_task(cfg: Dict[str, Any]) -> Dict[str, Any]:
    keywords = cfg.get("keywords", [])
    num_results = cfg.get("num_results", 10)
    try:
        start = time.time()
        data = gys.scrape_keywords(keywords, num_results=num_results, save=True)
        elapsed = time.time() - start
        # Return a structured response
        return {"status": "ok", "count": len(data) if isinstance(data, list) else 0, "data": data, "meta": {"elapsed_s": elapsed}}
    except Exception as e:
        traceback.print_exc()
        return {"status": "fail", "error": str(e)}
