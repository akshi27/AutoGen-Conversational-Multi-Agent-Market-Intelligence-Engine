"""
Planner agent (deterministic creation of plan). Provides a create_plan() function
and also exposes a ConversableAgent that can be used as we want LLM-driven planning.
"""
from __future__ import annotations
from typing import Dict, Any, List
from autogen.agentchat import ConversableAgent
import os

DEFAULT_BRANDS = [
    "atomberg","usha","havells","crompton","orient",
    "bajaj","luminous","lg","vguard","honeywell","polycab"
]
DEFAULT_PRODUCT_TYPES = ["smart fan","bldc fan","smart ceiling fan"]
DEFAULT_REDDIT_SUBS = [["smarthome", "homeautomation", "india", "mumbai", "Chennai", "Frugal_Ind", "Kerala", "delhi", "kolkata"]]

LLM_CONFIG = {"config_list": [{"model": "llama3", "api_key": os.getenv("GROQ_API_KEY", "")}], "api_key_env": "GROQ_API_KEY"}

planner_agent = ConversableAgent(
    name="PlannerAgent",
    system_message="""You are the Planner Agent. You will be given a high level mission and you should
produce a structured plan that lists platforms to scrape, keywords, brand lists, domains, and per-platform constraints.
Return a JSON-serializable plan only when asked.""",
    llm_config=LLM_CONFIG,
    human_input_mode="NEVER"
)


def create_plan(mission_text: str, accept_cheats: bool = True) -> Dict[str, Any]:
    """Deterministic planner used by orchestrator — you can replace or augment with planner_agent interaction."""
    keywords = [
        "atomberg smart fan", "atomberg bldc fan", "smart fan review",
        "bldc fan review", "atomberg vs orient", "atomberg vs crompton"
    ]
    platforms = {
        "ecommerce": {
            "domains": ["amazon", "reliancedigital", "croma"],
            "brands": DEFAULT_BRANDS,
            "product_types": DEFAULT_PRODUCT_TYPES,
            "reddit_subs": DEFAULT_REDDIT_SUBS,
            "num_results_per_domain": 10
        },
        "google_youtube": {
            "keywords": keywords,
            "num_results": 10
        },
        "social": {
            "platform": "x",
            "brands": DEFAULT_BRANDS,
            "product_types": DEFAULT_PRODUCT_TYPES,
            "max_posts_to_scrape": 40,
            "max_replies_per_post": 4
        }
    }
    return {"mission": mission_text, "keywords": keywords, "platforms": platforms, "accept_cheats": accept_cheats}
