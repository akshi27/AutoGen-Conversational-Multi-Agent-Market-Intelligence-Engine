"""
Insight agent: conversational wrapper around the existing insight generation code.
Exposes insight_agent (Conversable) and insight_agent_handle_task(aggregated) which runs analysis.

This file by default uses the generate_insights() function.
"""
from __future__ import annotations
import os, traceback, time, json
from autogen.agentchat import ConversableAgent
from typing import Dict, Any

# Try to import the in-package generate_insights 
# The code below expects generate_insights(aggregated) or generate_insights(use_groq=False) to be available.

LLM_CONFIG = {"config_list": [{"model": "llama3", "api_key": os.getenv("GROQ_API_KEY","")}], "api_key_env": "GROQ_API_KEY"}

insight_agent = ConversableAgent(
    name="Insight_Generator_Agent",
    system_message="You are the Insight Generator Agent. You will be given aggregated scraped data (or filenames). Run sentiment/metrics computation and produce a structured analysis and a short human-readable summary. This agent will call local analytic code to compute metrics.",
    llm_config=LLM_CONFIG,
    human_input_mode="NEVER"
)

# Try importing an existing insights implementation inside agents (preferred)
try:
    # Here we attempt to import generate_insights from agents.insight_impl or tools.insight_generator
    from .insight_impl import generate_insights as generate_insights_impl  # if you create this
except Exception:
    generate_insights_impl = None

# Fallback: try to import root-level insight_generator
if generate_insights_impl is None:
    try:
        from .. import insight_generator as root_insight_module
        if hasattr(root_insight_module, "generate_insights"):
            generate_insights_impl = lambda aggregated: root_insight_module.generate_insights(use_groq=False)
    except Exception:
        generate_insights_impl = None

def insight_agent_handle_task(aggregated: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if generate_insights_impl:
            print("[InsightAgent] Running local generate_insights implementation...")
            analysis = (
                generate_insights_impl(aggregated)
                if generate_insights_impl.__code__.co_argcount == 1
                else generate_insights_impl(use_groq=False)
            )

            # ===== Save to /data/processed/analysis_report.json =====
            out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "analysis_report.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            print(f"[InsightAgent] ✅ Analysis saved to {out_path}")

            return {"status": "ok", "analysis": analysis, "path": out_path}

        else:
            # minimal builtin fallback analysis
            print("[InsightAgent] No local insight_impl found — running fallback analysis.")
            simple = {
                "note": "fallback analysis",
                "counts": {
                    k: (v.get("count") if isinstance(v, dict) else None)
                    for k, v in aggregated.get("results", {}).items()
                }
            }

            # ===== Save fallback to /data/processed/analysis_report.json =====
            out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "analysis_report.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(simple, f, indent=2, ensure_ascii=False)
            print(f"[InsightAgent] ✅ Fallback analysis saved to {out_path}")

            return {"status": "ok", "analysis": simple, "path": out_path}

    except Exception as e:
        traceback.print_exc()
        return {"status": "fail", "error": str(e)}
