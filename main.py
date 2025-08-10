"""
Main entry point for the AutoGen-based Conversational Multi-Agentic AI Market Intelligence System.

This script runs the entire pipeline:
1. UserProxyAgent passes mission to PlannerAgent
2. PlannerAgent creates the execution plan
3. OrchestratorAgent coordinates scraping agents:
   - Ecommerce/Reddit Agent
   - Google/YouTube Agent
   - Social Media (X) Agent
4. InsightAgent processes all aggregated results into a strategic report
5. Final outputs saved in /data/processed/ and /data/output/
"""

import os
from agents.planning_agent import create_plan
from agents.orchestrator_agent import OrchestratorWrapper
from autogen.agentchat import ConversableAgent

# Mission text — you can change this for different runs
DEFAULT_MISSION = "Generate a comprehensive competitive market analysis for Atomberg smart fans"

# LLM configuration (Groq's llama-3.3-70b-versatile in this case)
LLM_CONFIG = {
    "config_list": [
        {
            "model": "llama-3.3-70b-versatile",
            "api_key": os.getenv("GROQ_API_KEY", "")
        }
    ],
    "api_key_env": "GROQ_API_KEY"
}

def run_pipeline(mission: str):
    """Run the full multi-agent pipeline for the given mission."""
    print("\n[MAIN] Starting Multi-Agent Market Intelligence Pipeline")
    print(f"[MAIN] Mission: {mission}")

    # Step 1: Create User Proxy Agent
    user_agent = ConversableAgent(
        name="UserProxyAgent",
        system_message="You are the user proxy that provides the mission to the Orchestrator.",
        llm_config=LLM_CONFIG,
        human_input_mode="NEVER"
    )

    # Step 2: Generate plan via Planner Agent
    print("[MAIN] Creating plan via PlannerAgent...")
    plan = create_plan(mission)
    print("[MAIN] Plan created.")

    # Step 3: Orchestrator runs scraping agents and Insight Agent
    orchestrator = OrchestratorWrapper(llm_config=LLM_CONFIG)
    results = orchestrator.run(plan, run_insights=True)

    print("\n[MAIN] Pipeline execution complete!")
    print(f"[MAIN] Processed insight report saved to /data/processed/analysis_report.json")
    print(f"[MAIN] Pipeline summary saved to /data/output/")

    return results

if __name__ == "__main__":
    run_pipeline(DEFAULT_MISSION)
