"""
Entry point. Creates a conversational User Proxy agent and invokes the Orchestrator.
Run from project root with:
python -m agents.user_proxy_agent --mission "Quantify SoV for Atomberg smart fans"
"""
from __future__ import annotations
import argparse
import os
from autogen.agentchat import ConversableAgent, UserProxyAgent
from .planning_agent import create_plan
from .orchestrator_agent import OrchestratorWrapper

# Minimal LLM config placeholder — set to Groq LLM provider config and env variables
LLM_CONFIG = {"config_list": [{"model": "llama3", "api_key": os.getenv("GROQ_API_KEY", "")}], "api_key_env": "GROQ_API_KEY"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", type=str, default="Generate a comprehensive competitive market analysis for Atomberg smart fans")
    parser.add_argument("--no-insights", action="store_true")
    args = parser.parse_args()
    mission = args.mission

    # Create a simple UserProxyAgent (conversable)
    user_agent = ConversableAgent(
        name="UserProxyAgent",
        system_message="You are the user proxy that provides the mission to the Orchestrator.",
        llm_config=LLM_CONFIG,
        human_input_mode="NEVER"
    )

    print(f"[UserProxy] Mission: {mission}")
    plan = create_plan(mission)

    # Kick off the orchestrator wrapper which uses ConversableAgents under the hood
    orchestrator = OrchestratorWrapper(llm_config=LLM_CONFIG)
    orchestrator.run(plan, run_insights=not args.no_insights)

if __name__ == "__main__":
    main()
