import os
import json
from dotenv import load_dotenv

# ====== Load env vars ======
load_dotenv()

# ====== Import agents ======
from agents.orchestrator_agent import orchestrator_agent
from agents.planning_agent import planning_agent
from agents.scraper_agent_ecommerce import ecommerce_scraper_agent
from agents.scraper_agent_google import google_scraper_agent
from agents.scraper_agent_social import social_scraper_agent
from agents.insight_agent import insight_agent

# ====== Mission ======
MISSION = """
Collect data for smart fan and BLDC fan brands:
1. E-commerce platforms
2. Google & YouTube
3. Social media
Then process them to generate actionable market insights.
"""

# ====== File save helpers ======
def save_json(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved: {filepath}")

# ====== Main workflow ======
def main():
    print("\n🚀 Starting Multi-Agent Data Collection Pipeline...\n")

    # Step 1: Orchestrator starts mission
    orchestrator_agent.receive_message(MISSION, sender="human")

    # Step 2: Planning agent creates a plan
    plan = planning_agent.create_plan(MISSION)
    print("📝 Plan created:", plan)

    # Step 3: Run scrapers
    print("\n🔍 Running E-commerce scraper...")
    ecommerce_data = ecommerce_scraper_agent.run()
    save_json(ecommerce_data, "data/raw/ecommerce_data.json")

    print("\n🔍 Running Google/YouTube scraper...")
    google_data = google_scraper_agent.run()
    save_json(google_data, "data/raw/google_youtube_data.json")

    print("\n🔍 Running Social Media scraper...")
    social_data = social_scraper_agent.run()
    save_json(social_data, "data/raw/social_data.json")

    # Step 4: Generate insights
    print("\n🧠 Generating insights...")
    insights = insight_agent.run({
        "ecommerce": ecommerce_data,
        "google_youtube": google_data,
        "social": social_data
    })
    save_json(insights, "data/insights/final_report.json")

    print("\n✅ Pipeline completed. All files saved in 'data/raw/' and 'data/insights/'\n")

if __name__ == "__main__":
    main()
