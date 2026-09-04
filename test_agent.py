from agents.deid_agent import DeidAgent
import os
from dotenv import load_dotenv

def test_agent_initialization():
    print("--- Testing Agent Initialization ---")
    try:
        agent = DeidAgent(provider="google")
        print("✅ Google (Gemini) Agent initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize Google Agent: {e}")

def test_agent_run():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️ Skipping Agent Run test: GOOGLE_API_KEY not found in .env")
        return

    print("\n--- Testing Agent PHI Detection ---")
    agent = DeidAgent()
    sample_text = "My name is John Doe and I live in New York. My phone is 555-0199."
    
    try:
        results = agent.run_deid(sample_text)
        print(f"Agent detected {len(results)} entities:")
        for res in results:
            print(f" - {res.get('text')} ({res.get('label')})")
        
        if len(results) > 0:
            print("✅ Agent run successful.")
        else:
            print("⚠️ Agent returned no results (check API key or model availability).")
    except Exception as e:
        print(f"❌ Agent run failed: {e}")

if __name__ == "__main__":
    test_agent_initialization()
    test_agent_run()
