# TP5/test_router.py
import uuid
import sys
import os

# Add parent directory to path
parent_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

from load_test_emails import load_all_emails
from agent.state import AgentState
from agent.nodes.classify_email import classify_email

if __name__ == "__main__":
    emails = load_all_emails()
    e = emails[0]
    
    state = AgentState(
        run_id=str(uuid.uuid4()),
        email_id=e["email_id"],
        subject=e["subject"],
        sender=e["from"],
        body=e["body"],
    )
    
    state = classify_email(state)
    print(state.decision.model_dump_json(indent=2))
