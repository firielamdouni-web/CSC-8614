import sys
import os

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from agent.logger import log_event

if __name__ == "__main__":
    run_id = "TEST_RUN"
    log_event(run_id, "node_start", {"node": "classify_email"})
    log_event(run_id, "node_end", {"node": "classify_email", "status": "ok"})
    print("OK - check runs/TEST_RUN.jsonl")
