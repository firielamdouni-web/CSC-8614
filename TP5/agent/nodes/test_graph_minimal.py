# TP5/test_graph_minimal.py
import uuid
import json

from TP5.load_test_emails import load_all_emails
from TP5.agent.state import AgentState
from TP5.agent.graph_minimal import build_graph

if __name__ == "__main__":
    emails = load_all_emails()
    indices = [0, 10]  # E01 (reply) and E11 (escalate) - adjust if needed
    app = build_graph()

    for idx in indices:
        e = emails[idx]
        state = AgentState(
            run_id=str(uuid.uuid4()),
            email_id=e["email_id"],
            subject=e["subject"],
            sender=e["from"],
            body=e["body"],
        )
        out = app.invoke(state)

        print("\n=== RUN ===", e["email_id"], "(index", idx, ")")
        print("=== DECISION ===")
        print(out["decision"].model_dump_json(indent=2))

        print("\n=== EVIDENCE ===")
        print(json.dumps([x.model_dump() for x in out["evidence"]], indent=2, ensure_ascii=False))

        print("\n=== DRAFT_V1 ===")
        print(out.get("draft_v1"))

        print("\n=== ACTIONS ===")
        print(out.get("actions"))

        print("\n=== FINAL ===")
        print("kind =", out.get("final_kind"))
        print(out.get("final_text"))
