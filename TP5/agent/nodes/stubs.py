# TP5/agent/nodes/stubs.py
from TP5.agent.logger import log_event
from TP5.agent.state import AgentState

def stub_reply(state: AgentState) -> AgentState:
    log_event(state.run_id, "node_start", {"node": "stub_reply"})
    state.draft_v1 = "[STUB] Réponse générée avec RAG (à implémenter)"
    log_event(state.run_id, "node_end", {"node": "stub_reply", "status": "ok"})
    return state

def stub_ask_clarification(state: AgentState) -> AgentState:
    log_event(state.run_id, "node_start", {"node": "stub_ask_clarification"})
    state.draft_v1 = "[STUB] Pouvez-vous préciser votre demande ? De quel type de dossier s'agit-il ?"
    log_event(state.run_id, "node_end", {"node": "stub_ask_clarification", "status": "ok"})
    return state

def stub_escalate(state: AgentState) -> AgentState:
    log_event(state.run_id, "node_start", {"node": "stub_escalate"})
    state.actions.append({
        "type": "handoff_human",
        "summary": f"Email escaladé: {state.subject} - nécessite intervention humaine",
    })
    log_event(state.run_id, "node_end", {"node": "stub_escalate", "status": "ok"})
    return state

def stub_ignore(state: AgentState) -> AgentState:
    log_event(state.run_id, "node_start", {"node": "stub_ignore"})
    state.actions.append({
        "type": "ignore",
        "reason": "Email hors périmètre ou suspect (spam/phishing)",
    })
    log_event(state.run_id, "node_end", {"node": "stub_ignore", "status": "ok"})
    return state
