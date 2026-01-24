import re
from typing import List
from TP5.agent.logger import log_event
from TP5.agent.state import AgentState

RE_CIT = re.compile(r"\[(doc_\d+)\]")

def _extract_citations(text: str) -> List[str]:
    return sorted(set(RE_CIT.findall(text or "")))


def finalize(state: AgentState) -> AgentState:
    log_event(state.run_id, "node_start", {"node": "finalize"})
    
    if not state.budget.can_step():
        log_event(state.run_id, "node_end", {"node": "finalize", "status": "budget_exceeded"})
        return state
    state.budget.steps_used += 1
    
    intent = state.decision.intent

    if intent == "reply":
        cits = _extract_citations(state.draft_v1)
        state.final_kind = "reply"
        if cits:
            state.final_text = state.draft_v1.strip() + "\n\nSources: " + " ".join(f"[{c}]" for c in cits)
        else:
            # fallback reply when draft empty (French)
            state.final_text = state.draft_v1.strip() or "Je ne dispose pas d'informations suffisantes pour rédiger une réponse fiable. Pouvez-vous fournir plus de détails ?"

    elif intent == "ask_clarification":
        state.final_kind = "clarification"
        if state.draft_v1.strip():
            state.final_text = state.draft_v1.strip()
        else:
            # fallback clarification questions (French)
            state.final_text = (
                "Pouvez-vous préciser votre demande ? Par exemple : 1) quelle(s) UE concernée(s) ; 2) quelles dates ; 3) quels documents justificatifs vous fournissez."
            )

    elif intent == "escalate":
        state.final_kind = "handoff"
        # generate a concise French summary (first 200 chars, cleaned)
        raw = state.draft_v1.strip() or state.body.strip()
        summary = " ".join(raw.split())[:200]
        if len(raw) > 200:
            summary = summary.rstrip() + "..."
        packet = {
            "type": "handoff_packet",
            "run_id": state.run_id,
            "email_id": state.email_id,
            "summary": summary,
            "evidence_ids": [d.doc_id for d in state.evidence],
        }
        state.actions.append(packet)
        state.final_text = "Votre demande nécessite une validation humaine. Un dossier de transmission a été créé."

    else:  # ignore
        state.final_kind = "ignore"
        state.final_text = state.draft_v1.strip() or "Aucune action requise."

    log_event(state.run_id, "node_end", {"node": "finalize", "status": "ok", "final_kind": state.final_kind})
    return state
