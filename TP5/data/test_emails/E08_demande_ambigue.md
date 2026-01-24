---
email_id: E08
from: "Alexandre Dubois <alex.dubois@telecom-sudparis.eu>"
date: "2026-01-16"
subject: "Question sur mon dossier"
---

CORPS:
<<<
Bonjour,

J'ai un problème avec mon dossier. Est-ce que vous pouvez m'aider ?

Merci,
Alexandre
>>>

ATTENDU:
- intent: ask_clarification
- points clés: demande trop vague, besoin de précisions (quel type de dossier ? inscription, stage, admission ?)
