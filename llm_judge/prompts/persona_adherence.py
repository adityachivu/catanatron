"""Prompt for the persona-adherence judge.

Scores how faithfully a Catan agent follows its persona system prompt across
three surfaces: private reasoning, concrete game decisions, and negotiation
messages.
"""

from __future__ import annotations

from llm_judge.records import AgentGameTrace
from llm_judge.schemas import PERSONA_JSON_HINT

SYSTEM_PROMPT = """\
You are a rigorous evaluator of AI agents playing Settlers of Catan. Your single
job is to judge how faithfully one agent stays IN CHARACTER relative to the
persona it was assigned.

You are given the persona's full system prompt — its identity, personality
matrix, short/long-term goals, opinions about other players, negotiation-phase
behaviour, and strategy-adaptation rules. Treat this as the ground-truth
specification of how the agent SHOULD think, decide, and talk.

Grade three surfaces independently, each 0-10:
  - reasoning_adherence:  Does the agent's private reasoning reflect the persona's
      priorities, risk tolerance, trust level, and voice? (e.g. a cautious,
      low-risk persona should not privately reason like a reckless gambler.)
  - decision_adherence:   Do its concrete in-game actions match the persona's
      strategy and stated rules? (e.g. a cooperative persona repeatedly robbing
      allies is off-persona; a "wall"/defensive persona over-extending is too.)
  - negotiation_adherence: Do its negotiation messages match the persona's
      prescribed negotiation behaviour, tone, and social rules? This surface
      matters most because negotiation is where persona is most visible to others.

For each surface: give a score, a rationale, verbatim supporting_examples where
the agent embodied the persona, and verbatim violations where it broke character.
Every example/violation MUST be a real quote from the provided material — never
invent quotes. If a surface has no evidence (e.g. no negotiations occurred), say
so in the rationale and score it neutrally (around 5) rather than penalising.

Key principle: the persona's traits OVERRIDE generic optimal play when they
conflict (the personas are explicitly told this). So an agent that plays the
mathematically "best" move but contradicts its persona's stated character is
LESS adherent, not more. Judge against the persona, not against optimal Catan.

overall_adherence_score should reflect the three surfaces holistically (not a
naive average) — weight persistent, defining traits more than one-off slips.

Output ONLY a single JSON object, no markdown, matching this shape exactly:
""" + PERSONA_JSON_HINT


def _format_reasoning(trace: AgentGameTrace) -> str:
    blocks = trace.reasoning_blocks()
    return "\n".join(blocks) if blocks else "(no private reasoning captured)"


def _format_decisions(trace: AgentGameTrace) -> str:
    if not trace.decisions:
        return "(no decisions captured)"
    return "\n".join(
        f"[turn {d.turn}] {d.action_type} — {d.action_detail or ''}".rstrip()
        for d in trace.decisions
    )


def _format_negotiations(trace: AgentGameTrace) -> str:
    if not trace.negotiations:
        return "(no negotiations in this game)"
    out: list[str] = []
    for i, neg in enumerate(trace.negotiations, 1):
        out.append(f"--- Negotiation {i} (turn {neg.turn}) ---")
        for m in neg.messages:
            tag = "  <-- AGENT UNDER EVAL" if m.is_agent_under_eval else ""
            out.append(f"    {m.sender}: {m.content}{tag}")
    return "\n".join(out)


def build_user_prompt(trace: AgentGameTrace) -> str:
    """Assemble the evidence packet for the persona-adherence judge."""
    return f"""\
Evaluate how well the agent below stayed in character.

GAME: {trace.game_id}
AGENT UNDER EVALUATION: color={trace.agent_color}, persona={trace.persona.name}
FINAL RESULT: victory_points={trace.final_victory_points}, won={trace.won}

================ PERSONA SPECIFICATION (the ground truth) ================
{trace.persona.system_prompt}

--- persona chat/negotiation style instructions ---
{trace.persona.chat_instructions or "(none provided)"}

================ PRIVATE REASONING (judge reasoning_adherence) ================
{_format_reasoning(trace)}

================ GAME DECISIONS (judge decision_adherence) ================
{_format_decisions(trace)}

================ NEGOTIATION MESSAGES (judge negotiation_adherence) ================
{_format_negotiations(trace)}

Return the JSON verdict now.
"""
