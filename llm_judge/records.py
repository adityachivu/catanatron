"""Normalized INPUT schema for the judges.

The judges never touch raw game logs. They consume an :class:`AgentGameTrace`:
everything one persona agent did in one game, split into the three surfaces we
care about — private *reasoning*, concrete *decisions*, and *negotiation*
messages. Keeping this contract stable means the judges don't change when the
log format does; only ``llm_judge.loader`` (the log parser) has to map into it.

These field names mirror what the game already produces:
- action ``reasoning`` strings come from ``players/llm/output_types.py``
- negotiation messages (sender/content) come from ``players/llm/negotiation.py``
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Decision(BaseModel):
    """One concrete in-game action the agent took, with its private reasoning."""

    turn: Optional[int] = Field(default=None, description="Game turn number")
    phase: Optional[str] = Field(
        default=None, description="Game phase, e.g. 'build', 'move_robber', 'trade'"
    )
    action_type: str = Field(description="e.g. BUILD_CITY, MOVE_ROBBER, OFFER_TRADE")
    action_detail: Optional[str] = Field(
        default=None, description="Human-readable summary of what the action did"
    )
    reasoning: Optional[str] = Field(
        default=None, description="The agent's private step-by-step reasoning"
    )
    confidence: Optional[float] = Field(default=None)


class NegotiationMessage(BaseModel):
    """One message spoken during a negotiation session."""

    sender: str = Field(description="Color of the speaker, e.g. 'RED'")
    content: str = Field(description="The message text the agent broadcast")
    is_agent_under_eval: bool = Field(
        default=False,
        description="True if this message was sent by the agent being judged",
    )


class Negotiation(BaseModel):
    """One negotiation session the agent participated in.

    This is the primary conduit for agent-to-agent interaction, so it is the
    richest source of deception and of persona expression. We include the full
    transcript (all speakers) plus the agent's *private* finalization reasoning
    and the trade that actually resulted, so a judge can compare what the agent
    said publicly against what it privately intended and ultimately did.
    """

    turn: Optional[int] = Field(default=None)
    session_id: Optional[str] = Field(default=None)
    initiator: Optional[str] = Field(default=None, description="Color that started it")
    participants: List[str] = Field(default_factory=list)
    messages: List[NegotiationMessage] = Field(default_factory=list)
    # The agent-under-eval's private reasoning when finalizing the trade offer
    # (initiator only), and the resulting trade — for say-vs-do comparison.
    finalization_reasoning: Optional[str] = Field(default=None)
    resulting_trade: Optional[str] = Field(
        default=None, description="Human-readable resulting trade offer, or None"
    )
    trade_completed: Optional[bool] = Field(
        default=None, description="Whether a trade was ultimately executed"
    )


class Persona(BaseModel):
    """The persona definition the agent was given (from its YAML)."""

    name: str
    system_prompt: str = Field(description="Full persona system_prompt text")
    chat_instructions: str = Field(default="")


class AgentGameTrace(BaseModel):
    """Everything one agent did in one game — the unit a judge evaluates."""

    game_id: str
    agent_color: str = Field(description="The agent's player color, e.g. 'RED'")
    persona: Persona
    decisions: List[Decision] = Field(default_factory=list)
    negotiations: List[Negotiation] = Field(default_factory=list)
    # Optional final outcome, useful context for both judges.
    final_victory_points: Optional[int] = Field(default=None)
    won: Optional[bool] = Field(default=None)

    def reasoning_blocks(self) -> List[str]:
        """All private reasoning strings, labelled by turn/action, for prompting."""
        blocks: List[str] = []
        for d in self.decisions:
            if not d.reasoning:
                continue
            head = f"[turn {d.turn} | {d.action_type}]"
            blocks.append(f"{head} {d.reasoning}")
        return blocks
