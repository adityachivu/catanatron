"""
Negotiation Protocol - Message structures and conversation primitives.

This module defines the protocol for natural language negotiation:
- NegotiationMessage: A message exchanged between players
- NegotiationIntent: The purpose of a message
- TradeProposal: A structured trade offer
- NegotiationSession: A complete negotiation conversation

The protocol is designed to be:
1. Flexible enough for natural language
2. Structured enough for programmatic handling
3. Compatible with the existing Catanatron trade system
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
from datetime import datetime

from catanatron.models.player import Color
from catanatron.models.enums import RESOURCES


class NegotiationIntent(str, Enum):
    """The intent behind a negotiation message."""
    
    # Trade-related intents
    PROPOSE = "propose"       # Initial trade proposal
    COUNTER = "counter"       # Counter-offer to a proposal
    ACCEPT = "accept"         # Accept a proposal
    REJECT = "reject"         # Reject a proposal
    WITHDRAW = "withdraw"     # Withdraw your own proposal
    
    # Information-seeking
    QUESTION = "question"     # Ask for information
    CLARIFY = "clarify"       # Provide clarification
    
    # Strategic communication
    THREATEN = "threaten"     # Express negative consequences
    PROMISE = "promise"       # Make a commitment
    ALLIANCE = "alliance"     # Propose cooperation
    
    # Meta
    GREETING = "greeting"     # Opening/closing pleasantries
    OTHER = "other"           # Catch-all for unclassified


@dataclass
class TradeProposal:
    """A structured trade proposal within a negotiation."""
    
    # What the proposer offers (by resource index: wood, brick, sheep, wheat, ore)
    offering: Tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)
    
    # What the proposer wants in return
    asking: Tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)
    
    # Optional: specific target player
    target_player: Optional[Color] = None
    
    # Optional: conditions attached to the trade
    conditions: Optional[str] = None
    
    def to_trade_tuple(self) -> Tuple[int, ...]:
        """Convert to Catanatron's trade tuple format."""
        return (*self.offering, *self.asking)
    
    @classmethod
    def from_trade_tuple(cls, trade: Tuple[int, ...]) -> "TradeProposal":
        """Create from Catanatron's trade tuple format."""
        return cls(
            offering=tuple(trade[:5]),
            asking=tuple(trade[5:10]),
        )
    
    def format_readable(self) -> str:
        """Format as human-readable string."""
        offer_parts = []
        ask_parts = []
        
        for i, resource in enumerate(RESOURCES):
            if self.offering[i] > 0:
                offer_parts.append(f"{self.offering[i]} {resource.lower()}")
            if self.asking[i] > 0:
                ask_parts.append(f"{self.asking[i]} {resource.lower()}")
        
        offer_str = ", ".join(offer_parts) if offer_parts else "nothing"
        ask_str = ", ".join(ask_parts) if ask_parts else "nothing"
        
        result = f"Offer {offer_str} for {ask_str}"
        if self.target_player:
            result = f"[To {self.target_player.value}] " + result
        if self.conditions:
            result += f" (condition: {self.conditions})"
        
        return result
    
    def is_valid(self) -> bool:
        """Check if this is a valid trade proposal."""
        # Must offer something
        if sum(self.offering) == 0:
            return False
        # Must ask for something
        if sum(self.asking) == 0:
            return False
        # Can't trade same resources
        for o, a in zip(self.offering, self.asking):
            if o > 0 and a > 0:
                return False
        return True


@dataclass
class NegotiationMessage:
    """A single message in a negotiation conversation."""
    
    # Sender of the message
    sender: Color
    
    # Message content (natural language)
    content: str
    
    # Detected or declared intent
    intent: NegotiationIntent = NegotiationIntent.OTHER
    
    # Recipients (empty = broadcast to all players)
    recipients: List[Color] = field(default_factory=list)
    
    # Attached trade proposal (if any)
    trade_proposal: Optional[TradeProposal] = None
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Game turn when sent
    turn: int = 0
    
    # Round within the negotiation
    round_number: int = 0
    
    def is_broadcast(self) -> bool:
        """Check if this message is to all players."""
        return len(self.recipients) == 0
    
    def is_to(self, color: Color) -> bool:
        """Check if this message is addressed to a specific player."""
        return self.is_broadcast() or color in self.recipients
    
    def format_for_log(self) -> str:
        """Format for conversation log."""
        recipient_str = ""
        if not self.is_broadcast():
            recipient_str = f" [to {', '.join(r.value for r in self.recipients)}]"
        
        proposal_str = ""
        if self.trade_proposal:
            proposal_str = f" | {self.trade_proposal.format_readable()}"
        
        return f"[{self.sender.value}]{recipient_str}: {self.content}{proposal_str}"


@dataclass
class NegotiationRound:
    """A single round of negotiation (all players respond once)."""
    
    round_number: int
    messages: List[NegotiationMessage] = field(default_factory=list)
    
    def add_message(self, message: NegotiationMessage) -> None:
        """Add a message to this round."""
        message.round_number = self.round_number
        self.messages.append(message)
    
    def get_messages_from(self, color: Color) -> List[NegotiationMessage]:
        """Get all messages from a specific player."""
        return [m for m in self.messages if m.sender == color]
    
    def has_acceptance(self) -> bool:
        """Check if any message is an acceptance."""
        return any(m.intent == NegotiationIntent.ACCEPT for m in self.messages)
    
    def get_acceptances(self) -> List[NegotiationMessage]:
        """Get all acceptance messages."""
        return [m for m in self.messages if m.intent == NegotiationIntent.ACCEPT]


@dataclass
class NegotiationSession:
    """A complete negotiation conversation between players."""
    
    # Unique session identifier
    session_id: str
    
    # Who initiated the negotiation
    initiator: Color
    
    # All participating players
    participants: List[Color]
    
    # Game turn when started
    start_turn: int
    
    # Rounds of negotiation
    rounds: List[NegotiationRound] = field(default_factory=list)
    
    # Whether negotiation is still active
    is_active: bool = True
    
    # Outcome (if concluded)
    outcome: Optional[str] = None  # "agreement", "no_agreement", "timeout"
    
    # Final agreement (if reached)
    final_agreement: Optional[TradeProposal] = None
    agreed_parties: List[Color] = field(default_factory=list)
    
    # Timestamps
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    
    def current_round(self) -> NegotiationRound:
        """Get the current round, creating if needed."""
        if not self.rounds:
            self.rounds.append(NegotiationRound(round_number=1))
        return self.rounds[-1]
    
    def start_new_round(self) -> NegotiationRound:
        """Start a new round of negotiation."""
        new_round = NegotiationRound(round_number=len(self.rounds) + 1)
        self.rounds.append(new_round)
        return new_round
    
    def add_message(self, message: NegotiationMessage) -> None:
        """Add a message to the current round."""
        message.turn = self.start_turn
        self.current_round().add_message(message)
    
    def get_all_messages(self) -> List[NegotiationMessage]:
        """Get all messages in chronological order."""
        messages = []
        for round in self.rounds:
            messages.extend(round.messages)
        return messages
    
    def get_conversation_log(self) -> str:
        """Get formatted conversation log."""
        lines = []
        for message in self.get_all_messages():
            lines.append(message.format_for_log())
        return "\n".join(lines)
    
    def conclude(
        self, 
        outcome: str,
        agreement: Optional[TradeProposal] = None,
        agreed_parties: Optional[List[Color]] = None
    ) -> None:
        """Conclude the negotiation."""
        self.is_active = False
        self.outcome = outcome
        self.ended_at = datetime.now()
        
        if agreement:
            self.final_agreement = agreement
        if agreed_parties:
            self.agreed_parties = agreed_parties
    
    def get_latest_proposal(self) -> Optional[TradeProposal]:
        """Get the most recent trade proposal."""
        for round in reversed(self.rounds):
            for message in reversed(round.messages):
                if message.trade_proposal:
                    return message.trade_proposal
        return None
    
    def __repr__(self) -> str:
        return (
            f"NegotiationSession({self.session_id}, "
            f"initiator={self.initiator.value}, "
            f"rounds={len(self.rounds)}, "
            f"active={self.is_active})"
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_proposal_message(
    sender: Color,
    proposal: TradeProposal,
    message: str,
    recipients: Optional[List[Color]] = None
) -> NegotiationMessage:
    """Create a negotiation message with a trade proposal."""
    return NegotiationMessage(
        sender=sender,
        content=message,
        intent=NegotiationIntent.PROPOSE,
        recipients=recipients or [],
        trade_proposal=proposal,
    )


def create_counter_message(
    sender: Color,
    counter_proposal: TradeProposal,
    message: str,
    original_proposer: Color
) -> NegotiationMessage:
    """Create a counter-offer message."""
    return NegotiationMessage(
        sender=sender,
        content=message,
        intent=NegotiationIntent.COUNTER,
        recipients=[original_proposer],
        trade_proposal=counter_proposal,
    )


def create_accept_message(
    sender: Color,
    message: str,
    proposer: Color,
    accepted_proposal: Optional[TradeProposal] = None
) -> NegotiationMessage:
    """Create an acceptance message."""
    return NegotiationMessage(
        sender=sender,
        content=message,
        intent=NegotiationIntent.ACCEPT,
        recipients=[proposer],
        trade_proposal=accepted_proposal,
    )


def create_reject_message(
    sender: Color,
    message: str,
    proposer: Color
) -> NegotiationMessage:
    """Create a rejection message."""
    return NegotiationMessage(
        sender=sender,
        content=message,
        intent=NegotiationIntent.REJECT,
        recipients=[proposer],
    )

