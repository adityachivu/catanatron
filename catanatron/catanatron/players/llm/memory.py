"""
Negotiation Memory - Tracks promises, reputation, and trade history.

This module provides structured memory for negotiation dynamics:
- Promises made and received
- Reputation scores based on behavior
- Trade history for pattern recognition

For POC: This is a stub implementation. The interface is defined but
memory is not persisted across decisions (stateless). The architecture
supports adding full memory later.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from datetime import datetime

from catanatron.models.player import Color


class PromiseType(str, Enum):
    """Types of promises that can be made during negotiation."""
    WILL_TRADE = "will_trade"  # Promise to accept a future trade
    WONT_ROBBER = "wont_robber"  # Promise not to place robber on someone
    WILL_ROBBER = "will_robber"  # Promise to robber a specific player
    ALLIANCE = "alliance"  # General cooperation promise
    NON_AGGRESSION = "non_aggression"  # Promise not to block/attack
    CUSTOM = "custom"  # Free-form promise


class PromiseStatus(str, Enum):
    """Status of a promise."""
    ACTIVE = "active"  # Promise is still valid
    FULFILLED = "fulfilled"  # Promise was honored
    BROKEN = "broken"  # Promise was violated
    EXPIRED = "expired"  # Promise time/condition passed
    CANCELLED = "cancelled"  # Mutually agreed cancellation


@dataclass
class Promise:
    """A promise made during negotiation."""
    
    id: str
    maker: Color  # Who made the promise
    recipient: Color  # Who received the promise
    promise_type: PromiseType
    description: str  # Human-readable description
    turn_made: int  # Game turn when promise was made
    expiration_turn: Optional[int] = None  # When promise expires (None = indefinite)
    status: PromiseStatus = PromiseStatus.ACTIVE
    fulfillment_turn: Optional[int] = None  # When status changed
    
    def is_active(self, current_turn: int) -> bool:
        """Check if promise is still active."""
        if self.status != PromiseStatus.ACTIVE:
            return False
        if self.expiration_turn and current_turn > self.expiration_turn:
            return False
        return True


@dataclass  
class TradeRecord:
    """Record of a completed trade."""
    
    turn: int
    proposer: Color
    accepter: Color
    offered: Tuple[int, int, int, int, int]  # [wood, brick, sheep, wheat, ore]
    received: Tuple[int, int, int, int, int]
    had_prior_negotiation: bool = False
    negotiation_rounds: int = 0


@dataclass
class NegotiationMessage:
    """A message exchanged during negotiation."""
    
    turn: int
    sender: Color
    recipients: List[Color]  # Empty list = broadcast to all
    content: str
    proposed_trade: Optional[Tuple[int, ...]] = None  # If proposing a trade
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ReputationScore:
    """Reputation metrics for a player."""
    
    color: Color
    trades_proposed: int = 0
    trades_accepted: int = 0
    trades_rejected: int = 0
    promises_made: int = 0
    promises_kept: int = 0
    promises_broken: int = 0
    times_robbered_you: int = 0
    times_you_robbered_them: int = 0
    
    @property
    def trade_acceptance_rate(self) -> float:
        """Rate at which this player accepts trades."""
        total = self.trades_accepted + self.trades_rejected
        if total == 0:
            return 0.5  # Neutral assumption
        return self.trades_accepted / total
    
    @property
    def promise_reliability(self) -> float:
        """Rate at which this player keeps promises."""
        total = self.promises_kept + self.promises_broken
        if total == 0:
            return 0.5  # Neutral assumption
        return self.promises_kept / total
    
    @property
    def overall_trust_score(self) -> float:
        """Combined trust metric (0-1, higher is more trustworthy)."""
        # Weight promise reliability higher than trade acceptance
        trade_weight = 0.3
        promise_weight = 0.7
        
        trade_score = self.trade_acceptance_rate
        promise_score = self.promise_reliability
        
        return trade_weight * trade_score + promise_weight * promise_score


class NegotiationMemory:
    """
    Tracks negotiation history, promises, and reputation.
    
    For POC: This is a stateless stub. It maintains memory within a single
    game but doesn't persist between games. The interface supports adding
    persistent memory later.
    
    Architecture for future:
    - Store in SQLite or similar for persistence
    - Add cross-game reputation tracking
    - Add pattern recognition for opponent strategies
    """
    
    def __init__(self, my_color: Color):
        """
        Initialize memory for a player.
        
        Args:
            my_color: The color of the player this memory belongs to
        """
        self.my_color = my_color
        self.promises: List[Promise] = []
        self.trade_history: List[TradeRecord] = []
        self.messages: List[NegotiationMessage] = []
        self.reputation: Dict[Color, ReputationScore] = {}
        self._promise_counter = 0
    
    def record_promise(
        self,
        maker: Color,
        recipient: Color,
        promise_type: PromiseType,
        description: str,
        current_turn: int,
        expiration_turn: Optional[int] = None
    ) -> Promise:
        """
        Record a new promise.
        
        Args:
            maker: Who made the promise
            recipient: Who received the promise
            promise_type: Type of promise
            description: Human-readable description
            current_turn: Current game turn
            expiration_turn: Optional turn when promise expires
            
        Returns:
            The created Promise object
        """
        self._promise_counter += 1
        promise = Promise(
            id=f"promise_{self._promise_counter}",
            maker=maker,
            recipient=recipient,
            promise_type=promise_type,
            description=description,
            turn_made=current_turn,
            expiration_turn=expiration_turn,
        )
        self.promises.append(promise)
        
        # Update reputation
        self._ensure_reputation(maker)
        self.reputation[maker].promises_made += 1
        
        return promise
    
    def fulfill_promise(self, promise_id: str, current_turn: int) -> None:
        """Mark a promise as fulfilled."""
        for promise in self.promises:
            if promise.id == promise_id:
                promise.status = PromiseStatus.FULFILLED
                promise.fulfillment_turn = current_turn
                
                # Update reputation
                self._ensure_reputation(promise.maker)
                self.reputation[promise.maker].promises_kept += 1
                return
    
    def break_promise(self, promise_id: str, current_turn: int) -> None:
        """Mark a promise as broken."""
        for promise in self.promises:
            if promise.id == promise_id:
                promise.status = PromiseStatus.BROKEN
                promise.fulfillment_turn = current_turn
                
                # Update reputation
                self._ensure_reputation(promise.maker)
                self.reputation[promise.maker].promises_broken += 1
                return
    
    def record_trade(
        self,
        proposer: Color,
        accepter: Color,
        offered: Tuple[int, int, int, int, int],
        received: Tuple[int, int, int, int, int],
        current_turn: int,
        had_negotiation: bool = False,
        negotiation_rounds: int = 0
    ) -> TradeRecord:
        """
        Record a completed trade.
        
        Args:
            proposer: Who proposed the trade
            accepter: Who accepted the trade
            offered: Resources offered by proposer
            received: Resources received by proposer
            current_turn: Current game turn
            had_negotiation: Whether there was pre-trade negotiation
            negotiation_rounds: How many rounds of negotiation
            
        Returns:
            The created TradeRecord
        """
        record = TradeRecord(
            turn=current_turn,
            proposer=proposer,
            accepter=accepter,
            offered=offered,
            received=received,
            had_prior_negotiation=had_negotiation,
            negotiation_rounds=negotiation_rounds,
        )
        self.trade_history.append(record)
        
        # Update reputation
        self._ensure_reputation(accepter)
        self.reputation[accepter].trades_accepted += 1
        
        return record
    
    def record_rejection(self, rejector: Color, current_turn: int) -> None:
        """Record that a player rejected a trade."""
        self._ensure_reputation(rejector)
        self.reputation[rejector].trades_rejected += 1
    
    def record_message(
        self,
        sender: Color,
        content: str,
        current_turn: int,
        recipients: Optional[List[Color]] = None,
        proposed_trade: Optional[Tuple[int, ...]] = None
    ) -> NegotiationMessage:
        """
        Record a negotiation message.
        
        Args:
            sender: Who sent the message
            content: Message text
            current_turn: Current game turn
            recipients: Specific recipients (None = broadcast)
            proposed_trade: Trade tuple if this is a proposal
            
        Returns:
            The created NegotiationMessage
        """
        message = NegotiationMessage(
            turn=current_turn,
            sender=sender,
            recipients=recipients or [],
            content=content,
            proposed_trade=proposed_trade,
        )
        self.messages.append(message)
        return message
    
    def record_robber(self, robber: Color, victim: Color, current_turn: int) -> None:
        """Record a robber placement/steal."""
        if victim == self.my_color:
            self._ensure_reputation(robber)
            self.reputation[robber].times_robbered_you += 1
        elif robber == self.my_color:
            self._ensure_reputation(victim)
            self.reputation[victim].times_you_robbered_them += 1
    
    def get_active_promises(self, current_turn: int) -> List[Promise]:
        """Get all active promises."""
        return [p for p in self.promises if p.is_active(current_turn)]
    
    def get_promises_to_me(self, current_turn: int) -> List[Promise]:
        """Get active promises made to this player."""
        return [
            p for p in self.promises 
            if p.is_active(current_turn) and p.recipient == self.my_color
        ]
    
    def get_promises_by_me(self, current_turn: int) -> List[Promise]:
        """Get active promises made by this player."""
        return [
            p for p in self.promises 
            if p.is_active(current_turn) and p.maker == self.my_color
        ]
    
    def get_reputation(self, color: Color) -> ReputationScore:
        """Get reputation score for a player."""
        self._ensure_reputation(color)
        return self.reputation[color]
    
    def get_context(self, current_turn: int) -> str:
        """
        Render memory as natural language for LLM context.
        
        This is the source of truth - LLM should not hallucinate
        beyond what's provided here.
        
        Args:
            current_turn: Current game turn
            
        Returns:
            Formatted memory context string
        """
        lines = []
        
        # Active promises to me
        promises_to_me = self.get_promises_to_me(current_turn)
        if promises_to_me:
            lines.append("PROMISES MADE TO YOU:")
            for p in promises_to_me:
                lines.append(f"  - {p.maker.value}: {p.description} (turn {p.turn_made})")
        
        # Active promises by me
        promises_by_me = self.get_promises_by_me(current_turn)
        if promises_by_me:
            lines.append("\nPROMISES YOU MADE:")
            for p in promises_by_me:
                lines.append(f"  - To {p.recipient.value}: {p.description} (turn {p.turn_made})")
        
        # Reputation scores
        if self.reputation:
            lines.append("\nPLAYER REPUTATION:")
            for color, rep in self.reputation.items():
                if color != self.my_color:
                    trust = rep.overall_trust_score
                    trust_label = "High" if trust > 0.7 else "Medium" if trust > 0.4 else "Low"
                    lines.append(
                        f"  - {color.value}: Trust={trust_label} "
                        f"(trades accepted: {rep.trades_accepted}/{rep.trades_accepted + rep.trades_rejected}, "
                        f"promises kept: {rep.promises_kept}/{rep.promises_kept + rep.promises_broken})"
                    )
        
        # Recent trades
        recent_trades = self.trade_history[-5:] if self.trade_history else []
        if recent_trades:
            lines.append("\nRECENT TRADES:")
            for t in recent_trades:
                offer_str = self._format_resources(t.offered)
                receive_str = self._format_resources(t.received)
                lines.append(
                    f"  - Turn {t.turn}: {t.proposer.value} gave {offer_str} to "
                    f"{t.accepter.value} for {receive_str}"
                )
        
        if not lines:
            return "(No negotiation history recorded)"
        
        return "\n".join(lines)
    
    def _ensure_reputation(self, color: Color) -> None:
        """Ensure a reputation entry exists for a color."""
        if color not in self.reputation:
            self.reputation[color] = ReputationScore(color=color)
    
    def _format_resources(self, resources: Tuple[int, ...]) -> str:
        """Format a resource tuple as readable string."""
        from catanatron.models.enums import RESOURCES
        
        parts = []
        for i, count in enumerate(resources[:5]):
            if count > 0:
                resource = RESOURCES[i] if i < len(RESOURCES) else f"R{i}"
                parts.append(f"{count} {resource.lower()}")
        
        return ", ".join(parts) if parts else "nothing"
    
    def clear(self) -> None:
        """Clear all memory (for testing or game reset)."""
        self.promises.clear()
        self.trade_history.clear()
        self.messages.clear()
        self.reputation.clear()
        self._promise_counter = 0


# =============================================================================
# POC STUB - Stateless Memory
# =============================================================================

class StatelessMemory(NegotiationMemory):
    """
    Stateless memory stub for POC.
    
    This maintains the same interface as NegotiationMemory but doesn't
    track anything between calls. Useful for initial testing where
    memory isn't the focus.
    """
    
    def get_context(self, current_turn: int) -> str:
        """Always returns empty context for stateless POC."""
        return "(No prior negotiations recorded - stateless POC mode)"
    
    def record_promise(self, *args, **kwargs) -> Promise:
        """No-op for stateless mode, returns a dummy promise."""
        return Promise(
            id="stateless",
            maker=args[0] if args else Color.RED,
            recipient=args[1] if len(args) > 1 else Color.BLUE,
            promise_type=PromiseType.CUSTOM,
            description="(stateless mode)",
            turn_made=0,
        )
    
    def record_trade(self, *args, **kwargs) -> TradeRecord:
        """No-op for stateless mode."""
        return TradeRecord(
            turn=0,
            proposer=Color.RED,
            accepter=Color.BLUE,
            offered=(0, 0, 0, 0, 0),
            received=(0, 0, 0, 0, 0),
        )
    
    def record_rejection(self, *args, **kwargs) -> None:
        """No-op for stateless mode."""
        pass
    
    def record_message(self, *args, **kwargs) -> NegotiationMessage:
        """No-op for stateless mode."""
        return NegotiationMessage(
            turn=0,
            sender=Color.RED,
            recipients=[],
            content="(stateless mode)",
        )

