"""
Negotiation system for LLM players in Catan.

This module provides the NegotiationManager class which orchestrates
multi-player negotiations before formal trade offers are made.

Key features:
- Round-robin messaging between LLM players
- Automatic participation for all LLM players
- Configurable round limits
- History storage for DECIDE_TRADE context
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, TYPE_CHECKING
import time

from pydantic_ai import UsageLimits

from catanatron.models.player import Color
from catanatron.models.enums import Action

if TYPE_CHECKING:
    from catanatron.game import Game
    from catanatron.players.llm.base import BaseLLMPlayer


@dataclass
class NegotiationMessage:
    """A single message in a negotiation session."""
    sender: Color
    content: str
    timestamp: float = field(default_factory=time.time)
    
    def __str__(self) -> str:
        return f"{self.sender.value}: {self.content}"


@dataclass
class NegotiationSession:
    """
    Represents an active negotiation between players.
    
    Attributes:
        initiator: The player who started the negotiation
        participants: All LLM players in seat order starting from initiator
        messages: List of all messages exchanged
        turn_index: Current speaker index in participants list
        is_active: Whether the negotiation is still ongoing
        max_rounds: Maximum number of complete rounds before timeout
        current_round: Current round number (increments when turn wraps)
        left_players: Players who have explicitly left the negotiation
    """
    initiator: Color
    participants: List[Color]
    messages: List[NegotiationMessage] = field(default_factory=list)
    turn_index: int = 0
    is_active: bool = True
    max_rounds: int = 10
    current_round: int = 1
    left_players: Set[Color] = field(default_factory=set)
    
    @property
    def current_speaker(self) -> Color:
        """Get the color of the current speaker."""
        return self.participants[self.turn_index]
    
    @property
    def active_participants(self) -> List[Color]:
        """Get participants who haven't left."""
        return [p for p in self.participants if p not in self.left_players]
    
    def get_history_text(self) -> str:
        """Format message history as text."""
        if not self.messages:
            return "No messages exchanged yet."
        return "\n".join(str(m) for m in self.messages)


class NegotiationManager:
    """
    Manages negotiation sessions between LLM players.
    
    This class orchestrates the negotiation process:
    1. Tracks which players are LLM players
    2. Enforces one negotiation per turn limit
    3. Manages round-robin messaging
    4. Stores history for DECIDE_TRADE context
    
    Usage:
        manager = NegotiationManager(max_rounds=10)
        # Register players during game setup
        for player in game.state.players:
            if isinstance(player, BaseLLMPlayer):
                manager.register_player(player)
                player.negotiation_manager = manager
    """
    
    def __init__(self, max_rounds: int = 10):
        """
        Initialize the negotiation manager.
        
        Args:
            max_rounds: Maximum messaging rounds before negotiation times out
        """
        self.players: Dict[Color, "BaseLLMPlayer"] = {}
        self.current_session: Optional[NegotiationSession] = None
        self.initiated_this_turn: Dict[int, Set[Color]] = {}
        self.max_rounds = max_rounds
    
    def register_player(self, player: "BaseLLMPlayer") -> None:
        """
        Register an LLM player for negotiation.
        
        Args:
            player: The LLM player to register
        """
        self.players[player.color] = player
    
    def can_initiate(self, color: Color, turn: int) -> bool:
        """
        Check if a player can initiate a negotiation.
        
        Args:
            color: Player's color
            turn: Current turn number
            
        Returns:
            True if the player can initiate, False otherwise
        """
        # Check if there's already an active session
        if self.current_session is not None:
            return False
        
        # Check if this player already initiated this turn
        turn_initiators = self.initiated_this_turn.get(turn, set())
        return color not in turn_initiators
    
    def start_negotiation(self, initiator: Color, game: "Game") -> NegotiationSession:
        """
        Start a new negotiation session.
        
        Args:
            initiator: Color of the player starting the negotiation
            game: Current game instance
            
        Returns:
            The new NegotiationSession
        """
        turn = game.state.num_turns
        
        # Record that this player initiated this turn
        if turn not in self.initiated_this_turn:
            self.initiated_this_turn[turn] = set()
        self.initiated_this_turn[turn].add(initiator)
        
        # Build participant list in seat order starting from initiator
        all_colors = list(game.state.colors)
        initiator_idx = all_colors.index(initiator)
        
        # Rotate to put initiator first, include only LLM players
        participants = []
        for i in range(len(all_colors)):
            color = all_colors[(initiator_idx + i) % len(all_colors)]
            if color in self.players:
                participants.append(color)
        
        self.current_session = NegotiationSession(
            initiator=initiator,
            participants=participants,
            max_rounds=self.max_rounds,
        )
        
        return self.current_session
    
    def add_message(self, sender: Color, content: str) -> NegotiationMessage:
        """
        Add a message to the current negotiation.
        
        Args:
            sender: Color of the message sender
            content: Message content
            
        Returns:
            The created NegotiationMessage
        """
        if self.current_session is None:
            raise RuntimeError("No active negotiation session")
        
        message = NegotiationMessage(sender=sender, content=content)
        self.current_session.messages.append(message)
        return message
    
    def advance_turn(self) -> None:
        """
        Advance to the next speaker in the negotiation.
        
        Handles wrapping around and incrementing rounds.
        """
        if self.current_session is None:
            return
        
        session = self.current_session
        
        # Find next active participant
        original_index = session.turn_index
        while True:
            session.turn_index = (session.turn_index + 1) % len(session.participants)
            
            # Check if we've wrapped around (new round)
            if session.turn_index == 0:
                session.current_round += 1
            
            # Skip players who have left
            if session.current_speaker not in session.left_players:
                break
            
            # Safety check - if we've gone full circle, everyone left
            if session.turn_index == original_index:
                self.end_negotiation()
                break
    
    def player_leaves(self, color: Color) -> None:
        """
        Mark a player as leaving the negotiation.
        
        Args:
            color: Color of the leaving player
        """
        if self.current_session is None:
            return
        
        self.current_session.left_players.add(color)
        
        # Check if only one player remains or initiator left
        active = self.current_session.active_participants
        if len(active) <= 1 or color == self.current_session.initiator:
            self.end_negotiation()
    
    def end_negotiation(self) -> List[NegotiationMessage]:
        """
        End the current negotiation and distribute history.
        
        Returns:
            List of all messages from the negotiation
        """
        if self.current_session is None:
            return []
        
        messages = self.current_session.messages.copy()
        
        # Store history on each participant for DECIDE_TRADE context
        for color in self.current_session.participants:
            if color in self.players:
                self.players[color].store_negotiation_history(messages)
        
        self.current_session = None
        return messages
    
    def run_negotiation(self, game: "Game") -> Optional[Action]:
        """
        Run the full negotiation loop.
        
        Called from initiate_negotiation tool. Blocks until negotiation
        completes (trade offer made, cancelled, or timeout).
        
        Args:
            game: Current game instance
            
        Returns:
            OFFER_TRADE Action if initiator makes an offer, None otherwise
        """
        from catanatron.players.llm.toolsets import (
            NEGOTIATION_PARTICIPANT_TOOLSET,
            NEGOTIATION_INITIATOR_TOOLSET,
            NegotiationDependencies,
        )
        
        if self.current_session is None:
            return None
        
        session = self.current_session
        
        # Skip the initiator's first turn - they just initiated and will
        # send their first message in the agent run that called this
        self.advance_turn()
        
        while session.is_active:
            # Check round limit
            if session.current_round > session.max_rounds:
                self._end_negotiation_timeout()
                return None
            
            current_color = session.current_speaker
            
            # Skip players who have left
            if current_color in session.left_players:
                self.advance_turn()
                continue
            
            player = self.players.get(current_color)
            if player is None:
                self.advance_turn()
                continue
            
            # Build negotiation prompt
            prompt = self._build_negotiation_prompt()
            
            # Select toolset based on whether this is the initiator
            is_initiator = current_color == session.initiator
            toolset = NEGOTIATION_INITIATOR_TOOLSET if is_initiator else NEGOTIATION_PARTICIPANT_TOOLSET
            
            # Build dependencies
            deps = NegotiationDependencies(
                color=current_color,
                game=game,
                playable_actions=game.state.playable_actions if hasattr(game.state, 'playable_actions') else [],
                strategy_recommendation=None,
                strategy_reasoning=None,
                turn_number=game.state.num_turns,
                is_my_turn=False,  # Not their formal turn
                negotiation_manager=self,
                player_instance=player,
            )
            
            try:
                # Run the player's agent with negotiation toolset
                result = player.agent.run_sync(
                    prompt,
                    deps=deps,
                    tools=toolset,
                    usage_limits=UsageLimits(tool_calls_limit=5),
                )
                
                # Check if a trade action was produced
                if hasattr(player, '_pending_trade_action') and player._pending_trade_action:
                    action = player._pending_trade_action
                    player._pending_trade_action = None
                    self.end_negotiation()
                    return action
                
            except Exception as e:
                # On error, skip this player's turn
                import logfire
                logfire.warning(
                    f"Negotiation error for {current_color}",
                    error=str(e)
                )
            
            # Check if negotiation was ended (by leave_negotiation or trade)
            if self.current_session is None:
                return None
            
            self.advance_turn()
        
        return None
    
    def _build_negotiation_prompt(self) -> str:
        """Build the prompt for a player during negotiation."""
        if self.current_session is None:
            return "No active negotiation."
        
        session = self.current_session
        
        parts = [
            "=== NEGOTIATION SESSION ===",
            f"You are {session.current_speaker.value} in a trade negotiation.",
            f"Initiated by: {session.initiator.value}",
            f"Round {session.current_round}/{session.max_rounds}",
            "",
            "=== CONVERSATION HISTORY ===",
            session.get_history_text(),
            "",
            "=== YOUR OPTIONS ===",
        ]
        
        if session.current_speaker == session.initiator:
            parts.append("- send_message: Continue negotiating")
            parts.append("- trade_offer: Make a formal trade offer")
            parts.append("- leave_negotiation: Cancel the negotiation")
        else:
            parts.append("- send_message: Respond to the negotiation")
            parts.append("- leave_negotiation: Leave the negotiation")
        
        parts.append("")
        parts.append("What would you like to do?")
        
        return "\n".join(parts)
    
    def _end_negotiation_timeout(self) -> None:
        """End negotiation due to timeout."""
        if self.current_session is None:
            return
        
        # Add a system message about timeout
        self.add_message(
            self.current_session.initiator,
            "[SYSTEM: Negotiation timed out after maximum rounds]"
        )
        
        self.end_negotiation()


def setup_negotiation(game: "Game", max_rounds: int = 10) -> NegotiationManager:
    """
    Set up negotiation for a game.
    
    Call this before game.play() to enable negotiation between LLM players.
    
    Args:
        game: The game instance
        max_rounds: Maximum messaging rounds per negotiation
        
    Returns:
        The configured NegotiationManager
        
    Example:
        game = Game(players)
        manager = setup_negotiation(game, max_rounds=10)
        game.play()
    """
    from catanatron.players.llm.base import BaseLLMPlayer
    
    manager = NegotiationManager(max_rounds=max_rounds)
    
    for player in game.state.players:
        if isinstance(player, BaseLLMPlayer):
            manager.register_player(player)
            player.negotiation_manager = manager
    
    return manager
