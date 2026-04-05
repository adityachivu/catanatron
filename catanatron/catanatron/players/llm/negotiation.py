"""
Negotiation framework for LLM-powered Catan players.

This module provides multi-player negotiation capabilities that run outside
the game engine, allowing LLM players to discuss trades before making
formal offers.

The negotiation flow has two phases:
1. Messaging phase: Up to k rounds of round-robin chat between all players.
   All players (including the initiator) use the same NEGOTIATION_PARTICIPANT_TOOLSET.
2. Finalization phase: One LLM call to the initiator with the TRADE_FINALIZE_TOOLSET,
   where they review the conversation and specify their final trade offer.

Key Components:
- NegotiationMessage: A single message in a negotiation
- NegotiationSession: State of an active negotiation
- NegotiationManager: Orchestrates negotiations between LLM players
- setup_negotiation(): Helper to wire up negotiation before game.play()

Usage:
    from catanatron.players.llm.negotiation import setup_negotiation
    
    game = Game(players)
    manager = setup_negotiation(game, max_rounds=3)
    winner = game.play()
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, TYPE_CHECKING
import time

try:
    import logfire
except ImportError:
    logfire = None  # type: ignore[assignment]
from contextlib import nullcontext
from pydantic_ai import UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded

from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.models.enums import Action

if TYPE_CHECKING:
    from catanatron.players.llm.base import BaseLLMPlayer

MAX_NEGOTIATION_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0
NEGOTIATION_TOOL_CALLS_LIMIT = 10


@dataclass
class NegotiationMessage:
    """A single message in a negotiation session."""
    sender: Color
    content: str
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "sender": self.sender.value,
            "content": self.content,
            "timestamp": self.timestamp,
        }


@dataclass
class NegotiationSession:
    """
    State of an active negotiation.
    
    Attributes:
        initiator: Color of the player who started the negotiation
        participants: List of player colors in seat order from initiator
        messages: List of messages exchanged during negotiation
        turn_index: Index of current speaker in participants list
        is_active: Whether the negotiation is still ongoing
        max_rounds: Maximum number of rounds before auto-ending
        current_round: Current round number (one round = all participants speak)
        game: Reference to the game for context
    """
    initiator: Color
    participants: List[Color]
    messages: List[NegotiationMessage] = field(default_factory=list)
    turn_index: int = 0
    is_active: bool = True
    max_rounds: int = 10
    current_round: int = 0
    game: Optional[Game] = None
    
    @property
    def current_speaker(self) -> Color:
        """Get the color of the current speaker."""
        return self.participants[self.turn_index]
    
    @property
    def is_initiator_turn(self) -> bool:
        """Check if it's the initiator's turn to speak."""
        return self.current_speaker == self.initiator
    
    def add_message(self, sender: Color, content: str) -> NegotiationMessage:
        """Add a message to the session."""
        msg = NegotiationMessage(sender=sender, content=content)
        self.messages.append(msg)
        return msg
    
    def advance_turn(self) -> None:
        """Advance to the next speaker in round-robin order."""
        self.turn_index = (self.turn_index + 1) % len(self.participants)
        
        # If we've cycled back to the initiator, increment round
        if self.turn_index == 0:
            self.current_round += 1
    
    def remove_participant(self, color: Color) -> bool:
        """
        Remove a participant from the negotiation.
        
        Returns True if removed, False if not found or is initiator.
        """
        if color == self.initiator:
            return False  # Initiator cannot leave
        
        if color not in self.participants:
            return False
        
        # Adjust turn_index if needed
        removed_index = self.participants.index(color)
        self.participants.remove(color)
        
        if removed_index < self.turn_index:
            self.turn_index -= 1
        elif removed_index == self.turn_index:
            # Current speaker left, stay at same index (next person)
            if self.turn_index >= len(self.participants):
                self.turn_index = 0
        
        return True
    
    def format_history(self) -> str:
        """Format message history as a string for prompts."""
        if not self.messages:
            return "No messages yet."
        
        lines = []
        for msg in self.messages:
            lines.append(f"{msg.sender.value}: {msg.content}")
        return "\n".join(lines)


class NegotiationManager:
    """
    Orchestrates negotiations between LLM players.
    
    The manager:
    - Tracks which players have initiated negotiations this turn
    - Manages active negotiation sessions
    - Runs the negotiation loop by calling each player's agent
    - Distributes negotiation history to all participants when done
    
    Usage:
        manager = NegotiationManager(max_rounds=10)
        manager.register_player(player1)
        manager.register_player(player2)
        
        # When a player calls initiate_negotiation tool:
        result = manager.start_negotiation(initiator_color, game)
    """
    
    def __init__(self, max_rounds: int = 10):
        """
        Initialize the negotiation manager.
        
        Args:
            max_rounds: Maximum rounds per negotiation before auto-ending
        """
        self.players: Dict[Color, "BaseLLMPlayer"] = {}
        self.current_session: Optional[NegotiationSession] = None
        self.initiated_this_turn: Dict[int, Set[Color]] = {}
        self.max_rounds = max_rounds
    
    def register_player(self, player: "BaseLLMPlayer") -> None:
        """
        Register an LLM player with the manager.
        
        Args:
            player: The LLM player to register
        """
        self.players[player.color] = player
        player.negotiation_manager = self
    
    def unregister_player(self, color: Color) -> None:
        """Remove a player from the manager."""
        if color in self.players:
            self.players[color].negotiation_manager = None
            del self.players[color]
    
    def can_initiate(self, color: Color, turn: int) -> bool:
        """
        Check if a player can initiate a negotiation.
        
        A player can only initiate one negotiation per turn.
        
        Args:
            color: The player's color
            turn: The current turn number
            
        Returns:
            True if the player can initiate a negotiation
        """
        if color not in self.players:
            return False
        
        if self.current_session is not None:
            return False  # Another negotiation is in progress
        
        turn_initiators = self.initiated_this_turn.get(turn, set())
        return color not in turn_initiators
    
    def _get_participants_in_seat_order(self, initiator: Color, game: Game) -> List[Color]:
        """
        Get all LLM players in seat order starting from initiator.
        
        Args:
            initiator: The initiating player's color
            game: The current game
            
        Returns:
            List of colors in seat order, starting with initiator
        """
        all_colors = list(game.state.colors)
        initiator_index = all_colors.index(initiator)
        
        # Reorder to start with initiator
        ordered = all_colors[initiator_index:] + all_colors[:initiator_index]
        
        # Filter to only include registered LLM players
        return [c for c in ordered if c in self.players]
    
    def start_negotiation(self, initiator: Color, game: Game) -> Dict[str, Any]:
        """
        Start a new negotiation session.
        
        This method:
        1. Creates a new session
        2. Marks the initiator as having initiated this turn
        3. Runs the negotiation loop until completion
        4. Returns the result (trade action or None)
        
        Args:
            initiator: Color of the player starting the negotiation
            game: The current game
            
        Returns:
            Dictionary with negotiation result:
            - "success": bool
            - "trade_action": Optional[Action] - the trade offer if made
            - "message": str - description of what happened
        """
        turn = game.state.num_turns
        
        # Check if can initiate
        if not self.can_initiate(initiator, turn):
            return {
                "success": False,
                "trade_action": None,
                "message": "Cannot initiate negotiation (already initiated this turn or in progress)",
            }
        
        # Mark as initiated this turn
        if turn not in self.initiated_this_turn:
            self.initiated_this_turn[turn] = set()
        self.initiated_this_turn[turn].add(initiator)
        
        # Get participants
        participants = self._get_participants_in_seat_order(initiator, game)
        
        if len(participants) < 2:
            return {
                "success": False,
                "trade_action": None,
                "message": "Not enough LLM players for negotiation",
            }
        
        # Create session
        self.current_session = NegotiationSession(
            initiator=initiator,
            participants=participants,
            max_rounds=self.max_rounds,
            game=game,
        )
        
        # Run the negotiation loop with optional logfire span
        span_ctx = (
            logfire.span(
                "catanatron.negotiation_session",
                initiator=initiator.value,
                participants=[c.value for c in participants],
                turn_number=turn,
                max_rounds=self.max_rounds,
            )
            if logfire is not None
            else nullcontext()
        )
        with span_ctx as span:
            trade_action = self._run_negotiation_loop(game)

            if span is not None and self.current_session:
                span.set_attribute("messages_count", len(self.current_session.messages))
                span.set_attribute("final_round", self.current_session.current_round)
                span.set_attribute("negotiation_messages", [
                    {"sender": m.sender.value, "content": m.content}
                    for m in self.current_session.messages
                ])

            if span is not None:
                span.set_attribute("resulted_in_trade", trade_action is not None)
            if trade_action is not None and span is not None:
                resource_names = ["wood", "brick", "sheep", "wheat", "ore"]
                offer_values = trade_action.value[:5]
                ask_values = trade_action.value[5:10]
                span.set_attribute("trade_offer", {
                    "offering": {
                        resource_names[i]: v
                        for i, v in enumerate(offer_values) if v > 0
                    },
                    "asking": {
                        resource_names[i]: v
                        for i, v in enumerate(ask_values) if v > 0
                    },
                })
        
        # End the session and distribute history
        self._end_negotiation()
        
        if trade_action:
            return {
                "success": True,
                "trade_action": trade_action,
                "message": "Negotiation completed with trade offer",
            }
        else:
            return {
                "success": True,
                "trade_action": None,
                "message": "Negotiation ended without trade offer",
            }
    
    def _run_negotiation_loop(self, game: Game) -> Optional[Action]:
        """
        Run the two-phase negotiation process.
        
        Phase 1 (Messaging): Up to max_rounds of round-robin chat. All players
        (including initiator) use NEGOTIATION_PARTICIPANT_TOOLSET. Ends when:
        - max_rounds reached
        - All non-initiator participants leave
        
        Phase 2 (Finalization): One LLM call to the initiator with
        TRADE_FINALIZE_TOOLSET to specify the final trade offer.
        
        Args:
            game: The current game
            
        Returns:
            OFFER_TRADE action if initiator made one, else None
        """
        session = self.current_session
        if session is None:
            return None
        
        self._run_messaging_phase(game, session)
        return self._run_finalization_phase(game, session)
    
    def _run_messaging_phase(self, game: Game, session: NegotiationSession) -> None:
        """
        Phase 1: Round-robin messaging between all participants.
        
        All players (including the initiator) use the same participant toolset
        with send_message and leave_negotiation tools.
        """
        from catanatron.players.llm.toolsets import NEGOTIATION_PARTICIPANT_TOOLSET
        from catanatron.players.llm.base import CatanDependencies
        
        while session.is_active:
            current_color = session.current_speaker
            player = self.players.get(current_color)
            
            if player is None:
                session.advance_turn()
                continue
            
            prompt = self._build_messaging_prompt(session, current_color)
            
            deps = CatanDependencies(
                color=current_color,
                game=game,
                playable_actions=[],
                strategy_recommendation=None,
                strategy_reasoning=None,
                turn_number=game.state.num_turns,
                is_my_turn=False,
                negotiation_manager=self,
                player_instance=player,
            )
            
            for attempt in range(MAX_NEGOTIATION_RETRIES):
                try:
                    player.agent.run_sync(
                        prompt,
                        deps=deps,
                        toolsets=[NEGOTIATION_PARTICIPANT_TOOLSET],
                        usage_limits=UsageLimits(
                            tool_calls_limit=NEGOTIATION_TOOL_CALLS_LIMIT,
                        ),
                    )
                    break
                except UsageLimitExceeded:
                    if logfire is not None:
                        logfire.error(
                            f"Negotiation messaging usage limit for {current_color}",
                            color=current_color.value,
                        )
                    break
                except Exception as e:
                    if attempt < MAX_NEGOTIATION_RETRIES - 1:
                        if logfire is not None:
                            logfire.warning(
                                f"Negotiation messaging retry {attempt + 1} for {current_color}: {e}",
                                color=current_color.value,
                            )
                        time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
                    else:
                        if logfire is not None:
                            logfire.error(
                                f"Negotiation messaging error for {current_color} after {MAX_NEGOTIATION_RETRIES} attempts: {e}",
                                color=current_color.value,
                            )
            
            session.advance_turn()
            
            if session.current_round >= session.max_rounds:
                session.is_active = False
                return
            
            if len(session.participants) <= 1:
                session.is_active = False
                return
    
    def _run_finalization_phase(
        self, game: Game, session: NegotiationSession
    ) -> Optional[Action]:
        """
        Phase 2: Initiator specifies the final trade offer.
        
        One LLM call with the TRADE_FINALIZE_TOOLSET. The initiator reviews
        the conversation and their resources, then calls finalize_trade to
        submit the offer (or declines to trade).
        """
        from catanatron.players.llm.toolsets import TRADE_FINALIZE_TOOLSET
        from catanatron.players.llm.base import CatanDependencies
        
        initiator = session.initiator
        player = self.players.get(initiator)
        if player is None:
            return None
        
        prompt = self._build_finalization_prompt(session)
        
        deps = CatanDependencies(
            color=initiator,
            game=game,
            playable_actions=[],
            strategy_recommendation=None,
            strategy_reasoning=None,
            turn_number=game.state.num_turns,
            is_my_turn=True,
            negotiation_manager=self,
            player_instance=player,
        )
        
        player._pending_trade_action = None
        
        for attempt in range(MAX_NEGOTIATION_RETRIES):
            try:
                player.agent.run_sync(
                    prompt,
                    deps=deps,
                    toolsets=[TRADE_FINALIZE_TOOLSET],
                    usage_limits=UsageLimits(
                        tool_calls_limit=NEGOTIATION_TOOL_CALLS_LIMIT,
                    ),
                )
                break
            except UsageLimitExceeded:
                if logfire is not None:
                    logfire.error(
                        f"Trade finalization usage limit for {initiator}",
                        color=initiator.value,
                    )
                break
            except Exception as e:
                if attempt < MAX_NEGOTIATION_RETRIES - 1:
                    if logfire is not None:
                        logfire.warning(
                            f"Trade finalization retry {attempt + 1} for {initiator}: {e}",
                            color=initiator.value,
                        )
                    time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
                else:
                    if logfire is not None:
                        logfire.error(
                            f"Trade finalization error for {initiator} after {MAX_NEGOTIATION_RETRIES} attempts: {e}",
                            color=initiator.value,
                        )
        
        if player._pending_trade_action is not None:
            trade_action = player._pending_trade_action
            player._pending_trade_action = None
            return trade_action
        
        return None
    
    def _build_messaging_prompt(self, session: NegotiationSession, speaker: Color) -> str:
        """
        Build the prompt for a messaging-phase turn.
        
        All players (including initiator) get the same tool set during messaging:
        send_message to chat, leave_negotiation to exit early.
        """
        lines = []
        
        lines.append("=== TRADE NEGOTIATION (Messaging Phase) ===")
        lines.append(f"You are {speaker.value} in a trade negotiation.")
        lines.append(f"Initiated by: {session.initiator.value}")
        lines.append(f"Participants: {[c.value for c in session.participants]}")
        lines.append(f"Round: {session.current_round + 1}/{session.max_rounds}")
        lines.append("")
        
        if session.messages:
            lines.append("=== CONVERSATION SO FAR ===")
            lines.append(session.format_history())
            lines.append("")
        else:
            lines.append("This is the start of the negotiation. No messages yet.")
            lines.append("")
        
        if speaker == session.initiator:
            lines.append("=== YOUR TURN (INITIATOR) ===")
            lines.append("You initiated this negotiation. Discuss what you want to trade.")
            lines.append("After all messaging rounds conclude, you will specify your final trade offer.")
            lines.append("")
            lines.append("Use send_message to communicate with other players.")
        else:
            lines.append("=== YOUR TURN ===")
            lines.append("You can:")
            lines.append("1. Send a message to respond or make counter-proposals")
            lines.append("2. Leave the negotiation if you're not interested")
            lines.append("")
            lines.append("Use send_message to respond or leave_negotiation to exit.")
        
        return "\n".join(lines)
    
    def _build_finalization_prompt(self, session: NegotiationSession) -> str:
        """
        Build the prompt for the finalization phase (initiator only).
        
        The initiator reviews the full conversation and their resources,
        then uses finalize_trade to submit the final offer.
        """
        lines = []
        
        lines.append("=== TRADE FINALIZATION ===")
        lines.append(f"You are {session.initiator.value}. The negotiation messaging has concluded.")
        lines.append(f"Participants: {[c.value for c in session.participants]}")
        lines.append("")
        
        if session.messages:
            lines.append("=== FULL NEGOTIATION HISTORY ===")
            lines.append(session.format_history())
            lines.append("")
        else:
            lines.append("No messages were exchanged during negotiation.")
            lines.append("")
        
        lines.append("=== YOUR TASK ===")
        lines.append("Based on the negotiation above, decide on your trade offer.")
        lines.append("Use get_game_and_action_analysis to review your resources,")
        lines.append("then use finalize_trade to submit your offer.")
        lines.append("")
        lines.append("Specify: offer=[wood, brick, sheep, wheat, ore], ask=[wood, brick, sheep, wheat, ore]")
        
        return "\n".join(lines)
    
    def add_message(self, sender: Color, content: str) -> None:
        """
        Add a message to the current session.
        
        Called by the send_message tool.
        
        Args:
            sender: Color of the message sender
            content: Message content
        """
        if self.current_session is None:
            return
        
        self.current_session.add_message(sender, content)
    
    def remove_participant(self, color: Color) -> bool:
        """
        Remove a participant from the current session.
        
        Called by the leave_negotiation tool.
        
        Args:
            color: Color of the participant to remove
            
        Returns:
            True if removed, False otherwise
        """
        if self.current_session is None:
            return False
        
        return self.current_session.remove_participant(color)
    
    def _end_negotiation(self) -> None:
        """
        End the current negotiation and distribute history.
        
        Stores the negotiation history on each participant player.
        """
        if self.current_session is None:
            return
        
        messages = self.current_session.messages.copy()
        
        # Store history on each participant
        for color in self.current_session.participants:
            player = self.players.get(color)
            if player is not None:
                player.store_negotiation_history(messages)
        
        self.current_session = None
    
    def reset(self) -> None:
        """Reset the manager state (e.g., between games)."""
        self.current_session = None
        self.initiated_this_turn.clear()


def setup_negotiation(game: Game, max_rounds: int = 10) -> NegotiationManager:
    """
    Set up negotiation support for a game.
    
    Call this before game.play() to enable negotiation between LLM players.
    
    Args:
        game: The game instance
        max_rounds: Maximum rounds per negotiation
        
    Returns:
        The NegotiationManager instance
        
    Example:
        game = Game(players)
        manager = setup_negotiation(game, max_rounds=10)
        winner = game.play()
    """
    from catanatron.players.llm.base import BaseLLMPlayer
    
    manager = NegotiationManager(max_rounds=max_rounds)
    
    for player in game.state.players:
        if isinstance(player, BaseLLMPlayer):
            manager.register_player(player)
    
    return manager
