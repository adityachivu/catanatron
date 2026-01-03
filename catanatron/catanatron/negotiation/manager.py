"""
Negotiation Manager - Coordinates multi-player conversations.

This module manages the negotiation flow between LLM players:
1. Starts a negotiation session when a player wants to trade
2. Routes messages between players
3. Detects when agreement is reached
4. Converts agreement to formal trade action

The manager operates OUTSIDE the normal game loop, allowing
back-and-forth conversation before formal trade actions are taken.
"""

import uuid
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Callable
import logging

from catanatron.game import Game
from catanatron.models.player import Color, Player
from catanatron.models.enums import Action, ActionType

from catanatron.negotiation.protocol import (
    NegotiationIntent,
    NegotiationMessage,
    TradeProposal,
    NegotiationRound,
    NegotiationSession,
    create_proposal_message,
    create_accept_message,
    create_reject_message,
)


logger = logging.getLogger(__name__)


@dataclass
class NegotiationConfig:
    """Configuration for negotiation manager."""
    
    # Maximum rounds of negotiation before timeout
    max_rounds: int = 5
    
    # Maximum messages per player per round
    max_messages_per_player: int = 2
    
    # Whether to allow private messages
    allow_private_messages: bool = True
    
    # Whether to auto-accept when all parties agree
    auto_finalize: bool = True
    
    # Timeout per response (seconds)
    response_timeout: float = 30.0


class NegotiationManager:
    """
    Manages negotiation sessions between players.
    
    The manager coordinates pre-trade conversations:
    1. Player A wants to trade → starts negotiation session
    2. Manager broadcasts opening message to other players
    3. Other players respond via their handle_negotiation_message()
    4. Manager routes messages until agreement or timeout
    5. If agreement, returns the formal OFFER_TRADE action
    
    This operates OUTSIDE the Catanatron game loop, enabling
    rich conversation before the formal trade mechanism kicks in.
    """
    
    def __init__(
        self,
        players: Dict[Color, Player],
        config: Optional[NegotiationConfig] = None
    ):
        """
        Initialize the negotiation manager.
        
        Args:
            players: Map of color to player instances
            config: Optional configuration
        """
        self.players = players
        self.config = config or NegotiationConfig()
        
        # Active sessions
        self.active_sessions: Dict[str, NegotiationSession] = {}
        
        # Session history
        self.completed_sessions: List[NegotiationSession] = []
        
        # Agreements reached (for lookup during formal trade)
        self.pending_agreements: Dict[Tuple[Color, Color], TradeProposal] = {}
        
        # Logging
        self.negotiation_log: List[Dict] = []
    
    def start_negotiation(
        self,
        game: Game,
        initiator: Color,
        opening_message: str,
        initial_proposal: Optional[TradeProposal] = None,
        target_players: Optional[List[Color]] = None
    ) -> NegotiationSession:
        """
        Start a new negotiation session.
        
        Args:
            game: Current game state
            initiator: Who is starting the negotiation
            opening_message: The opening message
            initial_proposal: Optional initial trade proposal
            target_players: Specific players to negotiate with (None = all)
            
        Returns:
            The created negotiation session
        """
        # Determine participants
        if target_players:
            participants = [initiator] + target_players
        else:
            participants = [c for c in game.state.colors]
        
        # Create session
        session_id = str(uuid.uuid4())[:8]
        session = NegotiationSession(
            session_id=session_id,
            initiator=initiator,
            participants=participants,
            start_turn=game.state.num_turns,
        )
        
        # Add opening message
        opening = NegotiationMessage(
            sender=initiator,
            content=opening_message,
            intent=NegotiationIntent.PROPOSE if initial_proposal else NegotiationIntent.GREETING,
            trade_proposal=initial_proposal,
            turn=game.state.num_turns,
        )
        session.add_message(opening)
        
        # Store session
        self.active_sessions[session_id] = session
        
        # Log
        self._log_event("session_started", {
            "session_id": session_id,
            "initiator": initiator.value,
            "participants": [p.value for p in participants],
            "has_proposal": initial_proposal is not None,
        })
        
        logger.info(f"Negotiation started: {session}")
        
        return session
    
    async def run_negotiation(
        self,
        game: Game,
        session: NegotiationSession
    ) -> Optional[Tuple[TradeProposal, List[Color]]]:
        """
        Run a negotiation session to completion.
        
        This is the main async loop that:
        1. Gets responses from all participants
        2. Checks for agreement
        3. Continues or concludes based on intents
        
        Args:
            game: Current game state
            session: The negotiation session to run
            
        Returns:
            (proposal, agreeing_parties) if agreement reached, None otherwise
        """
        initiator = session.initiator
        
        for round_num in range(self.config.max_rounds):
            # Start new round (except first, which has opening message)
            if round_num > 0:
                session.start_new_round()
            
            # Get responses from all non-initiator participants
            responses = await self._collect_responses(game, session)
            
            # Check for acceptances
            acceptances = [
                (color, msg) for color, msg in responses 
                if msg and msg.intent == NegotiationIntent.ACCEPT
            ]
            
            if acceptances:
                # We have agreement!
                proposal = session.get_latest_proposal()
                agreeing_parties = [initiator] + [color for color, _ in acceptances]
                
                session.conclude(
                    outcome="agreement",
                    agreement=proposal,
                    agreed_parties=agreeing_parties
                )
                
                # Store for formal trade lookup
                for color, _ in acceptances:
                    if proposal:
                        self.pending_agreements[(initiator, color)] = proposal
                
                self._log_event("agreement_reached", {
                    "session_id": session.session_id,
                    "parties": [p.value for p in agreeing_parties],
                    "proposal": proposal.format_readable() if proposal else None,
                })
                
                return proposal, agreeing_parties
            
            # Check for all rejections
            rejections = [
                (color, msg) for color, msg in responses 
                if msg and msg.intent == NegotiationIntent.REJECT
            ]
            
            non_initiator_count = len(session.participants) - 1
            if len(rejections) == non_initiator_count:
                # Everyone rejected
                session.conclude(outcome="no_agreement")
                self._log_event("all_rejected", {
                    "session_id": session.session_id,
                })
                return None
            
            # Check for counter-offers
            counters = [
                (color, msg) for color, msg in responses 
                if msg and msg.intent == NegotiationIntent.COUNTER
            ]
            
            if counters:
                # Let initiator respond to counters
                initiator_response = await self._get_initiator_response(
                    game, session, counters
                )
                if initiator_response:
                    session.add_message(initiator_response)
                    
                    if initiator_response.intent == NegotiationIntent.ACCEPT:
                        # Initiator accepted a counter
                        counter_from = counters[0][0]  # First counter-offerer
                        proposal = session.get_latest_proposal()
                        
                        session.conclude(
                            outcome="agreement",
                            agreement=proposal,
                            agreed_parties=[initiator, counter_from]
                        )
                        
                        if proposal:
                            self.pending_agreements[(initiator, counter_from)] = proposal
                        
                        return proposal, [initiator, counter_from]
        
        # Timeout - no agreement reached
        session.conclude(outcome="timeout")
        self._log_event("timeout", {
            "session_id": session.session_id,
            "rounds": len(session.rounds),
        })
        
        return None
    
    async def _collect_responses(
        self,
        game: Game,
        session: NegotiationSession
    ) -> List[Tuple[Color, Optional[NegotiationMessage]]]:
        """
        Collect responses from all non-initiator participants.
        
        Args:
            game: Current game state
            session: Current negotiation session
            
        Returns:
            List of (color, response_message) tuples
        """
        responses = []
        conversation_log = session.get_conversation_log()
        latest_proposal = session.get_latest_proposal()
        
        for color in session.participants:
            if color == session.initiator:
                continue
            
            player = self.players.get(color)
            if player is None:
                continue
            
            # Check if player has handle_negotiation_message method
            if hasattr(player, 'handle_negotiation_message'):
                try:
                    response_text = player.handle_negotiation_message(
                        game,
                        session.initiator,
                        conversation_log,
                        latest_proposal.to_trade_tuple() if latest_proposal else None
                    )
                    
                    if response_text:
                        # Parse response into message
                        # For POC, we use simple keyword detection for intent
                        intent = self._detect_intent(response_text)
                        
                        message = NegotiationMessage(
                            sender=color,
                            content=response_text,
                            intent=intent,
                            recipients=[session.initiator],
                            turn=session.start_turn,
                        )
                        session.add_message(message)
                        responses.append((color, message))
                    else:
                        # No response = implicit rejection
                        responses.append((color, None))
                        
                except Exception as e:
                    logger.warning(f"Error getting response from {color}: {e}")
                    responses.append((color, None))
            else:
                # Player doesn't support negotiation - auto-reject
                responses.append((color, None))
        
        return responses
    
    async def _get_initiator_response(
        self,
        game: Game,
        session: NegotiationSession,
        counters: List[Tuple[Color, NegotiationMessage]]
    ) -> Optional[NegotiationMessage]:
        """
        Get the initiator's response to counter-offers.
        
        For POC, this is simplified - real implementation would
        call the initiator's LLM to evaluate counters.
        """
        initiator = session.initiator
        player = self.players.get(initiator)
        
        if player is None or not hasattr(player, 'handle_negotiation_message'):
            return None
        
        # Format counters for the initiator
        counter_text = "\n".join([
            f"{color.value}: {msg.content}" for color, msg in counters
        ])
        
        try:
            response_text = player.handle_negotiation_message(
                game,
                counters[0][0],  # First counter-offerer as primary
                counter_text,
                counters[0][1].trade_proposal.to_trade_tuple() if counters[0][1].trade_proposal else None
            )
            
            if response_text:
                intent = self._detect_intent(response_text)
                return NegotiationMessage(
                    sender=initiator,
                    content=response_text,
                    intent=intent,
                    recipients=[c for c, _ in counters],
                    turn=session.start_turn,
                )
        except Exception as e:
            logger.warning(f"Error getting initiator response: {e}")
        
        return None
    
    def _detect_intent(self, message: str) -> NegotiationIntent:
        """
        Detect the intent of a message using simple keyword matching.
        
        For POC, this is a basic heuristic. Full implementation would
        use LLM-based intent classification.
        """
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["accept", "deal", "agreed", "yes", "i'll take"]):
            return NegotiationIntent.ACCEPT
        elif any(word in message_lower for word in ["reject", "no", "refuse", "decline", "not interested"]):
            return NegotiationIntent.REJECT
        elif any(word in message_lower for word in ["counter", "how about", "instead", "what if", "i'd prefer"]):
            return NegotiationIntent.COUNTER
        elif any(word in message_lower for word in ["?", "what", "why", "how", "do you"]):
            return NegotiationIntent.QUESTION
        elif any(word in message_lower for word in ["promise", "commit", "guarantee", "i will"]):
            return NegotiationIntent.PROMISE
        elif any(word in message_lower for word in ["threat", "or else", "beware", "don't"]):
            return NegotiationIntent.THREATEN
        
        return NegotiationIntent.OTHER
    
    def has_pending_agreement(
        self,
        player1: Color,
        player2: Color
    ) -> Optional[TradeProposal]:
        """
        Check if there's a pending agreement between two players.
        
        This is used when a formal OFFER_TRADE action is made to
        check if it matches a negotiated agreement.
        """
        return (
            self.pending_agreements.get((player1, player2)) or
            self.pending_agreements.get((player2, player1))
        )
    
    def clear_pending_agreement(self, player1: Color, player2: Color) -> None:
        """Clear a pending agreement after it's been executed."""
        self.pending_agreements.pop((player1, player2), None)
        self.pending_agreements.pop((player2, player1), None)
    
    def get_session(self, session_id: str) -> Optional[NegotiationSession]:
        """Get a session by ID."""
        return self.active_sessions.get(session_id)
    
    def end_session(self, session_id: str) -> None:
        """End and archive a session."""
        session = self.active_sessions.pop(session_id, None)
        if session:
            if session.is_active:
                session.conclude(outcome="cancelled")
            self.completed_sessions.append(session)
    
    def get_negotiation_stats(self) -> Dict[str, Any]:
        """Get statistics about negotiations."""
        completed = self.completed_sessions
        
        agreements = [s for s in completed if s.outcome == "agreement"]
        rejections = [s for s in completed if s.outcome == "no_agreement"]
        timeouts = [s for s in completed if s.outcome == "timeout"]
        
        return {
            "total_sessions": len(completed),
            "agreements": len(agreements),
            "no_agreements": len(rejections),
            "timeouts": len(timeouts),
            "agreement_rate": len(agreements) / len(completed) if completed else 0,
            "avg_rounds": sum(len(s.rounds) for s in completed) / len(completed) if completed else 0,
        }
    
    def _log_event(self, event_type: str, data: Dict) -> None:
        """Log a negotiation event."""
        self.negotiation_log.append({
            "type": event_type,
            "data": data,
        })
    
    def clear(self) -> None:
        """Clear all sessions and state."""
        self.active_sessions.clear()
        self.completed_sessions.clear()
        self.pending_agreements.clear()
        self.negotiation_log.clear()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_trade_action_from_proposal(
    proposer: Color,
    proposal: TradeProposal
) -> Action:
    """
    Create a formal OFFER_TRADE action from a negotiated proposal.
    
    This bridges the gap between natural language negotiation
    and the formal Catanatron trade system.
    """
    trade_tuple = proposal.to_trade_tuple()
    return Action(proposer, ActionType.OFFER_TRADE, trade_tuple)


def proposal_matches_trade(
    proposal: TradeProposal,
    trade_tuple: Tuple[int, ...]
) -> bool:
    """
    Check if a proposal matches a trade tuple.
    
    Used to verify that a formal trade matches a negotiated agreement.
    """
    if not trade_tuple or len(trade_tuple) < 10:
        return False
    
    return (
        proposal.offering == tuple(trade_tuple[:5]) and
        proposal.asking == tuple(trade_tuple[5:10])
    )

