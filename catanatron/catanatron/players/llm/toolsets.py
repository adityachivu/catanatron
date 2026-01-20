"""
PydanticAI Toolsets for the Catan LLM agent.

This module provides composable toolsets using Pydantic AI's FunctionToolset pattern.
Toolsets are selected at agent.run() time based on game state.

Toolsets:
- REASONING_TOOLSET: For main gameplay decisions (analysis + initiate_negotiation)
- CHAT_INITIATOR_TOOLSET: For negotiation initiator (analysis + send_message + trade_offer)
- CHAT_PARTICIPANT_TOOLSET: For negotiation participants (analysis + send_message + leave_negotiation)
"""

from typing import List, Dict, Any
from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

from catanatron.state_functions import get_player_freqdeck
from catanatron.models.enums import ActionType, Action

# Import helper functions from tools.py to avoid duplication
from catanatron.players.llm.tools import (
    _analyze_expansion,
    _analyze_blocking,
    _analyze_ports,
    _analyze_robber,
)

# Import CatanDependencies for type hints
from catanatron.players.llm.base import CatanDependencies


# ============================================================================
# Tool Implementation Functions
# ============================================================================

def _get_game_and_action_analysis(ctx: RunContext[CatanDependencies]) -> Dict[str, Any]:
    """Get comprehensive game state, available actions, and strategy recommendation."""
    from catanatron.players.llm.state_formatter import StateFormatter
    
    return {
        "game_state": StateFormatter.format_full_state(ctx.deps.game, ctx.deps.color),
        "available_actions": [
            StateFormatter.format_action(action, i) 
            for i, action in enumerate(ctx.deps.playable_actions)
        ],
        "strategy_recommendation": (
            ctx.deps.strategy_recommendation 
            if ctx.deps.strategy_recommendation 
            else "No strategy advisor configured"
        ),
        "strategy_reasoning": (
            ctx.deps.strategy_reasoning 
            if ctx.deps.strategy_reasoning 
            else "No detailed reasoning available"
        ),
    }


def _analyze_board(ctx: RunContext[CatanDependencies], focus: str) -> Dict[str, Any]:
    """Analyze the board for strategic insights based on focus area."""
    game = ctx.deps.game
    my_color = ctx.deps.color

    if focus == "expansion":
        return _analyze_expansion(game, my_color)
    elif focus == "blocking":
        return _analyze_blocking(game, my_color)
    elif focus == "ports":
        return _analyze_ports(game, my_color)
    elif focus == "robber":
        return _analyze_robber(game, my_color)
    else:
        return {
            "error": f"Unknown focus: {focus}",
            "valid_options": ["expansion", "blocking", "ports", "robber"],
        }


def _make_trade_offer(
    ctx: RunContext[CatanDependencies], 
    offer: List[int], 
    ask: List[int]
) -> Dict[str, Any]:
    """Create a trade offer action."""
    resource_names = ["wood", "brick", "sheep", "wheat", "ore"]
    
    # Validate list lengths
    if len(offer) != 5 or len(ask) != 5:
        return {
            "error": "Both offer and ask must be 5-element lists [wood, brick, sheep, wheat, ore]",
            "offer_received": offer,
            "ask_received": ask,
        }
    
    # Check player has enough resources
    player_resources = get_player_freqdeck(ctx.deps.game.state, ctx.deps.color)
    for i, (have, offering) in enumerate(zip(player_resources, offer)):
        if offering > have:
            return {
                "error": f"Insufficient {resource_names[i]}. Have: {have}, Offering: {offering}",
                "your_resources": dict(zip(resource_names, player_resources)),
            }
    
    # Validate trade semantics
    if sum(offer) == 0 or sum(ask) == 0:
        return {"error": "Cannot make a trade with nothing offered or nothing asked"}
    
    for o, a in zip(offer, ask):
        if o > 0 and a > 0:
            return {"error": "Cannot offer and ask for the same resource type"}
    
    # Create the trade action
    trade_value = tuple(offer + ask)
    
    if ctx.deps.player_instance is not None:
        ctx.deps.player_instance._pending_trade_action = Action(
            ctx.deps.color, 
            ActionType.OFFER_TRADE, 
            trade_value
        )
    
    return {
        "success": True,
        "trade_created": True,
        "offering": {resource_names[i]: v for i, v in enumerate(offer) if v > 0},
        "asking": {resource_names[i]: v for i, v in enumerate(ask) if v > 0},
        "message": "Trade offer created. Negotiation will end and offer will be submitted.",
    }


def _initiate_negotiation(ctx: RunContext[CatanDependencies]) -> Dict[str, Any]:
    """Start a negotiation session with other players."""
    player = ctx.deps.player_instance
    if player is None or player.negotiation_manager is None:
        return {
            "error": "Negotiation is not available. No NegotiationManager configured.",
            "suggestion": "You can still make direct trade offers using maritime trade.",
        }
    
    manager = player.negotiation_manager
    turn = ctx.deps.game.state.num_turns
    
    if not manager.can_initiate(ctx.deps.color, turn):
        return {
            "error": "You have already initiated a negotiation this turn.",
            "suggestion": "Use maritime trade or continue with other actions.",
        }
    
    # Start the negotiation - runs the negotiation loop with chat agents
    result = manager.start_negotiation(ctx.deps.color, ctx.deps.game)
    
    if result.get("trade_action"):
        ctx.deps.player_instance._pending_trade_action = result["trade_action"]
        return {
            "success": True,
            "negotiation_completed": True,
            "trade_offer_made": True,
            "message": "Negotiation completed with a trade offer.",
        }
    else:
        return {
            "success": True,
            "negotiation_completed": True,
            "trade_offer_made": False,
            "message": "Negotiation ended without a trade offer.",
        }


def _send_message(ctx: RunContext[CatanDependencies], message: str) -> Dict[str, Any]:
    """Send a message during negotiation."""
    player = ctx.deps.player_instance
    if player is None or player.negotiation_manager is None:
        return {"error": "Not in an active negotiation."}
    
    manager = player.negotiation_manager
    if manager.current_session is None:
        return {"error": "No active negotiation session."}
    
    manager.add_message(ctx.deps.color, message)
    
    return {
        "success": True,
        "message_sent": message,
        "your_color": ctx.deps.color.value,
    }


def _leave_negotiation(ctx: RunContext[CatanDependencies]) -> Dict[str, Any]:
    """Exit the current negotiation."""
    player = ctx.deps.player_instance
    if player is None or player.negotiation_manager is None:
        return {"error": "Not in an active negotiation."}
    
    manager = player.negotiation_manager
    if manager.current_session is None:
        return {"error": "No active negotiation session."}
    
    manager.remove_participant(ctx.deps.color)
    
    return {
        "success": True,
        "message": "You have left the negotiation.",
        "your_color": ctx.deps.color.value,
    }


# ============================================================================
# Toolset Factory Functions
# ============================================================================

def create_reasoning_toolset() -> FunctionToolset:
    """
    Create toolset for main gameplay reasoning.
    
    Contains analysis tools and initiate_negotiation.
    Does NOT contain trade_offer (that's for chat agent only).
    """
    toolset = FunctionToolset()
    
    @toolset.tool
    def get_game_and_action_analysis(ctx: RunContext[CatanDependencies]) -> Dict[str, Any]:
        """
        Get comprehensive game state, available actions, and strategy recommendation.
        
        Use this tool FIRST on every turn to understand the current situation.
        Returns game state summary, list of available actions with indices,
        and strategy advisor recommendation if configured.
        """
        return _get_game_and_action_analysis(ctx)
    
    @toolset.tool
    def analyze_board(ctx: RunContext[CatanDependencies], focus: str) -> Dict[str, Any]:
        """
        Analyze the board for strategic insights.

        Args:
            focus: Area to analyze - 'expansion' (build spots), 'blocking' (opponent blocking),
                   'ports' (trading options), or 'robber' (placement options)
        """
        return _analyze_board(ctx, focus)
    
    @toolset.tool
    def initiate_negotiation(ctx: RunContext[CatanDependencies]) -> Dict[str, Any]:
        """
        Start a negotiation chat with other players before making a trade.
        
        Call this before using maritime trade. Opens a chat session where you
        can discuss trades with other LLM players. The negotiation proceeds in
        round-robin order until you make a trade offer or it times out.
        
        Can only be used once per turn.
        """
        return _initiate_negotiation(ctx)
    
    return toolset


def create_chat_initiator_toolset() -> FunctionToolset:
    """
    Create toolset for the negotiation initiator's chat agent.
    
    Contains analysis tools, send_message, and trade_offer.
    The initiator can end negotiation by making a trade offer.
    """
    toolset = FunctionToolset()
    
    @toolset.tool
    def get_game_and_action_analysis(ctx: RunContext[CatanDependencies]) -> Dict[str, Any]:
        """
        Get your current game state and resources.
        
        Use this to understand what resources you have and what you need
        when negotiating trades.
        """
        return _get_game_and_action_analysis(ctx)
    
    @toolset.tool
    def analyze_board(ctx: RunContext[CatanDependencies], focus: str) -> Dict[str, Any]:
        """
        Analyze the board for strategic insights.

        Args:
            focus: Area to analyze - 'expansion', 'blocking', 'ports', or 'robber'
        """
        return _analyze_board(ctx, focus)
    
    @toolset.tool
    def send_message(ctx: RunContext[CatanDependencies], message: str) -> Dict[str, Any]:
        """
        Send a message to other players in the negotiation.
        
        Use this to propose trades, respond to offers, or negotiate terms.
        Be specific about what resources you want and what you're offering.
        
        Args:
            message: Your message to other players (natural language)
        """
        return _send_message(ctx, message)
    
    @toolset.tool
    def trade_offer(
        ctx: RunContext[CatanDependencies], 
        offer: List[int], 
        ask: List[int]
    ) -> Dict[str, Any]:
        """
        Make a formal trade offer to end the negotiation.
        
        This creates the actual trade that other players will accept or reject.
        Calling this ends the negotiation and submits the offer to the game.
        
        Args:
            offer: Resources you give [wood, brick, sheep, wheat, ore] as counts
            ask: Resources you want [wood, brick, sheep, wheat, ore] as counts
            
        Example: offer=[0,0,2,0,0] ask=[0,0,0,1,0] means give 2 sheep for 1 wheat
        """
        return _make_trade_offer(ctx, offer, ask)
    
    return toolset


def create_chat_participant_toolset() -> FunctionToolset:
    """
    Create toolset for negotiation participants (non-initiators).
    
    Contains analysis tools, send_message, and leave_negotiation.
    Participants cannot make trade offers, only discuss and leave.
    """
    toolset = FunctionToolset()
    
    @toolset.tool
    def get_game_and_action_analysis(ctx: RunContext[CatanDependencies]) -> Dict[str, Any]:
        """
        Get your current game state and resources.
        
        Use this to understand what resources you have and what you need
        when negotiating trades.
        """
        return _get_game_and_action_analysis(ctx)
    
    @toolset.tool
    def analyze_board(ctx: RunContext[CatanDependencies], focus: str) -> Dict[str, Any]:
        """
        Analyze the board for strategic insights.

        Args:
            focus: Area to analyze - 'expansion', 'blocking', 'ports', or 'robber'
        """
        return _analyze_board(ctx, focus)
    
    @toolset.tool
    def send_message(ctx: RunContext[CatanDependencies], message: str) -> Dict[str, Any]:
        """
        Send a message to other players in the negotiation.
        
        Use this to respond to trade proposals, make counter-offers,
        or indicate your interest in trading.
        
        Args:
            message: Your message to other players (natural language)
        """
        return _send_message(ctx, message)
    
    @toolset.tool
    def leave_negotiation(ctx: RunContext[CatanDependencies]) -> Dict[str, Any]:
        """
        Exit the negotiation early.
        
        Use this if you're not interested in trading or the proposed
        trades don't benefit you. You won't receive further messages.
        """
        return _leave_negotiation(ctx)
    
    return toolset


# ============================================================================
# Pre-built Toolset Instances
# ============================================================================

REASONING_TOOLSET = create_reasoning_toolset()
CHAT_INITIATOR_TOOLSET = create_chat_initiator_toolset()
CHAT_PARTICIPANT_TOOLSET = create_chat_participant_toolset()


# ============================================================================
# Toolset Selection Helper
# ============================================================================

def get_toolset_for_context(
    is_negotiation: bool = False,
    is_initiator: bool = False,
) -> FunctionToolset:
    """
    Get the appropriate toolset based on context.
    
    Args:
        is_negotiation: Whether currently in a negotiation session
        is_initiator: Whether this player initiated the negotiation
    
    Returns:
        The appropriate FunctionToolset instance
    """
    if is_negotiation:
        if is_initiator:
            return CHAT_INITIATOR_TOOLSET
        else:
            return CHAT_PARTICIPANT_TOOLSET
    else:
        return REASONING_TOOLSET
