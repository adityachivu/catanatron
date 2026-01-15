"""
PydanticAI Toolsets for the Catan LLM agent.

This module provides a toolset-based architecture for managing agent tools.
Tools are organized into logical toolsets that can be composed and selected
at runtime based on game state.

Toolsets:
- ANALYSIS_TOOLSET: Game state analysis tools
- TRADE_TOOLSET: Trade offer and negotiation initiation tools
- CHAT_TOOLSET: Negotiation messaging tools
- NORMAL_PLAY_TOOLSET: Analysis tools for normal gameplay
- NORMAL_PLAY_WITH_TRADE_TOOLSET: Analysis + trade tools when trading is available
- NEGOTIATION_PARTICIPANT_TOOLSET: Tools for players participating in negotiation
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass

from pydantic_ai import Tool, RunContext

from catanatron.models.enums import Action, ActionType, RESOURCES
from catanatron.models.map import LandTile, number_probability
from catanatron.state_functions import (
    get_visible_victory_points,
    get_player_freqdeck,
)

if TYPE_CHECKING:
    from catanatron.players.llm.base import CatanDependencies
    from catanatron.players.llm.negotiation import NegotiationManager


# ============ Extended Dependencies for Negotiation ============

@dataclass
class NegotiationDependencies:
    """
    Extended dependencies for negotiation context.
    
    Includes all CatanDependencies fields plus negotiation-specific context.
    """
    # Base dependencies
    color: Any  # Color
    game: Any  # Game
    playable_actions: List[Action]
    strategy_recommendation: Optional[Action]
    strategy_reasoning: Optional[str]
    turn_number: int
    is_my_turn: bool
    
    # Negotiation-specific
    negotiation_manager: Optional["NegotiationManager"] = None
    player_instance: Optional[Any] = None  # BaseLLMPlayer reference


# ============ Tool Implementation Functions ============
# These are plain functions that will be wrapped with Tool()

def get_game_and_action_analysis(ctx: RunContext) -> Dict[str, Any]:
    """
    Get comprehensive game state, available actions, and strategy recommendation.
    
    Use this IMMEDIATELY at the start of your turn to understand the game state
    before making any decisions.
    """
    from catanatron.players.llm.state_formatter import StateFormatter
    
    deps = ctx.deps
    return {
        "game_state": StateFormatter.format_full_state(deps.game, deps.color),
        "available_actions": [
            StateFormatter.format_action(action, i) 
            for i, action in enumerate(deps.playable_actions)
        ],
        "strategy_recommendation": (
            deps.strategy_recommendation 
            if deps.strategy_recommendation 
            else "No strategy advisor configured"
        ),
        "strategy_reasoning": (
            deps.strategy_reasoning 
            if deps.strategy_reasoning 
            else "No detailed reasoning available"
        ),
    }


def analyze_board(ctx: RunContext, focus: str) -> Dict[str, Any]:
    """
    Analyze the board for strategic insights.

    Args:
        focus: What to analyze. Options:
            - 'expansion': Where can I build next, best spots
            - 'blocking': How to block opponents' expansion
            - 'ports': Port accessibility and trading options
            - 'robber': Optimal robber placements
    """
    from catanatron.players.llm.tools import (
        _analyze_expansion,
        _analyze_blocking,
        _analyze_ports,
        _analyze_robber,
    )
    
    deps = ctx.deps
    game = deps.game
    my_color = deps.color

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


def make_trade_offer(
    ctx: RunContext,
    offer: List[int],
    ask: List[int],
    target_player: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create an OFFER_TRADE action to trade resources with other players.
    
    This will end any active negotiation and submit the formal trade offer.
    
    Args:
        offer: Resources you're offering [wood, brick, sheep, wheat, ore]
        ask: Resources you want in return [wood, brick, sheep, wheat, ore]
        target_player: Optional color of specific player to trade with (e.g., "RED", "BLUE")
    
    Returns:
        Dictionary with trade action details or error if invalid.
    """
    deps = ctx.deps
    
    # Validate offer/ask format
    if len(offer) != 5 or len(ask) != 5:
        return {
            "error": "offer and ask must each be 5-element lists [wood, brick, sheep, wheat, ore]",
            "offer_received": offer,
            "ask_received": ask,
        }
    
    # Check if any resources are being exchanged
    if sum(offer) == 0 or sum(ask) == 0:
        return {
            "error": "Must offer and ask for at least some resources",
            "offer": offer,
            "ask": ask,
        }
    
    # Verify player has the resources to offer
    my_resources = get_player_freqdeck(deps.game.state, deps.color)
    resource_names = ["wood", "brick", "sheep", "wheat", "ore"]
    
    insufficient = []
    for i, (have, offering) in enumerate(zip(my_resources, offer)):
        if offering > have:
            insufficient.append(f"{resource_names[i]}: have {have}, offering {offering}")
    
    if insufficient:
        return {
            "error": "Insufficient resources to make this offer",
            "problems": insufficient,
            "your_resources": dict(zip(resource_names, my_resources)),
        }
    
    # Create the trade tuple (offer + ask = 10 elements)
    trade_value = tuple(offer + ask)
    
    # Find the OFFER_TRADE action in playable actions
    for i, action in enumerate(deps.playable_actions):
        if action.action_type == ActionType.OFFER_TRADE and action.value == trade_value:
            # End any active negotiation
            if hasattr(deps, 'negotiation_manager') and deps.negotiation_manager:
                if deps.negotiation_manager.current_session:
                    deps.negotiation_manager.end_negotiation()
            
            # Store the action for the player to return
            if hasattr(deps, 'player_instance') and deps.player_instance:
                deps.player_instance._pending_trade_action = action
            
            return {
                "success": True,
                "action_index": i,
                "trade_offer": {
                    "offering": dict(zip(resource_names, offer)),
                    "asking": dict(zip(resource_names, ask)),
                },
                "message": "Trade offer created. Other players will now decide whether to accept.",
            }
    
    # OFFER_TRADE not in available actions - might need to construct it
    # Check if OFFER_TRADE is generally available
    offer_trade_available = any(
        a.action_type == ActionType.OFFER_TRADE 
        for a in deps.playable_actions
    )
    
    if not offer_trade_available:
        return {
            "error": "Cannot make trade offers right now",
            "reason": "OFFER_TRADE is not among available actions",
            "hint": "You may need to roll first, or trading might not be allowed in this phase",
        }
    
    # The specific trade is valid but not in pre-generated list
    # Create the action directly
    trade_action = Action(deps.color, ActionType.OFFER_TRADE, trade_value)
    
    # End any active negotiation
    if hasattr(deps, 'negotiation_manager') and deps.negotiation_manager:
        if deps.negotiation_manager.current_session:
            deps.negotiation_manager.end_negotiation()
    
    # Store for player to return
    if hasattr(deps, 'player_instance') and deps.player_instance:
        deps.player_instance._pending_trade_action = trade_action
    
    return {
        "success": True,
        "trade_offer": {
            "offering": dict(zip(resource_names, offer)),
            "asking": dict(zip(resource_names, ask)),
        },
        "message": "Trade offer created. Other players will now decide whether to accept.",
    }


def initiate_negotiation(ctx: RunContext) -> Dict[str, Any]:
    """
    Start a negotiation chat with other LLM players before making a trade offer.
    
    This allows you to discuss potential trades with other players before
    committing to a formal offer. All LLM players will automatically join
    the negotiation.
    
    Returns:
        Dictionary with negotiation session info or error if cannot initiate.
    """
    deps = ctx.deps
    
    # Check if negotiation manager is available
    if not hasattr(deps, 'negotiation_manager') or deps.negotiation_manager is None:
        return {
            "error": "Negotiation not available",
            "reason": "No negotiation manager configured for this game",
            "hint": "Use trade_offer directly to make trades",
        }
    
    manager = deps.negotiation_manager
    
    # Check if we can initiate
    if not manager.can_initiate(deps.color, deps.turn_number):
        return {
            "error": "Cannot initiate negotiation",
            "reason": "You have already initiated a negotiation this turn",
            "hint": "You can only start one negotiation per turn",
        }
    
    # Check if there are other LLM players to negotiate with
    other_llm_players = [
        color for color in manager.players.keys() 
        if color != deps.color
    ]
    
    if not other_llm_players:
        return {
            "error": "No other LLM players to negotiate with",
            "reason": "Negotiation requires at least one other LLM player",
            "hint": "Use trade_offer to trade with non-LLM players",
        }
    
    # Start the negotiation
    session = manager.start_negotiation(deps.color, deps.game)
    
    return {
        "success": True,
        "session_started": True,
        "participants": [c.value for c in session.participants],
        "max_rounds": session.max_rounds,
        "message": (
            f"Negotiation started with {len(session.participants)} players. "
            "Use send_message to communicate, then trade_offer to make an offer."
        ),
        "hint": "You speak first. Other players will respond in turn order.",
    }


def send_message(ctx: RunContext, message: str) -> Dict[str, Any]:
    """
    Send a message to other players during negotiation.
    
    Use this to communicate your trading intentions, ask what others need,
    or negotiate terms before making a formal offer.
    
    Args:
        message: The message to send to other players.
    
    Returns:
        Dictionary with message confirmation or error.
    """
    deps = ctx.deps
    
    if not hasattr(deps, 'negotiation_manager') or deps.negotiation_manager is None:
        return {
            "error": "Not in a negotiation",
            "hint": "Use initiate_negotiation first to start a negotiation session",
        }
    
    manager = deps.negotiation_manager
    
    if manager.current_session is None:
        return {
            "error": "No active negotiation session",
            "hint": "Use initiate_negotiation first to start a negotiation session",
        }
    
    session = manager.current_session
    
    # Verify it's this player's turn to speak
    current_speaker = session.participants[session.turn_index]
    if current_speaker != deps.color:
        return {
            "error": "Not your turn to speak",
            "current_speaker": current_speaker.value,
            "your_color": deps.color.value,
        }
    
    # Add the message
    manager.add_message(deps.color, message)
    
    return {
        "success": True,
        "message_sent": message,
        "your_color": deps.color.value,
        "messages_so_far": len(session.messages),
        "round": session.current_round,
        "hint": "Wait for other players to respond, then continue negotiating or make a trade_offer",
    }


def leave_negotiation(ctx: RunContext) -> Dict[str, Any]:
    """
    Exit the current negotiation early without making a trade offer.
    
    Use this if you decide you don't want to trade after all, or if
    negotiations aren't going well.
    
    Returns:
        Dictionary confirming exit from negotiation.
    """
    deps = ctx.deps
    
    if not hasattr(deps, 'negotiation_manager') or deps.negotiation_manager is None:
        return {
            "error": "Not in a negotiation",
        }
    
    manager = deps.negotiation_manager
    
    if manager.current_session is None:
        return {
            "error": "No active negotiation to leave",
        }
    
    # Mark this player as leaving
    manager.player_leaves(deps.color)
    
    return {
        "success": True,
        "message": "You have left the negotiation",
        "negotiation_ended": manager.current_session is None,
    }


# ============ Tool Wrappers ============

game_analysis_tool = Tool(
    get_game_and_action_analysis,
    name="get_game_and_action_analysis",
    description=(
        "Get comprehensive game state, available actions, and strategy recommendation. "
        "Use this IMMEDIATELY at the start of your turn before making any decisions."
    ),
)

board_analysis_tool = Tool(
    analyze_board,
    name="analyze_board",
    description=(
        "Analyze the board for strategic insights. "
        "Options: 'expansion' (where to build), 'blocking' (block opponents), "
        "'ports' (trading access), 'robber' (optimal robber placement)."
    ),
)

trade_offer_tool = Tool(
    make_trade_offer,
    name="trade_offer",
    description=(
        "Make a formal trade offer to other players. "
        "Format: offer=[wood,brick,sheep,wheat,ore], ask=[wood,brick,sheep,wheat,ore]. "
        "This ends any active negotiation and submits the offer."
    ),
)

initiate_negotiation_tool = Tool(
    initiate_negotiation,
    name="initiate_negotiation",
    description=(
        "Start a negotiation chat with other LLM players before making a trade offer. "
        "Allows discussing trades before committing. Can only be done once per turn."
    ),
)

send_message_tool = Tool(
    send_message,
    name="send_message",
    description=(
        "Send a message to other players during negotiation. "
        "Use to communicate trading intentions or negotiate terms."
    ),
)

leave_negotiation_tool = Tool(
    leave_negotiation,
    name="leave_negotiation",
    description=(
        "Exit the current negotiation early without making a trade offer."
    ),
)


# ============ Toolsets ============

# Analysis-only toolset
ANALYSIS_TOOLSET = [
    game_analysis_tool,
    board_analysis_tool,
]

# Trade tools (initiation and offer)
TRADE_TOOLSET = [
    trade_offer_tool,
    initiate_negotiation_tool,
]

# Chat tools for during negotiation
CHAT_TOOLSET = [
    send_message_tool,
    leave_negotiation_tool,
]

# Combined toolsets for common scenarios

# Normal gameplay without trade options
NORMAL_PLAY_TOOLSET = [
    game_analysis_tool,
    board_analysis_tool,
]

# Normal gameplay with trade options available
NORMAL_PLAY_WITH_TRADE_TOOLSET = [
    game_analysis_tool,
    board_analysis_tool,
    trade_offer_tool,
    initiate_negotiation_tool,
]

# For players participating in an active negotiation (not the initiator)
NEGOTIATION_PARTICIPANT_TOOLSET = [
    game_analysis_tool,
    board_analysis_tool,
    send_message_tool,
    leave_negotiation_tool,
]

# For the initiator during negotiation (can send messages and make offers)
NEGOTIATION_INITIATOR_TOOLSET = [
    game_analysis_tool,
    board_analysis_tool,
    send_message_tool,
    leave_negotiation_tool,
    trade_offer_tool,
]


def get_all_tools() -> List[Tool]:
    """Return all available tools for registration."""
    return [
        game_analysis_tool,
        board_analysis_tool,
        trade_offer_tool,
        initiate_negotiation_tool,
        send_message_tool,
        leave_negotiation_tool,
    ]
