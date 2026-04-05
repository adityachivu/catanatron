"""
PydanticAI Toolsets for the Catan LLM agent.

Tool functions are plain functions with RunContext[CatanDependencies] as
their first argument. Toolsets are composed from these functions using
FunctionToolset(tools=[...]) and selected at agent.run() time based on
game state.

Trade offers are NOT a standalone tool. The only path to a domestic trade is:
1. Player calls initiate_negotiation during normal play
2. NegotiationManager runs messaging rounds (all players use NEGOTIATION_PARTICIPANT_TOOLSET)
3. After messaging, initiator uses TRADE_FINALIZE_TOOLSET to specify the final offer
4. The resulting OFFER_TRADE action is returned to the game engine

Toolsets:
- NORMAL_PLAY_TOOLSET: Analysis tools only (pre-roll or no trade available)
- NORMAL_PLAY_WITH_TRADE_TOOLSET: Analysis + initiate_negotiation (after rolling)
- NEGOTIATION_PARTICIPANT_TOOLSET: Analysis + chat tools (during negotiation messaging)
- TRADE_FINALIZE_TOOLSET: Analysis + finalize_trade (post-negotiation trade decision)
"""

from typing import List, Dict, Any
from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

from catanatron.state_functions import (
    get_visible_victory_points,
    get_player_freqdeck,
)
from catanatron.models.enums import CITY, ActionType, Action
from catanatron.models.map import LandTile

# Import CatanDependencies directly for type hints in toolsets
# This is needed because Pydantic AI evaluates type hints at runtime
from catanatron.players.llm.base import CatanDependencies


# ============================================================================
# Tool Functions
# ============================================================================

def get_game_and_action_analysis(ctx: RunContext[CatanDependencies]) -> Dict[str, Any]:
    """
    Get comprehensive game state, available actions, and strategy recommendation.

    Use this tool FIRST on every turn to understand the current game situation
    before making any decisions.
    """
    from catanatron.players.llm.state_formatter import StateFormatter

    full_analysis = {
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
    return full_analysis


def analyze_board(ctx: RunContext[CatanDependencies], focus: str) -> Dict[str, Any]:
    """
    Analyze the board for strategic insights.

    Args:
        focus: What to analyze. Options:
            - 'expansion': Where can I build next, best spots
            - 'blocking': How to block opponents' expansion
            - 'ports': Port accessibility and trading options
            - 'robber': Optimal robber placements
    """
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


def finalize_trade(
    ctx: RunContext[CatanDependencies],
    offer: List[int],
    ask: List[int],
) -> Dict[str, Any]:
    """
    Submit your final trade offer based on the negotiation.

    This creates an OFFER_TRADE action that will be presented to other
    players for acceptance or rejection. Validates resource counts,
    ownership, and trade legality before creating the action.

    Args:
        offer: Resources you are offering [wood, brick, sheep, wheat, ore]
        ask: Resources you want in return [wood, brick, sheep, wheat, ore]
    """
    # Validate offer and ask are 5-element lists
    if len(offer) != 5 or len(ask) != 5:
        return {
            "error": "Both offer and ask must be 5-element lists [wood, brick, sheep, wheat, ore]",
            "offer_received": offer,
            "ask_received": ask,
        }

    # Check player has enough resources to offer
    player_resources = get_player_freqdeck(ctx.deps.game.state, ctx.deps.color)
    for i, (have, offering) in enumerate(zip(player_resources, offer)):
        if offering > have:
            resource_names = ["wood", "brick", "sheep", "wheat", "ore"]
            return {
                "error": f"You don't have enough {resource_names[i]} to offer. Have: {have}, Offering: {offering}",
                "your_resources": dict(zip(resource_names, player_resources)),
            }

    # Validate trade (can't give away for free, can't trade same resource)
    if sum(offer) == 0 or sum(ask) == 0:
        return {"error": "Cannot make a trade with nothing offered or nothing asked"}

    for o, a in zip(offer, ask):
        if o > 0 and a > 0:
            return {"error": "Cannot offer and ask for the same resource type"}

    # Create the trade action value tuple
    trade_value = tuple(offer + ask)

    # Store the pending trade action on the player instance
    # The decide() method will check for this and return it
    if ctx.deps.player_instance is not None:
        ctx.deps.player_instance._pending_trade_action = Action(
            ctx.deps.color,
            ActionType.OFFER_TRADE,
            trade_value,
        )

    resource_names = ["wood", "brick", "sheep", "wheat", "ore"]
    return {
        "success": True,
        "trade_created": True,
        "offering": {resource_names[i]: v for i, v in enumerate(offer) if v > 0},
        "asking": {resource_names[i]: v for i, v in enumerate(ask) if v > 0},
        "message": "Trade offer will be submitted. Other players will now decide whether to accept.",
    }


def initiate_negotiation(ctx: RunContext[CatanDependencies]) -> Dict[str, Any]:
    """
    !--Always use this tool before doing a MARITIME_TRADE. Play MARITIME_TRADE !ONLY! as a last resort. VERY IMPORTANT.--!

    Start a negotiation with other players to arrange a domestic trade.

    This opens a shared chat room where all players discuss potential trades
    in round-robin order for up to k rounds. After messaging concludes, you
    will specify your final trade offer based on the conversation.

    Note: You can only initiate one negotiation per turn. Once the negotiation
    concludes (whether or not a trade is made), no further trading is allowed
    this turn.
    """
    player = ctx.deps.player_instance
    if player is None or player.negotiation_manager is None:
        return {
            "error": "Negotiation is not available. No NegotiationManager configured.",
            "suggestion": "Consider using maritime trade actions from the available actions list.",
        }

    manager = player.negotiation_manager
    turn = ctx.deps.game.state.num_turns

    if not manager.can_initiate(ctx.deps.color, turn):
        return {
            "error": "You have already initiated a negotiation this turn. No further trading is allowed.",
        }

    # Set flag - decide() will handle this AFTER run_sync() completes
    # to avoid nested run_sync() calls which cause event loop binding errors
    player._pending_negotiation_request = True

    return {
        "success": True,
        "negotiation_will_start": True,
        "message": "Negotiation will start now. You will chat with other players, then specify your trade offer.",
    }


def send_message(ctx: RunContext[CatanDependencies], message: str) -> Dict[str, Any]:
    """
    Send a message to other players during negotiation.

    Use this to communicate your trading intentions, make proposals,
    or respond to other players' offers.

    Args:
        message: Your message to other players (free-form natural language)
    """
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


def leave_negotiation(ctx: RunContext[CatanDependencies]) -> Dict[str, Any]:
    """
    Exit the current negotiation early.

    Use this if you don't want to continue participating in the negotiation.
    Note: Only the initiator can end the negotiation with a trade offer.
    """
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
# Board Analysis Helpers (private, used by analyze_board tool)
# ============================================================================

def _analyze_expansion(game, my_color) -> Dict[str, Any]:
    """Analyze expansion opportunities."""
    board = game.state.board

    buildable_nodes = list(board.buildable_node_ids(my_color))
    buildable_edges = [list(e) for e in board.buildable_edges(my_color)]

    node_assessments = []
    for node_id in buildable_nodes[:5]:
        tiles = board.map.adjacent_tiles.get(node_id, [])
        resources = []
        total_prob = 0
        for tile in tiles:
            if isinstance(tile, LandTile) and tile.resource:
                from catanatron.models.map import number_probability

                prob = number_probability(tile.number) if tile.number else 0
                resources.append(
                    {"resource": tile.resource, "number": tile.number, "probability": prob}
                )
                total_prob += prob

        node_assessments.append(
            {
                "node_id": node_id,
                "resources": resources,
                "total_production_probability": round(total_prob, 3),
            }
        )

    node_assessments.sort(
        key=lambda x: x["total_production_probability"], reverse=True
    )

    return {
        "buildable_settlement_locations": len(buildable_nodes),
        "buildable_road_locations": len(buildable_edges),
        "best_settlement_spots": node_assessments,
        "expansion_advice": (
            "Focus on high-probability numbers (6, 8, 5, 9) and resource diversity"
            if buildable_nodes
            else "No settlement spots available - build roads to expand"
        ),
    }


def _analyze_blocking(game, my_color) -> Dict[str, Any]:
    """Analyze blocking opportunities."""
    state = game.state
    board = state.board

    blocking_opportunities = []
    for color in state.colors:
        if color == my_color:
            continue

        their_buildable = list(board.buildable_node_ids(color, initial_build_phase=False))
        my_buildable = set(board.buildable_node_ids(my_color))

        overlap = [n for n in their_buildable if n in my_buildable]
        if overlap:
            blocking_opportunities.append(
                {
                    "opponent": color.value,
                    "blockable_nodes": overlap[:3],
                    "their_vps": get_visible_victory_points(state, color),
                }
            )

    return {
        "blocking_opportunities": blocking_opportunities,
        "advice": (
            "Consider blocking the leading player's expansion"
            if blocking_opportunities
            else "No immediate blocking opportunities"
        ),
    }


def _analyze_ports(game, my_color) -> Dict[str, Any]:
    """Analyze port access and trading."""
    board = game.state.board
    my_ports = list(board.get_player_port_resources(my_color))

    port_info = {
        "current_ports": my_ports,
        "has_3_to_1_port": None in my_ports,
        "specialized_ports": [p for p in my_ports if p is not None],
    }

    rates = {"wood": 4, "brick": 4, "sheep": 4, "wheat": 4, "ore": 4}
    if None in my_ports:
        rates = {r: 3 for r in rates}
    for port_resource in my_ports:
        if port_resource:
            rates[port_resource.lower()] = 2

    port_info["trading_rates"] = rates
    port_info["advice"] = (
        "Good port access - consider using maritime trades"
        if len(my_ports) > 0
        else "No ports yet - consider building towards coastal settlements"
    )

    return port_info


def _analyze_robber(game, my_color) -> Dict[str, Any]:
    """Analyze robber placement options."""
    state = game.state
    board = state.board

    placements = []
    for coord, tile in board.map.tiles.items():
        if not isinstance(tile, LandTile) or tile.resource is None:
            continue
        if coord == board.robber_coordinate:
            continue

        affected = {}
        for node_id in tile.nodes.values():
            building = board.buildings.get(node_id)
            if building:
                color, btype = building
                if color != my_color:
                    if color.value not in affected:
                        affected[color.value] = 0
                    affected[color.value] += 2 if btype == CITY else 1

        if affected:
            from catanatron.models.map import number_probability

            prob = number_probability(tile.number) if tile.number else 0
            placements.append(
                {
                    "coordinate": coord,
                    "resource": tile.resource,
                    "number": tile.number,
                    "probability": prob,
                    "affected_players": affected,
                    "total_impact": sum(affected.values()) * prob,
                }
            )

    placements.sort(key=lambda x: x["total_impact"], reverse=True)

    return {
        "best_robber_placements": placements[:5],
        "current_robber_location": board.robber_coordinate,
        "advice": "Target the leading player's highest-production tile",
    }


# ============================================================================
# Toolset Composition
# ============================================================================

NORMAL_PLAY_TOOLSET = FunctionToolset(tools=[
    get_game_and_action_analysis,
    analyze_board,
])

NORMAL_PLAY_WITH_TRADE_TOOLSET = FunctionToolset(tools=[
    get_game_and_action_analysis,
    analyze_board,
    initiate_negotiation,
])

NEGOTIATION_PARTICIPANT_TOOLSET = FunctionToolset(tools=[
    get_game_and_action_analysis,
    analyze_board,
    send_message,
    leave_negotiation,
])

TRADE_FINALIZE_TOOLSET = FunctionToolset(tools=[
    get_game_and_action_analysis,
    analyze_board,
    finalize_trade,
])


# ============================================================================
# Toolset Selection Helper
# ============================================================================

def get_toolsets_for_game_state(
    game,
    color,
    has_rolled: bool,
    is_my_turn: bool,
    in_negotiation: bool = False,
    is_finalization_phase: bool = False,
    negotiation_enabled: bool = True,
) -> List[FunctionToolset]:
    """
    Select appropriate toolsets based on game state.

    Args:
        game: The current game instance
        color: The player's color
        has_rolled: Whether the player has rolled this turn
        is_my_turn: Whether it's currently this player's turn
        in_negotiation: Whether currently in a negotiation messaging phase
        is_finalization_phase: Whether in post-negotiation trade finalization
        negotiation_enabled: Whether negotiation feature is enabled

    Returns:
        List of FunctionToolset instances to use for agent.run()
    """
    if is_finalization_phase:
        return [TRADE_FINALIZE_TOOLSET]

    if in_negotiation:
        return [NEGOTIATION_PARTICIPANT_TOOLSET]

    if is_my_turn and has_rolled and negotiation_enabled:
        return [NORMAL_PLAY_WITH_TRADE_TOOLSET]
    else:
        return [NORMAL_PLAY_TOOLSET]
