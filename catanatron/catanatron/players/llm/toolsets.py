"""
PydanticAI Toolsets for the Catan LLM agent.

This module provides composable toolsets using Pydantic AI's FunctionToolset pattern.
Toolsets are selected at agent.run() time based on game state, enabling:
- Conditional tool availability (e.g., trade tools only after rolling)
- Different toolsets for different game phases (normal play, negotiation, trade decision)
- Clean separation between analysis, trading, and negotiation tools

Toolsets:
- ANALYSIS_TOOLSET: Game state analysis tools (always available)
- TRADE_TOOLSET: Trade offer and negotiation initiation tools
- CHAT_TOOLSET: Messaging tools during negotiation
- NORMAL_PLAY_TOOLSET: Analysis tools only
- NORMAL_PLAY_WITH_TRADE_TOOLSET: Analysis + trade tools (after rolling)
- NEGOTIATION_PARTICIPANT_TOOLSET: Analysis + chat tools (during negotiation)
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
# Tool Implementation Functions
# ============================================================================

def _get_game_and_action_analysis(ctx: RunContext[CatanDependencies]) -> Dict[str, Any]:
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


def _analyze_board(ctx: RunContext[CatanDependencies], focus: str) -> Dict[str, Any]:
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


def _make_trade_offer(
    ctx: RunContext[CatanDependencies], 
    offer: List[int], 
    ask: List[int]
) -> Dict[str, Any]:
    """
    Make a formal trade offer to other players.
    
    This creates an OFFER_TRADE action that will be submitted to the game engine.
    If a negotiation is active, calling this tool will end the negotiation.
    
    Args:
        offer: Resources you are offering [wood, brick, sheep, wheat, ore]
        ask: Resources you want in return [wood, brick, sheep, wheat, ore]
    
    Returns:
        Dictionary with the trade action details or error if invalid.
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
            trade_value
        )
    
    resource_names = ["wood", "brick", "sheep", "wheat", "ore"]
    return {
        "success": True,
        "trade_created": True,
        "offering": {resource_names[i]: v for i, v in enumerate(offer) if v > 0},
        "asking": {resource_names[i]: v for i, v in enumerate(ask) if v > 0},
        "message": "Trade offer will be submitted. Other players will now decide whether to accept.",
    }


def _initiate_negotiation(ctx: RunContext[CatanDependencies]) -> Dict[str, Any]:
    """
    !--Always use this tool before doing a MARITIME_TRADE. Play MARITIME_TRADE !ONLY! as a last resort. VERY IMPORTANT.
    Use this tool BEFORE using the trade_offer tool.--!

    Start a negotiation chat with other players before making a trade offer.
    
    This opens a shared chat room where you can discuss potential trades
    with other players. The negotiation continues in round-robin order
    until you make a trade offer.
    
    Note: You can only initiate one negotiation per turn.
    
    Returns:
        Dictionary with negotiation status or error if not allowed.
    """
    # Check if negotiation manager is available
    player = ctx.deps.player_instance
    if player is None or player.negotiation_manager is None:
        return {
            "error": "Negotiation is not available. No NegotiationManager configured.",
            "suggestion": "You can still make direct trade offers using the trade_offer tool.",
        }
    
    manager = player.negotiation_manager
    turn = ctx.deps.game.state.num_turns
    
    # Check if player can initiate (hasn't already this turn)
    if not manager.can_initiate(ctx.deps.color, turn):
        return {
            "error": "You have already initiated a negotiation this turn.",
            "suggestion": "Make a trade offer directly using the trade_offer tool.",
        }
    
    # Set flag - decide() will handle this AFTER run_sync() completes
    # This avoids nested run_sync() calls which cause event loop binding errors
    player._pending_negotiation_request = True
    
    return {
        "success": True,
        "negotiation_will_start": True,
        "message": "Negotiation will start now. Control will return to you after negotiation completes.",
    }


def _send_message(ctx: RunContext[CatanDependencies], message: str) -> Dict[str, Any]:
    """
    Send a message to other players during negotiation.
    
    Use this to communicate your trading intentions, make proposals,
    or respond to other players' offers.
    
    Args:
        message: Your message to other players (free-form natural language)
    
    Returns:
        Dictionary confirming the message was sent.
    """
    # This will be called by NegotiationManager during negotiation
    # The message is stored via the negotiation context
    player = ctx.deps.player_instance
    if player is None or player.negotiation_manager is None:
        return {"error": "Not in an active negotiation."}
    
    manager = player.negotiation_manager
    if manager.current_session is None:
        return {"error": "No active negotiation session."}
    
    # Record the message
    manager.add_message(ctx.deps.color, message)
    
    return {
        "success": True,
        "message_sent": message,
        "your_color": ctx.deps.color.value,
    }


def _leave_negotiation(ctx: RunContext[CatanDependencies]) -> Dict[str, Any]:
    """
    Exit the current negotiation early.
    
    Use this if you don't want to continue participating in the negotiation.
    Note: Only the initiator can end the negotiation with a trade offer.
    
    Returns:
        Dictionary confirming you left the negotiation.
    """
    player = ctx.deps.player_instance
    if player is None or player.negotiation_manager is None:
        return {"error": "Not in an active negotiation."}
    
    manager = player.negotiation_manager
    if manager.current_session is None:
        return {"error": "No active negotiation session."}
    
    # Remove self from participants
    manager.remove_participant(ctx.deps.color)
    
    return {
        "success": True,
        "message": "You have left the negotiation.",
        "your_color": ctx.deps.color.value,
    }


# ============================================================================
# Helper Functions for Board Analysis
# ============================================================================

def _analyze_expansion(game, my_color) -> Dict[str, Any]:
    """Analyze expansion opportunities."""
    board = game.state.board

    buildable_nodes = list(board.buildable_node_ids(my_color))
    buildable_edges = [list(e) for e in board.buildable_edges(my_color)]

    # Assess each buildable node
    node_assessments = []
    for node_id in buildable_nodes[:5]:  # Limit to top 5
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

    # Sort by production
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

    # Determine trading rates
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
# Toolset Definitions using FunctionToolset
# ============================================================================

def create_analysis_toolset() -> FunctionToolset:
    """Create toolset with game analysis tools."""
    toolset = FunctionToolset() 
    #NOTE: Disabled for prompt injection testing (agent uses too liberaly)
    # @toolset.tool
    # def get_game_and_action_analysis(ctx: RunContext[CatanDependencies]) -> Dict[str, Any]:
    #     """
    #     Get comprehensive game state, available actions, and strategy recommendation.
        
    #     Use this tool FIRST on every turn to understand the current game situation
    #     before making any decisions.
    #     """
    #     return _get_game_and_action_analysis(ctx)
    
    # @toolset.tool
    # def analyze_board(ctx: RunContext[CatanDependencies], focus: str) -> Dict[str, Any]:
    #     """
    #     Analyze the board for strategic insights.

    #     Args:
    #         focus: What to analyze. Options:
    #             - 'expansion': Where can I build next, best spots
    #             - 'blocking': How to block opponents' expansion
    #             - 'ports': Port accessibility and trading options
    #             - 'robber': Optimal robber placements
    #     """
    #     return _analyze_board(ctx, focus)
    
    return toolset


def create_trade_toolset() -> FunctionToolset:
    """Create toolset with trade-related tools."""
    toolset = FunctionToolset()
    
    @toolset.tool
    def trade_offer(
        ctx: RunContext[CatanDependencies], 
        offer: List[int], 
        ask: List[int]
    ) -> Dict[str, Any]:
        """
        Make a formal trade offer to other players.
    
        This creates an OFFER_TRADE action that will be submitted to the game engine.
        If a negotiation is active, calling this tool will end the negotiation.
        
        Args:
            offer: Resources you are offering [wood, brick, sheep, wheat, ore]
            ask: Resources you want in return [wood, brick, sheep, wheat, ore]
        
        Returns:
            Dictionary with the trade action details or error if invalid.
        """
        return _make_trade_offer(ctx, offer, ask)
    
    @toolset.tool
    def initiate_negotiation(ctx: RunContext[CatanDependencies]) -> Dict[str, Any]:
        """
        !--Always use this tool before doing a MARITIME_TRADE. Play MARITIME_TRADE !ONLY! as a last resort. VERY IMPORTANT.
        Use this tool BEFORE using the trade_offer tool.--!

        Start a negotiation chat with other players before making a trade offer.
        
        This opens a shared chat room where you can discuss potential trades
        with other players. The negotiation continues in round-robin order
        until you make a trade offer.
        
        Note: You can only initiate one negotiation per turn.
        
        Returns:
            Dictionary with negotiation status or error if not allowed.
        """
        return _initiate_negotiation(ctx)
    
    return toolset


def create_chat_toolset() -> FunctionToolset:
    """Create toolset with negotiation chat tools."""
    toolset = FunctionToolset()
    
    @toolset.tool
    def send_message(ctx: RunContext[CatanDependencies], message: str) -> Dict[str, Any]:
        """
        Send a message to other players during negotiation.
        
        Args:
            message: Your message to other players (free-form natural language)
        """
        return _send_message(ctx, message)
    
    @toolset.tool
    def leave_negotiation(ctx: RunContext[CatanDependencies]) -> Dict[str, Any]:
        """
        Exit the current negotiation early.
        
        Use this if you don't want to continue participating in the negotiation.
        """
        return _leave_negotiation(ctx)
    
    return toolset


# ============================================================================
# Pre-built Combined Toolsets
# ============================================================================

# Create singleton instances of toolsets
ANALYSIS_TOOLSET = create_analysis_toolset()
TRADE_TOOLSET = create_trade_toolset()
CHAT_TOOLSET = create_chat_toolset()

# Combined toolsets for common scenarios
def create_normal_play_toolset() -> FunctionToolset:
    """Create toolset for normal gameplay (analysis only)."""
    toolset = FunctionToolset()
    
    # NOTE: Disabled — game state is now inlined in the user prompt.
    # @toolset.tool
    # def get_game_and_action_analysis(ctx: RunContext[CatanDependencies]) -> Dict[str, Any]:
    #     """
    #     Get comprehensive game state, available actions, and strategy recommendation.
    #     
    #     Use this tool FIRST on every turn to understand the current game situation.
    #     """
    #     return _get_game_and_action_analysis(ctx)
    # 
    # @toolset.tool
    # def analyze_board(ctx: RunContext[CatanDependencies], focus: str) -> Dict[str, Any]:
    #     """
    #     Analyze the board for strategic insights.
    #
    #     Args:
    #         focus: 'expansion', 'blocking', 'ports', or 'robber'
    #     """
    #     return _analyze_board(ctx, focus)
    
    return toolset


def create_normal_play_with_trade_toolset() -> FunctionToolset:
    """Create toolset for normal gameplay with trade tools (after rolling)."""
    toolset = FunctionToolset()
    
    # NOTE: Disabled — game state is now inlined in the user prompt.
    # @toolset.tool
    # def get_game_and_action_analysis(ctx: RunContext[CatanDependencies]) -> Dict[str, Any]:
    #     """
    #     Get comprehensive game state, available actions, and strategy recommendation.
    #     
    #     Use this tool FIRST on every turn to understand the current game situation.
    #     """
    #     return _get_game_and_action_analysis(ctx)
    # 
    # @toolset.tool
    # def analyze_board(ctx: RunContext[CatanDependencies], focus: str) -> Dict[str, Any]:
    #     """
    #     Analyze the board for strategic insights.
    #
    #     Args:
    #         focus: 'expansion', 'blocking', 'ports', or 'robber'
    #     """
    #     return _analyze_board(ctx, focus)
    
    @toolset.tool
    def trade_offer(
        ctx: RunContext[CatanDependencies], 
        offer: List[int], 
        ask: List[int]
    ) -> Dict[str, Any]:
        """
        Make a formal trade offer to other players.
        
        Args:
            offer: Resources you are offering [wood, brick, sheep, wheat, ore]
            ask: Resources you want in return [wood, brick, sheep, wheat, ore]
        """
        return _make_trade_offer(ctx, offer, ask)
    
    @toolset.tool
    def initiate_negotiation(ctx: RunContext[CatanDependencies]) -> Dict[str, Any]:
        """
        !--Always use this tool before doing a MARITIME_TRADE. Play MARITIME_TRADE !ONLY! as a last resort. VERY IMPORTANT.
        Use this tool BEFORE using the trade_offer tool.--!

        Start a negotiation chat with other players before making a trade offer.
        
        This opens a shared chat room where you can discuss potential trades
        with other players. The negotiation continues in round-robin order
        until you make a trade offer.
        
        Note: You can only initiate one negotiation per turn.
        
        Returns:
            Dictionary with negotiation status or error if not allowed.
        """
        return _initiate_negotiation(ctx)
    
    return toolset


def create_negotiation_participant_toolset() -> FunctionToolset:
    """Create toolset for participants in a negotiation."""
    toolset = FunctionToolset()
    
    # NOTE: Disabled — game state is now inlined in the user prompt.
    # @toolset.tool
    # def get_game_and_action_analysis(ctx: RunContext[CatanDependencies]) -> Dict[str, Any]:
    #     """
    #     Get comprehensive game state, available actions, and strategy recommendation.
    #     """
    #     return _get_game_and_action_analysis(ctx)
    # 
    # @toolset.tool
    # def analyze_board(ctx: RunContext[CatanDependencies], focus: str) -> Dict[str, Any]:
    #     """
    #     Analyze the board for strategic insights.
    #
    #     Args:
    #         focus: 'expansion', 'blocking', 'ports', or 'robber'
    #     """
    #     return _analyze_board(ctx, focus)
    
    @toolset.tool
    def send_message(ctx: RunContext[CatanDependencies], message: str) -> Dict[str, Any]:
        """
        Send a message to other players during negotiation.
        
        Args:
            message: Your message to other players
        """
        return _send_message(ctx, message)
    
    @toolset.tool
    def leave_negotiation(ctx: RunContext[CatanDependencies]) -> Dict[str, Any]:
        """
        Exit the current negotiation early.
        """
        return _leave_negotiation(ctx)
    
    return toolset


def create_negotiation_initiator_toolset() -> FunctionToolset:
    """Create toolset for the player who initiated the negotiation."""
    toolset = FunctionToolset()
    
    # NOTE: Disabled — game state is now inlined in the user prompt.
    # @toolset.tool
    # def get_game_and_action_analysis(ctx: RunContext[CatanDependencies]) -> Dict[str, Any]:
    #     """
    #     Get comprehensive game state, available actions, and strategy recommendation.
    #     """
    #     return _get_game_and_action_analysis(ctx)
    # 
    # @toolset.tool
    # def analyze_board(ctx: RunContext[CatanDependencies], focus: str) -> Dict[str, Any]:
    #     """
    #     Analyze the board for strategic insights.
    #
    #     Args:
    #         focus: 'expansion', 'blocking', 'ports', or 'robber'
    #     """
    #     return _analyze_board(ctx, focus)
    
    @toolset.tool
    def send_message(ctx: RunContext[CatanDependencies], message: str) -> Dict[str, Any]:
        """
        Send a message to other players during negotiation.
        
        Args:
            message: Your message to other players
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
        
        Args:
            offer: Resources you are offering [wood, brick, sheep, wheat, ore]
            ask: Resources you want in return [wood, brick, sheep, wheat, ore]
        """
        return _make_trade_offer(ctx, offer, ask)
    
    return toolset


# Pre-built combined toolset instances
NORMAL_PLAY_TOOLSET = create_normal_play_toolset()
NORMAL_PLAY_WITH_TRADE_TOOLSET = create_normal_play_with_trade_toolset()
NEGOTIATION_PARTICIPANT_TOOLSET = create_negotiation_participant_toolset()
NEGOTIATION_INITIATOR_TOOLSET = create_negotiation_initiator_toolset()


# ============================================================================
# Deprecated/Compatibility Exports
# ============================================================================

# Alias for backward compatibility
NegotiationDependencies = None  # Deprecated: Use CatanDependencies instead


def get_all_tools() -> List[FunctionToolset]:
    """
    Get all available toolsets.
    
    Deprecated: Use specific toolsets directly instead.
    
    Returns:
        List of all toolset instances
    """
    return [
        ANALYSIS_TOOLSET,
        TRADE_TOOLSET,
        CHAT_TOOLSET,
        NORMAL_PLAY_TOOLSET,
        NORMAL_PLAY_WITH_TRADE_TOOLSET,
        NEGOTIATION_PARTICIPANT_TOOLSET,
        NEGOTIATION_INITIATOR_TOOLSET,
    ]


# ============================================================================
# Toolset Selection Helper
# ============================================================================

def get_toolsets_for_game_state(
    game,
    color,
    has_rolled: bool,
    is_my_turn: bool,
    in_negotiation: bool = False,
    is_negotiation_initiator: bool = False,
    negotiation_enabled: bool = True,
) -> List[FunctionToolset]:
    """
    Select appropriate toolsets based on game state.
    
    Args:
        game: The current game instance
        color: The player's color
        has_rolled: Whether the player has rolled this turn
        is_my_turn: Whether it's currently this player's turn
        in_negotiation: Whether currently in a negotiation session
        is_negotiation_initiator: Whether this player initiated the negotiation
        negotiation_enabled: Whether negotiation feature is enabled
    
    Returns:
        List of FunctionToolset instances to use for agent.run()
    """
    if in_negotiation:
        if is_negotiation_initiator:
            return [NEGOTIATION_INITIATOR_TOOLSET]
        else:
            return [NEGOTIATION_PARTICIPANT_TOOLSET]
    
    # Normal gameplay
    if is_my_turn and has_rolled and negotiation_enabled:
        return [NORMAL_PLAY_WITH_TRADE_TOOLSET]
    else:
        return [NORMAL_PLAY_TOOLSET]
