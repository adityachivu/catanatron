"""
PydanticAI tools for the Catan LLM agent.

DEPRECATED: This module's register_tools() function is deprecated.
Use the toolsets module instead for dynamic tool selection.

This module now primarily provides helper functions used by toolsets.py:
- _analyze_expansion: Analyze expansion opportunities
- _analyze_blocking: Analyze blocking opportunities
- _analyze_ports: Analyze port access
- _analyze_robber: Analyze robber placement options
- _assess_threat: Assess opponent threat level
"""

import warnings
from typing import Dict, Any
from pydantic_ai import Agent, RunContext

from catanatron.players.llm.base import CatanDependencies

from catanatron.state_functions import get_visible_victory_points
from catanatron.models.enums import CITY
from catanatron.models.map import LandTile
from catanatron.players.llm.state_formatter import StateFormatter


def register_tools(agent: Agent) -> None:
    """
    Register all tools with the agent.
    
    DEPRECATED: This function is deprecated. Tools are now managed via
    toolsets and passed to agent.run_sync() dynamically based on game state.
    
    Use the toolsets module instead:
        from catanatron.players.llm.toolsets import NORMAL_PLAY_TOOLSET
        
        result = agent.run_sync(prompt, deps=deps, tools=NORMAL_PLAY_TOOLSET)

    Args:
        agent: The PydanticAI agent to register tools on
    """
    warnings.warn(
        "register_tools() is deprecated. Use toolsets module and pass tools "
        "to agent.run_sync() instead. See catanatron.players.llm.toolsets.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Keep the old implementation for backwards compatibility
    @agent.tool()
    def get_game_and_action_analysis(ctx: RunContext[CatanDependencies]) -> Dict[str, Any]:
        """
        Combine the analysis of the board into a single analysis.
        """
        full_analysis = {
            "game_state": StateFormatter.format_full_state(ctx.deps.game, ctx.deps.color),
            "available_actions": [StateFormatter.format_action(action, i) for i, action in enumerate(ctx.deps.playable_actions)],
            "strategy_recommendation": ctx.deps.strategy_recommendation if ctx.deps.strategy_recommendation else "No strategy advisor configured",
            "strategy_reasoning": ctx.deps.strategy_reasoning if ctx.deps.strategy_reasoning else "No detailed reasoning available",
        }
        return full_analysis

    @agent.tool
    def analyze_board(
        ctx: RunContext[CatanDependencies], focus: str
    ) -> Dict[str, Any]:
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


# ============ Helper Functions ============
# These are used by both the deprecated register_tools() and the new toolsets module


def _assess_threat(state, opponent_color, my_color) -> str:
    """Assess how threatening an opponent is."""
    opp_vps = get_visible_victory_points(state, opponent_color)
    my_vps = get_visible_victory_points(state, my_color)

    if opp_vps >= 8:
        return "CRITICAL - Close to winning!"
    elif opp_vps >= 6:
        return "HIGH - Strong position"
    elif opp_vps > my_vps:
        return "MEDIUM - Ahead of you"
    else:
        return "LOW - Behind or equal"


def _analyze_expansion(game, my_color) -> Dict[str, Any]:
    """Analyze expansion opportunities."""
    board = game.state.board

    buildable_nodes = list(board.buildable_node_ids(my_color))
    buildable_edges = [list(e) for e in board.buildable_edges(my_color)]

    # Assess each buildable node
    node_assessments = []
    for node_id in buildable_nodes[:5]:  # Limit to top 5
        # Get production at this node
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

        # Find nodes that would block this opponent
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
        # Skip non-land tiles (Water, Port) and desert tiles
        if not isinstance(tile, LandTile) or tile.resource is None:
            continue
        if coord == board.robber_coordinate:
            continue

        # Who would be affected?
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

    # Sort by impact
    placements.sort(key=lambda x: x["total_impact"], reverse=True)

    return {
        "best_robber_placements": placements[:5],
        "current_robber_location": board.robber_coordinate,
        "advice": "Target the leading player's highest-production tile",
    }
