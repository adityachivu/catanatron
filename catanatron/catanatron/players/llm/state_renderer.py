"""
State Renderer - Converts game state to natural language for LLM context.

This module renders the game state from a player's perspective,
providing the LLM with a clear understanding of:
- Current resources and victory points
- Board position and buildings
- Other players' visible information
- Recent game history
"""

from typing import List, Optional, Dict
from dataclasses import dataclass

from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.models.enums import (
    RESOURCES, 
    SETTLEMENT, 
    CITY, 
    ROAD,
    ActionType,
    ActionRecord,
)
from catanatron.state_functions import (
    player_key,
    player_num_resource_cards,
    player_num_dev_cards,
    get_player_buildings,
    get_longest_road_length,
    get_played_dev_cards,
)
from catanatron.players.llm.strategic_advisor import ActionRanking


@dataclass
class PlayerSummary:
    """Summary of a player's visible state."""
    color: Color
    victory_points: int
    num_resources: int
    num_dev_cards: int
    num_settlements: int
    num_cities: int
    num_roads: int
    longest_road_length: int
    has_longest_road: bool
    has_largest_army: bool
    knights_played: int


class StateRenderer:
    """
    Renders game state as natural language text for LLM consumption.
    
    The renderer provides a comprehensive view of the game from a
    specific player's perspective, including:
    - Their own complete state (resources, buildings, dev cards)
    - Other players' visible state (VP, settlements, cities)
    - Board topology highlights
    - Recent action history
    """
    
    def __init__(self, include_coordinates: bool = False):
        """
        Initialize the state renderer.
        
        Args:
            include_coordinates: Whether to include node/edge IDs
                               (useful for debugging, verbose for LLM)
        """
        self.include_coordinates = include_coordinates
    
    def render_full_context(
        self,
        game: Game,
        perspective_color: Color,
        rankings: Optional[List[ActionRanking]] = None,
        memory_context: Optional[str] = None,
        conversation_history: Optional[List[str]] = None,
    ) -> str:
        """
        Render the complete context for LLM decision-making.
        
        Args:
            game: Current game state
            perspective_color: The player's color
            rankings: Optional strategic rankings to include
            memory_context: Optional memory/reputation context
            conversation_history: Optional negotiation history
            
        Returns:
            Formatted string for LLM prompt
        """
        sections = []
        
        # Game state header
        sections.append(self._render_game_state_header(game, perspective_color))
        
        # Your resources and hand
        sections.append(self._render_your_resources(game, perspective_color))
        
        # Your buildings and position
        sections.append(self._render_your_position(game, perspective_color))
        
        # Other players
        sections.append(self._render_other_players(game, perspective_color))
        
        # Strategic recommendations
        if rankings:
            sections.append(self._render_rankings(rankings))
        
        # Memory context (if provided)
        if memory_context:
            sections.append(self._render_memory_section(memory_context))
        
        # Conversation history (if in negotiation)
        if conversation_history:
            sections.append(self._render_conversation(conversation_history))
        
        # Recent history
        sections.append(self._render_recent_history(game, perspective_color))
        
        return "\n\n".join(sections)
    
    def _render_game_state_header(self, game: Game, color: Color) -> str:
        """Render the game state header with basic info."""
        key = player_key(game.state, color)
        vps = game.state.player_state.get(f"{key}_VICTORY_POINTS", 0)
        actual_vps = game.state.player_state.get(f"{key}_ACTUAL_VICTORY_POINTS", 0)
        
        turn_info = f"Turn {game.state.num_turns}"
        phase = self._get_phase_description(game)
        
        lines = [
            "=" * 70,
            f"GAME STATE (You are {color.value}, {actual_vps} victory points)",
            "=" * 70,
            f"Game Progress: {turn_info}, {phase}",
            f"Target: {game.vps_to_win} VP to win",
        ]
        
        return "\n".join(lines)
    
    def _render_your_resources(self, game: Game, color: Color) -> str:
        """Render the player's resources and development cards."""
        key = player_key(game.state, color)
        
        # Resources
        resource_parts = []
        for resource in RESOURCES:
            count = player_num_resource_cards(game.state, color, resource)
            if count > 0:
                resource_parts.append(f"{count} {resource.lower()}")
        
        resources_str = ", ".join(resource_parts) if resource_parts else "none"
        total_resources = sum(
            player_num_resource_cards(game.state, color, r) for r in RESOURCES
        )
        
        # Development cards
        dev_cards = player_num_dev_cards(game.state, color)
        
        # Knights played
        knights = get_played_dev_cards(game.state, color, "KNIGHT")
        
        lines = [
            "YOUR HAND",
            "-" * 40,
            f"Resources ({total_resources} total): {resources_str}",
            f"Development cards: {dev_cards}",
            f"Knights played: {knights}",
        ]
        
        return "\n".join(lines)
    
    def _render_your_position(self, game: Game, color: Color) -> str:
        """Render the player's board position."""
        key = player_key(game.state, color)
        
        # Buildings
        settlements = get_player_buildings(game.state, color, SETTLEMENT)
        cities = get_player_buildings(game.state, color, CITY)
        
        # Road info
        road_length = get_longest_road_length(game.state, color)
        has_road = game.state.player_state.get(f"{key}_HAS_ROAD", False)
        has_army = game.state.player_state.get(f"{key}_HAS_ARMY", False)
        
        # Available pieces
        settlements_left = game.state.player_state.get(f"{key}_SETTLEMENTS_AVAILABLE", 0)
        cities_left = game.state.player_state.get(f"{key}_CITIES_AVAILABLE", 0)
        roads_left = game.state.player_state.get(f"{key}_ROADS_AVAILABLE", 0)
        
        lines = [
            "YOUR POSITION",
            "-" * 40,
            f"Settlements: {len(settlements)} placed, {settlements_left} available",
            f"Cities: {len(cities)} placed, {cities_left} available",
            f"Roads: {15 - roads_left} placed, {roads_left} available",
            f"Longest road length: {road_length}" + (" (LONGEST ROAD)" if has_road else ""),
        ]
        
        if has_army:
            lines.append("You have LARGEST ARMY")
        
        # Port access
        ports = self._get_player_ports(game, color)
        if ports:
            lines.append(f"Port access: {', '.join(ports)}")
        
        return "\n".join(lines)
    
    def _render_other_players(self, game: Game, perspective_color: Color) -> str:
        """Render information about other players."""
        lines = [
            "OTHER PLAYERS",
            "-" * 40,
        ]
        
        for color in game.state.colors:
            if color == perspective_color:
                continue
            
            summary = self._get_player_summary(game, color)
            
            status_parts = []
            if summary.has_longest_road:
                status_parts.append("LONGEST ROAD")
            if summary.has_largest_army:
                status_parts.append("LARGEST ARMY")
            
            status_str = f" [{', '.join(status_parts)}]" if status_parts else ""
            
            lines.append(
                f"  {color.value} ({summary.victory_points} VP){status_str}:"
            )
            lines.append(
                f"    Resources: {summary.num_resources} cards, "
                f"Dev cards: {summary.num_dev_cards}"
            )
            lines.append(
                f"    Buildings: {summary.num_settlements} settlements, "
                f"{summary.num_cities} cities, "
                f"road length {summary.longest_road_length}"
            )
        
        return "\n".join(lines)
    
    def _render_rankings(self, rankings: List[ActionRanking]) -> str:
        """Render strategic action rankings as a table."""
        lines = [
            "=" * 70,
            "STRATEGIC RECOMMENDATIONS (from game analysis)",
            "=" * 70,
        ]
        
        if not rankings:
            lines.append("No actions available")
            return "\n".join(lines)
        
        # Table header
        lines.append(
            f"{'#':<3} {'Action':<40} {'Value':<8} {'Explanation'}"
        )
        lines.append("-" * 70)
        
        for ranking in rankings:
            action_str = self._format_action_short(ranking.action)
            value_str = f"{ranking.normalized_value:.2f}"
            lines.append(
                f"{ranking.rank:<3} {action_str:<40} {value_str:<8} {ranking.explanation}"
            )
        
        return "\n".join(lines)
    
    def _render_memory_section(self, memory_context: str) -> str:
        """Render the memory/negotiation context section."""
        lines = [
            "=" * 70,
            "NEGOTIATION CONTEXT (from memory)",
            "=" * 70,
            memory_context if memory_context else "(No prior negotiations recorded)",
        ]
        return "\n".join(lines)
    
    def _render_conversation(self, history: List[str]) -> str:
        """Render conversation history."""
        lines = [
            "=" * 70,
            "CURRENT NEGOTIATION",
            "=" * 70,
        ]
        
        if history:
            for message in history[-10:]:  # Last 10 messages
                lines.append(message)
        else:
            lines.append("(No messages yet)")
        
        return "\n".join(lines)
    
    def _render_recent_history(
        self, 
        game: Game, 
        perspective_color: Color,
        num_actions: int = 10
    ) -> str:
        """Render recent game actions."""
        lines = [
            "RECENT ACTIONS",
            "-" * 40,
        ]
        
        records = game.state.action_records[-num_actions:] if game.state.action_records else []
        
        if not records:
            lines.append("(Game just started)")
            return "\n".join(lines)
        
        for record in records:
            action = record.action
            action_str = self._format_action_short(action)
            lines.append(f"  {action.color.value}: {action_str}")
        
        return "\n".join(lines)
    
    def _get_phase_description(self, game: Game) -> str:
        """Get a human-readable description of the current game phase."""
        state = game.state
        
        if state.is_initial_build_phase:
            return "Initial placement phase"
        if state.is_discarding:
            return "Discarding phase (7 rolled)"
        if state.is_moving_knight:
            return "Moving robber"
        if state.is_road_building:
            return "Road building (free roads)"
        if state.is_resolving_trade:
            return "Resolving trade offer"
        
        prompt = state.current_prompt
        if prompt:
            return prompt.value.replace("_", " ").title()
        
        return "Main phase"
    
    def _get_player_summary(self, game: Game, color: Color) -> PlayerSummary:
        """Get a summary of a player's visible state."""
        key = player_key(game.state, color)
        
        settlements = get_player_buildings(game.state, color, SETTLEMENT)
        cities = get_player_buildings(game.state, color, CITY)
        
        return PlayerSummary(
            color=color,
            victory_points=game.state.player_state.get(f"{key}_VICTORY_POINTS", 0),
            num_resources=sum(
                player_num_resource_cards(game.state, color, r) for r in RESOURCES
            ),
            num_dev_cards=player_num_dev_cards(game.state, color),
            num_settlements=len(settlements),
            num_cities=len(cities),
            num_roads=15 - game.state.player_state.get(f"{key}_ROADS_AVAILABLE", 15),
            longest_road_length=get_longest_road_length(game.state, color),
            has_longest_road=game.state.player_state.get(f"{key}_HAS_ROAD", False),
            has_largest_army=game.state.player_state.get(f"{key}_HAS_ARMY", False),
            knights_played=get_played_dev_cards(game.state, color, "KNIGHT"),
        )
    
    def _get_player_ports(self, game: Game, color: Color) -> List[str]:
        """Get list of ports the player has access to."""
        try:
            port_resources = game.state.board.get_player_port_resources(color)
            ports = []
            for resource in port_resources:
                if resource is None:
                    ports.append("3:1 (any)")
                else:
                    ports.append(f"2:1 ({resource})")
            return ports
        except Exception:
            return []
    
    def _format_action_short(self, action) -> str:
        """Format an action as a short string."""
        action_type = action.action_type
        value = action.value
        
        type_name = action_type.value.replace("_", " ").title()
        
        if action_type == ActionType.BUILD_SETTLEMENT:
            return f"Build Settlement" + (f" at {value}" if self.include_coordinates else "")
        
        elif action_type == ActionType.BUILD_CITY:
            return f"Build City" + (f" at {value}" if self.include_coordinates else "")
        
        elif action_type == ActionType.BUILD_ROAD:
            return f"Build Road" + (f" at {value}" if self.include_coordinates else "")
        
        elif action_type == ActionType.OFFER_TRADE:
            if value and len(value) >= 10:
                offering = value[:5]
                asking = value[5:10]
                offer_parts = []
                ask_parts = []
                for i, resource in enumerate(RESOURCES):
                    if offering[i] > 0:
                        offer_parts.append(f"{offering[i]} {resource.lower()}")
                    if asking[i] > 0:
                        ask_parts.append(f"{asking[i]} {resource.lower()}")
                offer_str = "+".join(offer_parts) or "nothing"
                ask_str = "+".join(ask_parts) or "nothing"
                return f"Trade: {offer_str} for {ask_str}"
            return "Offer Trade"
        
        elif action_type == ActionType.MARITIME_TRADE:
            if value:
                giving = [r for r in value[:-1] if r is not None]
                receiving = value[-1]
                if giving:
                    return f"Port: {len(giving)} {giving[0].lower()} -> 1 {receiving.lower()}"
            return "Maritime Trade"
        
        elif action_type == ActionType.MOVE_ROBBER:
            if value and len(value) >= 2:
                coord, victim = value
                if victim:
                    return f"Move Robber, steal from {victim.value}"
            return "Move Robber"
        
        elif action_type in (ActionType.ROLL, ActionType.END_TURN, 
                             ActionType.DISCARD, ActionType.ACCEPT_TRADE,
                             ActionType.REJECT_TRADE, ActionType.CANCEL_TRADE):
            return type_name
        
        elif action_type == ActionType.PLAY_MONOPOLY:
            return f"Play Monopoly ({value})" if value else "Play Monopoly"
        
        elif action_type == ActionType.PLAY_YEAR_OF_PLENTY:
            if value:
                resources = ", ".join(str(r).lower() for r in value)
                return f"Year of Plenty ({resources})"
            return "Play Year of Plenty"
        
        return type_name

