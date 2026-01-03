"""
Strategic Advisor - Ranks actions using existing value functions.

This module provides action rankings to LLM players without modifying
existing player implementations. It wraps the proven value.py heuristics.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from catanatron.game import Game
from catanatron.models.enums import Action, ActionType, RESOURCES
from catanatron.models.player import Color
from catanatron.players.value import get_value_fn, DEFAULT_WEIGHTS
from catanatron.state_functions import (
    player_key,
    player_num_resource_cards,
    get_player_buildings,
)
from catanatron.models.enums import SETTLEMENT, CITY


@dataclass
class ActionRanking:
    """A ranked action with its strategic value and explanation."""
    
    action: Action
    value: float
    normalized_value: float  # 0-1 scale for LLM consumption
    explanation: str
    rank: int


class StrategicAdvisor:
    """
    Standalone component that evaluates actions using existing value functions.
    
    This provides strategic recommendations to LLM players without modifying
    any existing player code. It uses the proven heuristics from value.py.
    
    Trade-offs accepted for POC:
    - O(n) game copies per decision (~500ms for 50 actions)
    - 1-ply lookahead only (no deep search)
    - No opponent modeling (LLM handles social prediction)
    """
    
    def __init__(
        self, 
        value_fn_name: str = "base_fn", 
        params: Optional[dict] = None
    ):
        """
        Initialize the strategic advisor.
        
        Args:
            value_fn_name: Name of value function ("base_fn" or "contender_fn")
            params: Optional custom weights for value function
        """
        self.value_fn_name = value_fn_name
        self.params = params or DEFAULT_WEIGHTS
        self.value_fn = get_value_fn(value_fn_name, self.params)
    
    def rank_actions(
        self, 
        game: Game, 
        playable_actions: List[Action], 
        top_n: int = 5
    ) -> List[ActionRanking]:
        """
        Rank playable actions by their strategic value.
        
        For each action:
        1. Copy the game state
        2. Execute the action on the copy
        3. Evaluate the resulting state with the value function
        4. Generate a human-readable explanation
        
        Args:
            game: Current game state
            playable_actions: List of legal actions
            top_n: Number of top actions to return
            
        Returns:
            List of ActionRanking, sorted by value (highest first)
        """
        if not playable_actions:
            return []
        
        current_color = game.state.current_color()
        rankings: List[Tuple[Action, float, str]] = []
        
        # Evaluate each action
        for action in playable_actions:
            try:
                game_copy = game.copy()
                game_copy.execute(action, validate_action=False)
                value = self.value_fn(game_copy, current_color)
                explanation = self._explain_action(action, game)
                rankings.append((action, value, explanation))
            except Exception:
                # Skip actions that cause errors during evaluation
                continue
        
        if not rankings:
            return []
        
        # Sort by value descending
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        # Normalize values to 0-1 range
        max_value = rankings[0][1]
        min_value = rankings[-1][1]
        value_range = max_value - min_value if max_value != min_value else 1.0
        
        # Create ActionRanking objects
        result = []
        for rank, (action, value, explanation) in enumerate(rankings[:top_n], start=1):
            normalized = (value - min_value) / value_range if value_range > 0 else 0.5
            result.append(ActionRanking(
                action=action,
                value=value,
                normalized_value=normalized,
                explanation=explanation,
                rank=rank
            ))
        
        return result
    
    def _explain_action(self, action: Action, game: Game) -> str:
        """
        Generate a human-readable explanation for an action.
        
        Args:
            action: The action to explain
            game: Current game state (before action)
            
        Returns:
            Brief explanation string
        """
        action_type = action.action_type
        value = action.value
        
        if action_type == ActionType.BUILD_SETTLEMENT:
            resources = self._get_node_resources(game, value)
            return f"Gains production: {resources}"
        
        elif action_type == ActionType.BUILD_CITY:
            resources = self._get_node_resources(game, value)
            return f"Doubles production: {resources}"
        
        elif action_type == ActionType.BUILD_ROAD:
            return "Expands road network"
        
        elif action_type == ActionType.BUY_DEVELOPMENT_CARD:
            return "Buys development card (unknown benefit)"
        
        elif action_type == ActionType.PLAY_KNIGHT_CARD:
            return "Plays knight, moves robber"
        
        elif action_type == ActionType.PLAY_YEAR_OF_PLENTY:
            if value:
                resources = ", ".join(str(r) for r in value)
                return f"Takes {resources} from bank"
            return "Takes resources from bank"
        
        elif action_type == ActionType.PLAY_MONOPOLY:
            return f"Monopolizes all {value}"
        
        elif action_type == ActionType.PLAY_ROAD_BUILDING:
            return "Builds 2 free roads"
        
        elif action_type == ActionType.MARITIME_TRADE:
            return self._explain_maritime_trade(value)
        
        elif action_type == ActionType.OFFER_TRADE:
            return self._explain_domestic_trade(value)
        
        elif action_type == ActionType.ROLL:
            return "Roll dice to start turn"
        
        elif action_type == ActionType.END_TURN:
            return "End turn, no further action"
        
        elif action_type == ActionType.MOVE_ROBBER:
            coord, victim = value if value else (None, None)
            if victim:
                return f"Move robber, steal from {victim.value}"
            return "Move robber (no steal)"
        
        elif action_type == ActionType.DISCARD:
            return "Discard cards (required)"
        
        elif action_type == ActionType.ACCEPT_TRADE:
            return "Accept proposed trade"
        
        elif action_type == ActionType.REJECT_TRADE:
            return "Reject proposed trade"
        
        elif action_type == ActionType.CONFIRM_TRADE:
            return "Confirm trade with accepting player"
        
        elif action_type == ActionType.CANCEL_TRADE:
            return "Cancel trade offer"
        
        return f"{action_type.value}"
    
    def _get_node_resources(self, game: Game, node_id: int) -> str:
        """Get resources produced by a node as a readable string."""
        resources = []
        try:
            adjacent_tiles = game.state.board.map.adjacent_tiles.get(node_id, [])
            for tile in adjacent_tiles:
                if hasattr(tile, 'resource') and tile.resource:
                    resources.append(tile.resource)
        except Exception:
            pass
        
        if not resources:
            return "unknown resources"
        
        # Count occurrences
        from collections import Counter
        counts = Counter(resources)
        parts = [f"{count}x {res}" if count > 1 else res for res, count in counts.items()]
        return ", ".join(parts)
    
    def _explain_maritime_trade(self, trade_value: tuple) -> str:
        """Explain a maritime (port/bank) trade."""
        if not trade_value or len(trade_value) < 5:
            return "Trade with bank/port"
        
        # trade_value is (resource_out, resource_out, ..., resource_in)
        giving = [r for r in trade_value[:-1] if r is not None]
        receiving = trade_value[-1]
        
        if giving:
            give_count = len(giving)
            give_resource = giving[0] if giving else "?"
            return f"Trade {give_count} {give_resource} for 1 {receiving}"
        
        return "Trade with bank/port"
    
    def _explain_domestic_trade(self, trade_value: tuple) -> str:
        """Explain a domestic (player-to-player) trade offer."""
        if not trade_value or len(trade_value) < 10:
            return "Trade with another player"
        
        # First 5 elements: what you're offering (by resource index)
        # Last 5 elements: what you're asking for
        offering = trade_value[:5]
        asking = trade_value[5:10]
        
        offer_parts = []
        ask_parts = []
        
        for i, resource in enumerate(RESOURCES):
            if offering[i] > 0:
                offer_parts.append(f"{offering[i]} {resource}")
            if asking[i] > 0:
                ask_parts.append(f"{asking[i]} {resource}")
        
        offer_str = ", ".join(offer_parts) if offer_parts else "nothing"
        ask_str = ", ".join(ask_parts) if ask_parts else "nothing"
        
        return f"Offer {offer_str} for {ask_str}"
    
    def get_trade_recommendations(
        self, 
        game: Game, 
        color: Color
    ) -> List[Tuple[Color, str]]:
        """
        Get recommendations for who to trade with and why.
        
        Analyzes other players' resources and positions to suggest
        potential trade partners.
        
        Args:
            game: Current game state
            color: The player's color
            
        Returns:
            List of (target_color, reason) tuples
        """
        recommendations = []
        
        for other_color in game.state.colors:
            if other_color == color:
                continue
            
            # Get their resource counts
            other_resources = {}
            for resource in RESOURCES:
                count = player_num_resource_cards(game.state, other_color, resource)
                other_resources[resource] = count
            
            # Find what they have plenty of
            abundant = [r for r, c in other_resources.items() if c >= 3]
            
            if abundant:
                reason = f"Has surplus: {', '.join(abundant)}"
                recommendations.append((other_color, reason))
        
        return recommendations

