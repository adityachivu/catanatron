"""
LLM Negotiating Player - Combines strategic advisor with LLM reasoning.

This player:
1. Gets strategic rankings from the StrategicAdvisor
2. Renders game state for LLM consumption
3. Uses LLM for decision-making and negotiation
4. Tracks negotiation memory (promises, reputation)

For POC: Uses stateless memory but architecture supports full memory.
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Union
from pydantic import BaseModel, Field

from catanatron.game import Game
from catanatron.models.player import Player, Color
from catanatron.models.enums import Action, ActionType, RESOURCES

from catanatron.players.llm.strategic_advisor import StrategicAdvisor, ActionRanking
from catanatron.players.llm.state_renderer import StateRenderer
from catanatron.players.llm.memory import NegotiationMemory, StatelessMemory
from catanatron.players.llm.providers import LLMConfig, LLMClient, MockLLMClient
from catanatron.players.llm.prompts import get_system_prompt, build_decision_prompt


# =============================================================================
# RESPONSE MODELS (Pydantic models for structured LLM output)
# =============================================================================

class DecisionResponse(BaseModel):
    """Structured response for action decisions."""
    
    reasoning: str = Field(
        description="Brief explanation of the decision process"
    )
    chosen_action_index: int = Field(
        ge=1, le=10,
        description="Which ranked action to take (1-based index)"
    )
    wants_to_negotiate: bool = Field(
        default=False,
        description="Whether to initiate negotiation before the action"
    )
    negotiation_message: Optional[str] = Field(
        default=None,
        description="Message to send if negotiating"
    )
    negotiation_target: Optional[str] = Field(
        default=None,
        description="Specific player color to address (e.g., 'BLUE')"
    )
    promise: Optional[str] = Field(
        default=None,
        description="Explicit commitment being made"
    )


class NegotiationResponse(BaseModel):
    """Structured response for negotiation messages."""
    
    message: str = Field(
        description="Response message to other player(s)"
    )
    intent: str = Field(
        description="One of: ACCEPT, REJECT, COUNTER, QUESTION"
    )
    counter_offer: Optional[Dict[str, List[int]]] = Field(
        default=None,
        description="Counter offer if intent is COUNTER"
    )
    reasoning: str = Field(
        description="Brief explanation of the response"
    )


class TradeDecisionResponse(BaseModel):
    """Structured response for trade accept/reject."""
    
    accept: bool = Field(
        description="Whether to accept the trade"
    )
    reasoning: str = Field(
        description="Why accepting or rejecting"
    )


# =============================================================================
# LLM NEGOTIATING PLAYER
# =============================================================================

@dataclass
class LLMPlayerConfig:
    """Configuration for LLM player."""
    
    llm_config: LLMConfig
    persona: str = "balanced"  # cooperative, competitive, balanced
    top_n_rankings: int = 5
    use_stateless_memory: bool = True  # True for POC
    enable_negotiation: bool = True
    max_negotiation_rounds: int = 3
    fallback_to_random: bool = True  # If LLM fails, pick randomly


class LLMNegotiatingPlayer(Player):
    """
    A player that uses LLM for strategic decisions and negotiation.
    
    This player:
    1. Evaluates actions using the StrategicAdvisor (value function)
    2. Renders game state as natural language
    3. Asks LLM to choose from ranked actions
    4. Supports multi-turn negotiation before trades
    
    For POC, memory is stateless. The architecture supports full memory
    for tracking promises, reputation, and trade history.
    """
    
    def __init__(
        self,
        color: Color,
        config: LLMPlayerConfig,
        is_bot: bool = True
    ):
        """
        Initialize the LLM player.
        
        Args:
            color: Player color
            config: LLM configuration
            is_bot: Whether this is a bot (always True for LLM players)
        """
        super().__init__(color, is_bot)
        self.config = config
        
        # Initialize components
        self.advisor = StrategicAdvisor()
        self.renderer = StateRenderer()
        
        # Memory (stateless for POC)
        if config.use_stateless_memory:
            self.memory = StatelessMemory(color)
        else:
            self.memory = NegotiationMemory(color)
        
        # LLM client (lazy initialization)
        self._llm_client: Optional[LLMClient] = None
        self._system_prompt = get_system_prompt(config.persona)
        
        # Negotiation state
        self._pending_negotiations: Dict[str, Any] = {}
        self._conversation_history: List[str] = []
        
        # Logging
        self.decision_log: List[Dict] = []
    
    def _get_llm_client(self) -> LLMClient:
        """Get or create the LLM client (lazy initialization)."""
        if self._llm_client is None:
            self._llm_client = LLMClient(
                self.config.llm_config,
                DecisionResponse,
                self._system_prompt
            )
        return self._llm_client
    
    def decide(self, game: Game, playable_actions: List[Action]) -> Action:
        """
        Decide which action to take.
        
        This is the main entry point called by the game engine.
        
        Args:
            game: Current game state
            playable_actions: List of legal actions
            
        Returns:
            The chosen action
        """
        # Handle trivial cases
        if len(playable_actions) == 1:
            return playable_actions[0]
        
        # Handle trade decisions specially
        current_prompt = game.state.current_prompt
        if current_prompt and current_prompt.value == "DECIDE_TRADE":
            return self._handle_trade_decision(game, playable_actions)
        
        # Use async decision-making
        try:
            return asyncio.run(self._decide_async(game, playable_actions))
        except Exception as e:
            # Fallback on error
            if self.config.fallback_to_random:
                import random
                return random.choice(playable_actions)
            raise
    
    async def _decide_async(
        self, 
        game: Game, 
        playable_actions: List[Action]
    ) -> Action:
        """
        Async decision-making with LLM.
        
        Args:
            game: Current game state
            playable_actions: List of legal actions
            
        Returns:
            The chosen action
        """
        # Get strategic rankings
        rankings = self.advisor.rank_actions(
            game, 
            playable_actions, 
            top_n=self.config.top_n_rankings
        )
        
        if not rankings:
            # No valid rankings, pick first action
            return playable_actions[0]
        
        # Render context
        memory_context = self.memory.get_context(game.state.num_turns)
        full_context = self.renderer.render_full_context(
            game,
            self.color,
            rankings=rankings,
            memory_context=memory_context,
            conversation_history=self._conversation_history if self._conversation_history else None,
        )
        
        # Build prompt
        prompt = build_decision_prompt(
            game_state_text=full_context,
            rankings_text="",  # Already included in full_context
            memory_text="",    # Already included
        )
        
        # Call LLM
        try:
            client = self._get_llm_client()
            response = await client.run(prompt)
            
            # Log decision
            self.decision_log.append({
                "turn": game.state.num_turns,
                "rankings": [(r.rank, r.action.action_type.value) for r in rankings],
                "chosen_index": response.chosen_action_index,
                "reasoning": response.reasoning,
                "wants_to_negotiate": response.wants_to_negotiate,
            })
            
            # Get chosen action
            chosen_index = min(response.chosen_action_index - 1, len(rankings) - 1)
            chosen_index = max(0, chosen_index)
            chosen_action = rankings[chosen_index].action
            
            # Handle negotiation if requested
            if (
                response.wants_to_negotiate 
                and self.config.enable_negotiation
                and response.negotiation_message
            ):
                # Record the negotiation message
                self.memory.record_message(
                    sender=self.color,
                    content=response.negotiation_message,
                    current_turn=game.state.num_turns,
                )
                
                # Record any promises
                if response.promise:
                    from catanatron.players.llm.memory import PromiseType
                    target = None
                    if response.negotiation_target:
                        try:
                            target = Color(response.negotiation_target)
                        except ValueError:
                            pass
                    
                    if target:
                        self.memory.record_promise(
                            maker=self.color,
                            recipient=target,
                            promise_type=PromiseType.CUSTOM,
                            description=response.promise,
                            current_turn=game.state.num_turns,
                        )
            
            return chosen_action
            
        except Exception as e:
            # On LLM error, fall back to top-ranked action
            if self.config.fallback_to_random:
                return rankings[0].action
            raise
    
    def _handle_trade_decision(
        self, 
        game: Game, 
        playable_actions: List[Action]
    ) -> Action:
        """
        Handle ACCEPT_TRADE / REJECT_TRADE decision.
        
        This uses simpler logic than full decision-making:
        1. Check if we have a pending agreement from negotiation
        2. If so, honor the agreement
        3. Otherwise, evaluate the trade on its merits
        """
        current_trade = game.state.current_trade
        
        # Find accept and reject actions
        accept_action = None
        reject_action = None
        for action in playable_actions:
            if action.action_type == ActionType.ACCEPT_TRADE:
                accept_action = action
            elif action.action_type == ActionType.REJECT_TRADE:
                reject_action = action
        
        # If we can't accept (don't have resources), must reject
        if accept_action is None:
            return reject_action
        
        # For POC with stateless memory, use simple heuristic
        # In full version, would check negotiation memory for agreements
        
        # Simple heuristic: accept if the trade seems balanced
        if self._is_trade_acceptable(game, current_trade):
            return accept_action
        else:
            return reject_action
    
    def _is_trade_acceptable(self, game: Game, trade: tuple) -> bool:
        """
        Simple heuristic to evaluate if a trade is acceptable.
        
        For POC, this uses basic resource counting.
        In full version, would use LLM reasoning.
        """
        if not trade or len(trade) < 10:
            return False
        
        # What we'd give vs what we'd receive
        giving = trade[5:10]  # What's asked of us
        receiving = trade[0:5]  # What's offered to us
        
        giving_count = sum(giving)
        receiving_count = sum(receiving)
        
        # Accept if we receive at least as much as we give
        # This is a very simple heuristic
        return receiving_count >= giving_count
    
    def handle_negotiation_message(
        self,
        game: Game,
        sender: Color,
        message: str,
        proposed_trade: Optional[tuple] = None
    ) -> Optional[str]:
        """
        Handle an incoming negotiation message from another player.
        
        This is called by the NegotiationManager when another player
        sends a message during negotiation.
        
        Args:
            game: Current game state
            sender: Who sent the message
            message: The message content
            proposed_trade: Trade proposal if any
            
        Returns:
            Response message or None
        """
        # Record incoming message
        self.memory.record_message(
            sender=sender,
            content=message,
            current_turn=game.state.num_turns,
            proposed_trade=proposed_trade,
        )
        
        # Add to conversation history
        self._conversation_history.append(f"{sender.value}: {message}")
        
        # For POC, return None (no response)
        # Full version would use LLM to generate response
        return None
    
    def reset_state(self):
        """Reset player state between games."""
        self.memory.clear()
        self._pending_negotiations.clear()
        self._conversation_history.clear()
        self.decision_log.clear()
    
    def __repr__(self) -> str:
        return (
            f"LLMNegotiatingPlayer({self.color.value}, "
            f"model={self.config.llm_config.model})"
        )


# =============================================================================
# MOCK LLM PLAYER (for testing without actual LLM calls)
# =============================================================================

class MockLLMPlayer(LLMNegotiatingPlayer):
    """
    Mock LLM player for testing.
    
    This player uses deterministic logic instead of LLM calls,
    useful for:
    - Unit testing
    - Development without API costs
    - Benchmarking
    """
    
    def __init__(
        self,
        color: Color,
        strategy: str = "top_ranked",  # top_ranked, random, second_best
        is_bot: bool = True
    ):
        """
        Initialize mock player.
        
        Args:
            color: Player color
            strategy: Decision strategy to use
            is_bot: Whether this is a bot
        """
        # Create a dummy config
        config = LLMPlayerConfig(
            llm_config=LLMConfig.openai("mock"),
            use_stateless_memory=True,
        )
        super().__init__(color, config, is_bot)
        self.strategy = strategy
    
    async def _decide_async(
        self, 
        game: Game, 
        playable_actions: List[Action]
    ) -> Action:
        """Mock decision - no LLM call."""
        rankings = self.advisor.rank_actions(
            game, 
            playable_actions, 
            top_n=self.config.top_n_rankings
        )
        
        if not rankings:
            return playable_actions[0]
        
        if self.strategy == "top_ranked":
            return rankings[0].action
        elif self.strategy == "second_best" and len(rankings) > 1:
            return rankings[1].action
        elif self.strategy == "random":
            import random
            return random.choice([r.action for r in rankings])
        else:
            return rankings[0].action
    
    def __repr__(self) -> str:
        return f"MockLLMPlayer({self.color.value}, strategy={self.strategy})"


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_llm_player(
    color: Color,
    provider: str = "gemini",
    model: str = "gemini-2.5-pro",
    persona: str = "balanced",
    **kwargs
) -> LLMNegotiatingPlayer:
    """
    Factory function to create an LLM player.
    
    Args:
        color: Player color
        provider: LLM provider (openai, anthropic, gemini, ollama)
        model: Model name
        persona: Negotiation persona
        **kwargs: Additional config options
        
    Returns:
        Configured LLMNegotiatingPlayer
    """
    if provider == "openai":
        llm_config = LLMConfig.openai(model)
    elif provider == "anthropic":
        llm_config = LLMConfig.anthropic(model)
    elif provider == "gemini":
        llm_config = LLMConfig.gemini(model)
    elif provider == "ollama":
        llm_config = LLMConfig.ollama(model)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    config = LLMPlayerConfig(
        llm_config=llm_config,
        persona=persona,
        **kwargs
    )
    
    return LLMNegotiatingPlayer(color, config)


def create_mock_player(
    color: Color,
    strategy: str = "top_ranked"
) -> MockLLMPlayer:
    """
    Factory function to create a mock LLM player for testing.
    
    Args:
        color: Player color
        strategy: Decision strategy
        
    Returns:
        MockLLMPlayer instance
    """
    return MockLLMPlayer(color, strategy)

