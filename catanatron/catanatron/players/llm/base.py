"""
Base class for LLM-powered Catan players using PydanticAI.

This module provides:
- CatanDependencies: Dataclass for dependency injection into the agent
- BaseLLMPlayer: Abstract base class for LLM players with strategy advisor support

The toolset-based architecture allows dynamic tool selection at runtime
based on game state (e.g., trade tools only available after rolling).
"""

from dataclasses import dataclass
from typing import Optional, List, Literal, Any, TYPE_CHECKING

from pydantic_ai import Agent, UsageLimits, ModelSettings, Tool

from catanatron.game import Game
from catanatron.models.player import Player, Color
from catanatron.models.enums import Action, ActionPrompt

from catanatron.players.llm.output_types import ActionByIndex
from catanatron.players.llm.history import ConversationHistoryManager
from catanatron.state_functions import player_has_rolled

import logfire

if TYPE_CHECKING:
    from catanatron.players.llm.negotiation import NegotiationManager, NegotiationMessage


@dataclass
class CatanDependencies:
    """
    Dependencies injected into the PydanticAI agent via RunContext.

    This provides the agent with all necessary game state and context
    for making decisions.
    """

    color: Color
    game: Game
    playable_actions: List[Action]
    strategy_recommendation: Optional[Action]
    strategy_reasoning: Optional[str]
    turn_number: int
    is_my_turn: bool
    
    # Negotiation support
    negotiation_manager: Optional["NegotiationManager"] = None
    player_instance: Optional[Any] = None  # Reference to the player for storing trade actions


# System prompt for the Catan agent
CATAN_SYSTEM_PROMPT = """You are an expert Settlers of Catan player. Your goal is to reach 10 victory points before your opponents.

## Game Rules Summary
- Victory points come from: settlements (1 VP), cities (2 VP), longest road (2 VP), largest army (2 VP), victory point cards (1 VP each)
- Resources: Wood, Brick, Sheep, Wheat, Ore
- Building costs:
  - Road: 1 Wood + 1 Brick
  - Settlement: 1 Wood + 1 Brick + 1 Sheep + 1 Wheat
  - City: 2 Wheat + 3 Ore
  - Development Card: 1 Sheep + 1 Wheat + 1 Ore

## Strategy Tips
- Diversify resource production by building on different numbers (6 and 8 are best)
- Secure important intersection spots early in the game
- Build towards valuable port locations for better trading rates
- Consider blocking opponents' expansion paths
- Time development card plays strategically (knights before rolling if robber is on you)
- Balance between expansion and resource accumulation

## Decision Making
Every turn use the `get_game_and_action_analysis` tool IMMEDIATELY before doing any thinking or reasoning. Do this exactly once per turn to get a comprehensive analysis of the game state and available actions.
Use the available tools to analyze the game state before making decisions.
Consider both immediate gains and long-term strategy.
If a strategy advisor recommendation is provided, consider it but you may choose differently based on your analysis.

Always return a valid action from the available options."""


class BaseLLMPlayer(Player):
    """
    Base class for LLM-powered players.

    This class can be combined with strategy players via multiple inheritance
    to use their decide() method as a recommendation source.

    Example:
        class LLMAlphaBetaPlayer(BaseLLMPlayer, AlphaBetaPlayer):
            pass

    The LLM will receive the AlphaBetaPlayer's recommendation as context
    but can choose to follow it or make a different decision.
    
    Negotiation Support:
        Players can participate in pre-trade negotiations with other LLM players.
        Use setup_negotiation(game) before game.play() to enable this feature.
    """
    
    # Class-level type hints for negotiation support
    negotiation_manager: Optional["NegotiationManager"]
    negotiation_history: List["NegotiationMessage"]
    _pending_trade_action: Optional[Action]
    _last_negotiation_turn: int

    def __init__(
        self,
        color: Color,
        model: str = "anthropic:claude-sonnet-4-20250514",
        output_mode: Literal["index", "structured"] = "index",
        timeout: Optional[float] = 120.0,
        tool_calls_limit: int = 10,
        is_bot: bool = True,
        **strategy_kwargs,
    ):
        """
        Initialize the LLM player.

        Args:
            color: Player color
            model: PydanticAI model string (e.g., "anthropic:claude-sonnet-4-20250514", "openai:gpt-4o")
            output_mode: "index" for fast mode, "structured" for detailed action types
            timeout: Timeout in seconds for LLM calls (default: 120.0)
            tool_calls_limit: Overall tool call limit per decision (default: 10)
            is_bot: Whether this is a bot player (default: True)
            **strategy_kwargs: Arguments passed to parent strategy player (e.g., depth for AlphaBeta)
        """
        # Initialize parent player(s)
        # Note: is_bot is handled explicitly here to avoid passing it through
        # strategy_kwargs to strategy players that don't accept it
        super().__init__(color, **strategy_kwargs)
        
        # Set is_bot after parent chain init (Player sets it, strategy players may not)
        self.is_bot = is_bot

        self.model = model
        self.output_mode = output_mode
        self.timeout = timeout
        self.tool_calls_limit = tool_calls_limit
        self.history_manager = ConversationHistoryManager()
        
        # Negotiation support
        self.negotiation_manager: Optional["NegotiationManager"] = None
        self.negotiation_history: List["NegotiationMessage"] = []
        self._pending_trade_action: Optional[Action] = None
        self._last_negotiation_turn: int = -1

        # Create the agent - defer to allow subclasses to customize
        self.agent = self._create_agent()

    def _create_agent(self) -> Agent:
        """
        Create the PydanticAI agent.
        
        Note: Tools are no longer registered here. Instead, they are passed
        dynamically to agent.run_sync() based on game state via _select_tools().
        """
        # Use index-based output for simplicity and reliability
        agent = Agent(
            self.model,
            deps_type=CatanDependencies,
            output_type=ActionByIndex,
            system_prompt=CATAN_SYSTEM_PROMPT,
            tool_timeout=self.timeout,
        )

        return agent
    
    def _select_tools(self, game: Game) -> List[Tool]:
        """
        Select appropriate tools based on current game state.
        
        This method determines which tools should be available to the agent
        for the current decision. Trade tools are only available when:
        - It's the PLAY_TURN phase
        - The player has already rolled
        - They haven't already initiated a negotiation this turn
        
        Args:
            game: Current game instance
            
        Returns:
            List of Tool instances to pass to agent.run_sync()
        """
        from catanatron.players.llm.toolsets import (
            NORMAL_PLAY_TOOLSET,
            NORMAL_PLAY_WITH_TRADE_TOOLSET,
        )
        
        if self._can_trade(game):
            return NORMAL_PLAY_WITH_TRADE_TOOLSET
        else:
            return NORMAL_PLAY_TOOLSET
    
    def _can_trade(self, game: Game) -> bool:
        """
        Check if trade tools should be available.
        
        Trade tools are available when:
        - Current prompt is PLAY_TURN
        - Player has rolled the dice
        - It's this player's turn
        - Player hasn't already initiated a negotiation this turn
        
        Args:
            game: Current game instance
            
        Returns:
            True if trade tools should be available
        """
        state = game.state
        
        # Must be in PLAY_TURN phase
        if state.current_prompt != ActionPrompt.PLAY_TURN:
            return False
        
        # Must have rolled
        if not player_has_rolled(state, self.color):
            return False
        
        # Must be this player's turn
        if state.current_turn_index != state.color_to_index[self.color]:
            return False
        
        # Check if already initiated negotiation this turn
        if self._has_initiated_negotiation_this_turn(state.num_turns):
            return False
        
        return True
    
    def _has_initiated_negotiation_this_turn(self, turn: int) -> bool:
        """Check if this player has already initiated a negotiation this turn."""
        if self.negotiation_manager is None:
            return False
        
        turn_initiators = self.negotiation_manager.initiated_this_turn.get(turn, set())
        return self.color in turn_initiators
    
    def store_negotiation_history(self, messages: List["NegotiationMessage"]) -> None:
        """
        Store negotiation history after a session ends.
        
        This is called by NegotiationManager when a negotiation concludes.
        The history is then available for DECIDE_TRADE context.
        
        Args:
            messages: List of messages from the negotiation
        """
        self.negotiation_history = messages.copy()
    
    def get_negotiation_context(self) -> str:
        """
        Format negotiation history for inclusion in prompts.
        
        Returns:
            Formatted string of negotiation messages, or empty message if none.
        """
        if not self.negotiation_history:
            return "No prior negotiation."
        
        return "\n".join(
            f"{m.sender.value}: {m.content}" 
            for m in self.negotiation_history
        )
    
    def clear_negotiation_history(self) -> None:
        """Clear the stored negotiation history."""
        self.negotiation_history = []

    def _get_strategy_recommendation(
        self, game: Game, playable_actions: List[Action]
    ) -> tuple[Optional[Action], Optional[str]]:
        """
        Get recommendation from parent strategy player.

        This method traverses the MRO to find a parent class with a decide()
        method (other than BaseLLMPlayer and Player base class).

        Returns:
            Tuple of (recommended_action, reasoning_string)
        """
        for cls in type(self).__mro__:
            # Skip this class and the base Player class
            if cls is BaseLLMPlayer or cls is Player:
                continue

            # Check if this is a Player subclass with its own decide method
            if issubclass(cls, Player) and "decide" in cls.__dict__:
                try:
                    recommendation = cls.decide(self, game, playable_actions)
                    reasoning = self._explain_recommendation(recommendation, game, cls)
                    return recommendation, reasoning
                except Exception as e:
                    # If strategy player fails, continue without recommendation
                    return None, f"Strategy advisor failed: {e}"

        return None, None

    def _explain_recommendation(
        self, recommendation: Action, game: Game, strategy_cls: type
    ) -> str:
        """Generate explanation for why the strategy player chose this action."""
        action_type = recommendation.action_type.value
        value = recommendation.value

        explanation = f"{strategy_cls.__name__} recommends: {action_type}"

        if value is not None:
            if action_type == "BUILD_SETTLEMENT":
                explanation += f" at node {value}"
            elif action_type == "BUILD_ROAD":
                explanation += f" on edge {value}"
            elif action_type == "BUILD_CITY":
                explanation += f" upgrading node {value}"
            elif action_type == "MOVE_ROBBER":
                coord, victim = value
                explanation += f" to {coord}"
                if victim:
                    explanation += f", stealing from {victim.value}"
            elif action_type in ("PLAY_YEAR_OF_PLENTY", "PLAY_MONOPOLY"):
                explanation += f" for {value}"

        return explanation

    def _build_prompt(self, game: Game) -> str:
        """Build the user prompt for the current decision."""
        state = game.state
        prompt_parts = []

        # Current phase
        prompt_parts.append(f"Current phase: {state.current_prompt.value}")
        prompt_parts.append(f"Turn number: {state.num_turns}")

        # Trade context if applicable
        if state.is_resolving_trade:
            offer = state.current_trade[:5]
            ask = state.current_trade[5:10]
            resource_names = ["wood", "brick", "sheep", "wheat", "ore"]
            offer_str = ", ".join(f"{resource_names[i]}: {v}" for i, v in enumerate(offer) if v > 0)
            ask_str = ", ".join(f"{resource_names[i]}: {v}" for i, v in enumerate(ask) if v > 0)
            prompt_parts.append(f"Active trade - Offering: [{offer_str}], Asking: [{ask_str}]")
        
        # Include negotiation history for DECIDE_TRADE
        if state.current_prompt == ActionPrompt.DECIDE_TRADE:
            negotiation_context = self.get_negotiation_context()
            if negotiation_context and negotiation_context != "No prior negotiation.":
                prompt_parts.append("\n=== PRIOR NEGOTIATION ===")
                prompt_parts.append(negotiation_context)
                prompt_parts.append("=========================")
                prompt_parts.append(
                    "\nConsider the negotiation context when deciding whether to accept or reject."
                )

        prompt_parts.append(
            "\nUse the available tools to understand the game state, then choose your action."
        )
        prompt_parts.append(
            "Return the index of your chosen action from the available actions list."
        )

        return "\n".join(prompt_parts)

    def _resolve_action(
        self, output: ActionByIndex, playable_actions: List[Action]
    ) -> Action:
        """
        Resolve the LLM output to a valid Action.

        Args:
            output: The ActionByIndex from the LLM
            playable_actions: List of valid actions

        Returns:
            The selected Action, or first action if index is invalid
        """
        if isinstance(output, ActionByIndex):
            idx = output.action_index
            if 0 <= idx < len(playable_actions):
                return playable_actions[idx]
            # Invalid index, fall back to first action
            return playable_actions[0]

        # For structured output mode (future enhancement)
        # Would need to match against playable_actions
        return playable_actions[0]

    def decide(self, game: Game, playable_actions: List[Action]) -> Action:
        """
        Make a decision using the LLM agent.

        This is the main entry point called by the game engine.
        """
        # 1. Get strategy recommendation from parent (if any)
        recommendation, reasoning = self._get_strategy_recommendation(
            game, playable_actions
        )

        # 2. Check for turn boundary, manage history
        current_turn = game.state.num_turns
        if self.history_manager.is_new_turn(current_turn):
            self.history_manager.clear()
            self.history_manager.set_turn(current_turn)
            # Clear negotiation history on new turn
            self.clear_negotiation_history()

        # 3. Clear any pending trade action from previous runs
        self._pending_trade_action = None

        # 4. Build dependencies with negotiation support
        deps = CatanDependencies(
            color=self.color,
            game=game,
            playable_actions=playable_actions,
            strategy_recommendation=recommendation,
            strategy_reasoning=reasoning,
            turn_number=current_turn,
            is_my_turn=game.state.current_turn_index
            == game.state.color_to_index[self.color],
            negotiation_manager=self.negotiation_manager,
            player_instance=self,
        )

        # 5. Select appropriate tools based on game state
        tools = self._select_tools(game)

        # 6. Run agent with history, tools, and timeout
        try:
            # Configure model settings with timeout for the entire run
            model_settings = None
            if self.timeout is not None and self.timeout > 0:
                model_settings = ModelSettings(timeout=self.timeout)
            
            result = self.agent.run_sync(
                self._build_prompt(game),
                deps=deps,
                message_history=self.history_manager.get_messages(),
                tools=tools,
                usage_limits=UsageLimits(
                    tool_calls_limit=self.tool_calls_limit,
                ),
                model_settings=model_settings,
            )

            # 7. Update history
            self.history_manager.update(result.all_messages())

            # 8. Check if a trade action was set by a tool
            if self._pending_trade_action is not None:
                action = self._pending_trade_action
                self._pending_trade_action = None
                logfire.info(
                    f"LLM player {self.color} made trade offer",
                    turn_number=current_turn,
                    action_type=action.action_type.value
                )
                return action

            # 9. Map output to Action
            action = self._resolve_action(result.output, playable_actions)
            logfire.info(f"LLM player {self.color} chose action {action.action_type}", turn_number=current_turn)
            return action

        except TimeoutError as e:
            # Timeout occurred - fall back to strategy recommendation
            logfire.warning(
                f"LLM player {self.color} timed out after {self.timeout}s",
                turn_number=current_turn,
                error=str(e)
            )
        except Exception as e:
            # On any other error (including UsageLimitExceeded), fall back to strategy recommendation or first action
            logfire.error(f"LLM player {self.color} error: {e}", turn_number=current_turn)
        finally:
            if recommendation is not None:
                return recommendation
            return playable_actions[0]

    def reset_state(self):
        """Reset state between games."""
        super().reset_state()
        self.history_manager.clear()
        self.clear_negotiation_history()
        self._pending_trade_action = None
        self._last_negotiation_turn = -1

    def __getstate__(self):
        """
        Custom pickle serialization.
        
        Exclude unpickleable objects (agent, history_manager, negotiation_manager)
        that contain SSLContext, thread locks, and other non-serializable components.
        """
        state = self.__dict__.copy()
        # Remove unpickleable objects
        state.pop('agent', None)
        state.pop('history_manager', None)
        state.pop('negotiation_manager', None)
        state.pop('negotiation_history', None)
        state.pop('_pending_trade_action', None)
        return state

    def __setstate__(self, state):
        """
        Custom pickle deserialization.
        
        Recreate unpickleable objects after deserialization.
        """
        self.__dict__.update(state)
        # Recreate the agent and history manager
        self.history_manager = ConversationHistoryManager()
        self.agent = self._create_agent()
        # Reset negotiation state
        self.negotiation_manager = None
        self.negotiation_history = []
        self._pending_trade_action = None
        self._last_negotiation_turn = -1
