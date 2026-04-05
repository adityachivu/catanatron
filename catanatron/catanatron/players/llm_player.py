"""
Concrete LLM-powered player implementations.

Provides ready-to-use player classes that combine LLM decision making
with various strategy advisors (AlphaBeta, MCTS, Value Function).

Strategy advisors are composed via the strategy_advisor parameter —
no multiple inheritance is used.

Model Configuration:
    All player classes accept flexible model configuration:

        from pydantic_ai.models.test import TestModel
        from catanatron.players.llm import ModelConfig

        # String shorthand
        player = PydanticAIPlayer(Color.RED, model="openai:gpt-4o")

        # Direct TestModel for testing
        player = PydanticAIPlayer(Color.RED, model=TestModel())

        # ModelConfig for full control
        config = ModelConfig(model_name="openai:gpt-4o", temperature=0.7)
        player = PydanticAIPlayer(Color.RED, model=config)
"""

from typing import Literal, Optional

from catanatron.models.player import Color

from catanatron.players.llm.base import BaseLLMPlayer
from catanatron.players.llm.models import ModelInput
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.mcts import MCTSPlayer
from catanatron.players.value import ValueFunctionPlayer


class PydanticAIPlayer(BaseLLMPlayer):
    """
    Pure LLM player without a strategy advisor.

    Makes decisions entirely based on LLM reasoning using the available tools.

    Example:
        player = PydanticAIPlayer(Color.RED, model="anthropic:claude-sonnet-4-20250514")
    """

    def __init__(
        self,
        color: Color,
        model: ModelInput = None,
        output_mode: Literal["index", "structured"] = "index",
        is_bot: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        super().__init__(
            color,
            model=model,
            output_mode=output_mode,
            is_bot=is_bot,
            temperature=temperature,
            max_tokens=max_tokens,
        )


class LLMAlphaBetaPlayer(BaseLLMPlayer):
    """
    LLM player with AlphaBeta search as strategy advisor.

    The AlphaBeta algorithm looks ahead multiple moves and provides
    a recommendation, which the LLM can follow or override.

    Example:
        player = LLMAlphaBetaPlayer(
            Color.RED,
            model="anthropic:claude-sonnet-4-20250514",
            depth=2,
            prunning=True,
        )
    """

    def __init__(
        self,
        color: Color,
        model: ModelInput = None,
        depth: int = 2,
        prunning: bool = False,
        timeout: Optional[float] = 120.0,
        tool_calls_limit: int = 10,
        output_mode: Literal["index", "structured"] = "index",
        is_bot: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        advisor = AlphaBetaPlayer(color, depth=depth, prunning=prunning)
        super().__init__(
            color,
            model=model,
            strategy_advisor=advisor,
            output_mode=output_mode,
            is_bot=is_bot,
            timeout=timeout,
            tool_calls_limit=tool_calls_limit,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def __repr__(self) -> str:
        a = self.strategy_advisor
        return (
            f"LLMAlphaBetaPlayer:{self.color.value}"
            f"(depth={a.depth},prunning={a.prunning})"
            f"[{self.model}]"
        )


class LLMMCTSPlayer(BaseLLMPlayer):
    """
    LLM player with Monte Carlo Tree Search as strategy advisor.

    MCTS runs simulations to estimate the value of each action,
    providing a probabilistic recommendation to the LLM.

    Example:
        player = LLMMCTSPlayer(
            Color.RED,
            model="anthropic:claude-sonnet-4-20250514",
            num_simulations=20,
        )
    """

    def __init__(
        self,
        color: Color,
        model: ModelInput = None,
        num_simulations: int = 10,
        prunning: bool = False,
        timeout: Optional[float] = 120.0,
        tool_calls_limit: int = 10,
        output_mode: Literal["index", "structured"] = "index",
        is_bot: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        advisor = MCTSPlayer(color, num_simulations=num_simulations, prunning=prunning)
        super().__init__(
            color,
            model=model,
            strategy_advisor=advisor,
            output_mode=output_mode,
            is_bot=is_bot,
            timeout=timeout,
            tool_calls_limit=tool_calls_limit,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def __repr__(self) -> str:
        a = self.strategy_advisor
        return (
            f"LLMMCTSPlayer:{self.color.value}"
            f"({a.num_simulations}:{a.prunning})"
            f"[{self.model}]"
        )


class LLMValuePlayer(BaseLLMPlayer):
    """
    LLM player with heuristic value function as strategy advisor.

    The value function provides a fast, greedy recommendation based on
    hand-crafted heuristics. Less computationally expensive than
    AlphaBeta or MCTS.

    Example:
        player = LLMValuePlayer(Color.RED, model="anthropic:claude-sonnet-4-20250514")
    """

    def __init__(
        self,
        color: Color,
        model: ModelInput = None,
        value_fn_builder_name: Optional[str] = None,
        output_mode: Literal["index", "structured"] = "index",
        timeout: Optional[float] = 120.0,
        tool_calls_limit: int = 10,
        is_bot: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        advisor = ValueFunctionPlayer(color, value_fn_builder_name=value_fn_builder_name)
        super().__init__(
            color,
            model=model,
            strategy_advisor=advisor,
            output_mode=output_mode,
            is_bot=is_bot,
            timeout=timeout,
            tool_calls_limit=tool_calls_limit,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def __repr__(self) -> str:
        a = self.strategy_advisor
        return (
            f"LLMValuePlayer:{self.color.value}"
            f"(value_fn={a.value_fn_builder_name})"
            f"[{self.model}]"
        )


# Aliases for convenience
LLMPlayer = PydanticAIPlayer
