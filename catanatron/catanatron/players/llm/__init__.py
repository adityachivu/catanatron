"""
LLM-powered players for Catanatron with natural language negotiation.

This module provides:
- LLMNegotiatingPlayer: A player that uses LLMs for strategic decisions and negotiation
- StrategicAdvisor: Ranks actions using existing value functions
- StateRenderer: Converts game state to natural language
- NegotiationMemory: Tracks promises, reputation, and trade history

Usage:
    from catanatron.players.llm import create_llm_player, create_mock_player
    from catanatron.models.player import Color
    
    # For real LLM usage (requires pydantic-ai and API keys)
    player = create_llm_player(Color.RED, provider="openai", model="gpt-4")
    
    # For testing without LLM
    player = create_mock_player(Color.RED, strategy="top_ranked")
"""

# Lazy imports to avoid requiring pydantic-ai at module load time
def __getattr__(name):
    """Lazy import to avoid requiring all dependencies at import time."""
    if name == "StrategicAdvisor":
        from catanatron.players.llm.strategic_advisor import StrategicAdvisor
        return StrategicAdvisor
    elif name == "ActionRanking":
        from catanatron.players.llm.strategic_advisor import ActionRanking
        return ActionRanking
    elif name == "StateRenderer":
        from catanatron.players.llm.state_renderer import StateRenderer
        return StateRenderer
    elif name == "NegotiationMemory":
        from catanatron.players.llm.memory import NegotiationMemory
        return NegotiationMemory
    elif name == "StatelessMemory":
        from catanatron.players.llm.memory import StatelessMemory
        return StatelessMemory
    elif name == "LLMNegotiatingPlayer":
        from catanatron.players.llm.player import LLMNegotiatingPlayer
        return LLMNegotiatingPlayer
    elif name == "MockLLMPlayer":
        from catanatron.players.llm.player import MockLLMPlayer
        return MockLLMPlayer
    elif name == "LLMPlayerConfig":
        from catanatron.players.llm.player import LLMPlayerConfig
        return LLMPlayerConfig
    elif name == "create_llm_player":
        from catanatron.players.llm.player import create_llm_player
        return create_llm_player
    elif name == "create_mock_player":
        from catanatron.players.llm.player import create_mock_player
        return create_mock_player
    elif name == "LLMConfig":
        from catanatron.players.llm.providers import LLMConfig
        return LLMConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Player classes
    "LLMNegotiatingPlayer",
    "MockLLMPlayer",
    "LLMPlayerConfig",
    
    # Factory functions
    "create_llm_player",
    "create_mock_player",
    
    # Components
    "StrategicAdvisor",
    "ActionRanking",
    "StateRenderer",
    "NegotiationMemory",
    "StatelessMemory",
    "LLMConfig",
]

