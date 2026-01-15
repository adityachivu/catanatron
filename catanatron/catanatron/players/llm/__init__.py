"""
LLM-powered player implementations using PydanticAI.

This package provides LLM-based players that can use existing strategy players
(AlphaBetaPlayer, MCTSPlayer, etc.) as advisors while making decisions via LLM.

Negotiation Support:
    The package includes a negotiation framework that allows LLM players to
    communicate with each other before making formal trade offers. To enable:
    
        from catanatron.players.llm import setup_negotiation
        
        game = Game(players)
        manager = setup_negotiation(game, max_rounds=10)
        game.play()

Toolsets:
    Tools are now managed via toolsets that can be dynamically selected at
    runtime based on game state. Available toolsets:
    
    - NORMAL_PLAY_TOOLSET: Analysis tools for normal gameplay
    - NORMAL_PLAY_WITH_TRADE_TOOLSET: Analysis + trade tools
    - NEGOTIATION_PARTICIPANT_TOOLSET: Tools for negotiation participants
    - NEGOTIATION_INITIATOR_TOOLSET: Tools for negotiation initiators
"""

from catanatron.players.llm.base import BaseLLMPlayer, CatanDependencies
from catanatron.players.llm.output_types import ActionOutput, ActionByIndex
from catanatron.players.llm.state_formatter import StateFormatter
from catanatron.players.llm.history import ConversationHistoryManager

# Negotiation support
from catanatron.players.llm.negotiation import (
    NegotiationManager,
    NegotiationSession,
    NegotiationMessage,
    setup_negotiation,
)

# Toolsets for external use/testing
from catanatron.players.llm.toolsets import (
    ANALYSIS_TOOLSET,
    TRADE_TOOLSET,
    CHAT_TOOLSET,
    NORMAL_PLAY_TOOLSET,
    NORMAL_PLAY_WITH_TRADE_TOOLSET,
    NEGOTIATION_PARTICIPANT_TOOLSET,
    NEGOTIATION_INITIATOR_TOOLSET,
    NegotiationDependencies,
    get_all_tools,
)

__all__ = [
    # Core player classes
    "BaseLLMPlayer",
    "CatanDependencies",
    "ActionOutput",
    "ActionByIndex",
    "StateFormatter",
    "ConversationHistoryManager",
    
    # Negotiation
    "NegotiationManager",
    "NegotiationSession",
    "NegotiationMessage",
    "NegotiationDependencies",
    "setup_negotiation",
    
    # Toolsets
    "ANALYSIS_TOOLSET",
    "TRADE_TOOLSET",
    "CHAT_TOOLSET",
    "NORMAL_PLAY_TOOLSET",
    "NORMAL_PLAY_WITH_TRADE_TOOLSET",
    "NEGOTIATION_PARTICIPANT_TOOLSET",
    "NEGOTIATION_INITIATOR_TOOLSET",
    "get_all_tools",
]
