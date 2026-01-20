"""
LLM-powered player implementations using PydanticAI.

This package provides LLM-based players that can use existing strategy players
(AlphaBetaPlayer, MCTSPlayer, etc.) as advisors while making decisions via LLM.

Model Configuration:
    Players accept flexible model configuration:
    
        from pydantic_ai.models.test import TestModel
        from catanatron.players.llm import ModelConfig
        
        # String shorthand
        player = PydanticAIPlayer(Color.RED, model="openai:gpt-4o")
        
        # Direct TestModel for testing
        player = PydanticAIPlayer(Color.RED, model=TestModel())
        
        # ModelConfig for full control
        config = ModelConfig(model_name="openai:gpt-4o", temperature=0.7)
        player = PydanticAIPlayer(Color.RED, model=config)

Testing:
    Use TestModel or FunctionModel for testing without real LLM calls:
    
        from catanatron.players.llm.testing import (
            create_test_player,
            scripted_response,
            always_action,
        )
        
        # Deterministic test player
        player = create_test_player(Color.RED)
        
        # Scripted responses
        model = scripted_response([ActionByIndex(action_index=0)])
        player = PydanticAIPlayer(Color.RED, model=model)

Negotiation Support:
    The package includes a negotiation framework that allows LLM players to
    communicate with each other before making formal trade offers. To enable:
    
        from catanatron.players.llm import setup_negotiation
        
        game = Game(players)
        manager = setup_negotiation(game, max_rounds=10)
        game.play()

Toolsets:
    Tools are managed via toolsets selected at runtime based on context:
    
    - REASONING_TOOLSET: For main gameplay (analysis + initiate_negotiation)
    - CHAT_INITIATOR_TOOLSET: For negotiation initiator (analysis + send_message + trade_offer)
    - CHAT_PARTICIPANT_TOOLSET: For negotiation participants (analysis + send_message + leave_negotiation)
"""

from catanatron.players.llm.base import (
    BaseLLMPlayer,
    CatanDependencies,
    CHAT_SYSTEM_PROMPT,
)
from catanatron.players.llm.output_types import (
    ActionOutput,
    ActionByIndex,
    ChatResponse,
)
from catanatron.players.llm.state_formatter import StateFormatter
from catanatron.players.llm.history import ConversationHistoryManager

# Model configuration
from catanatron.players.llm.models import (
    ModelConfig,
    ModelInput,
    create_model,
    is_test_model,
    anthropic_model,
    openai_model,
    gemini_model,
    groq_model,
)

# Testing utilities
from catanatron.players.llm.testing import (
    create_test_model,
    create_test_player,
    scripted_response,
    always_action,
    negotiation_script,
    trade_offer_script,
    create_custom_response_model,
)

# Negotiation support
from catanatron.players.llm.negotiation import (
    NegotiationManager,
    NegotiationSession,
    NegotiationMessage,
    setup_negotiation,
)

# Toolsets
from catanatron.players.llm.toolsets import (
    REASONING_TOOLSET,
    CHAT_INITIATOR_TOOLSET,
    CHAT_PARTICIPANT_TOOLSET,
    get_toolset_for_context,
)

__all__ = [
    # Core player classes
    "BaseLLMPlayer",
    "CatanDependencies",
    "CHAT_SYSTEM_PROMPT",
    "ActionOutput",
    "ActionByIndex",
    "ChatResponse",
    "StateFormatter",
    "ConversationHistoryManager",
    
    # Model configuration
    "ModelConfig",
    "ModelInput",
    "create_model",
    "is_test_model",
    "anthropic_model",
    "openai_model",
    "gemini_model",
    "groq_model",
    
    # Testing utilities
    "create_test_model",
    "create_test_player",
    "scripted_response",
    "always_action",
    "negotiation_script",
    "trade_offer_script",
    "create_custom_response_model",
    
    # Negotiation
    "NegotiationManager",
    "NegotiationSession",
    "NegotiationMessage",
    "setup_negotiation",
    
    # Toolsets
    "REASONING_TOOLSET",
    "CHAT_INITIATOR_TOOLSET",
    "CHAT_PARTICIPANT_TOOLSET",
    "get_toolset_for_context",
]
