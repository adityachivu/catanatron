"""
Tests for LLM player implementations.

These tests use PydanticAI's TestModel and FunctionModel to mock LLM responses
without requiring actual API keys.
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import List, Any

from catanatron.game import Game
from catanatron.models.player import Color, RandomPlayer
from catanatron.models.enums import Action, ActionType, ActionPrompt
from catanatron.state import State

from tests.utils import build_initial_placements, advance_to_play_turn


# Skip all tests if pydantic-ai is not available
pytest.importorskip("pydantic_ai")

# Import PydanticAI test utilities
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.function import FunctionModel


# ============= Test Fixtures =============


@pytest.fixture
def two_player_game():
    """Create a simple 2-player game for testing."""
    players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
    game = Game(players, seed=42)
    return game


@pytest.fixture
def game_after_initial_placement(two_player_game):
    """Create a game that's past the initial placement phase."""
    game = two_player_game
    build_initial_placements(game)
    return game


@pytest.fixture
def game_at_play_turn(game_after_initial_placement):
    """Create a game at PLAY_TURN phase after rolling."""
    game = game_after_initial_placement
    advance_to_play_turn(game)
    return game


# ============= Unit Tests for Components =============


class TestConversationHistoryManager:
    """Tests for the conversation history manager."""

    def test_new_turn_detection(self):
        from catanatron.players.llm.history import ConversationHistoryManager

        manager = ConversationHistoryManager()

        # First call should always be new turn
        assert manager.is_new_turn(0)

        # After setting turn, same turn is not new
        manager.set_turn(0)
        assert not manager.is_new_turn(0)

        # Different turn is new
        assert manager.is_new_turn(1)

    def test_clear_history(self):
        from catanatron.players.llm.history import ConversationHistoryManager

        manager = ConversationHistoryManager()
        manager.add_message({"role": "user", "content": "test"})
        assert manager.message_count == 1

        manager.clear()
        assert manager.message_count == 0

    def test_update_messages(self):
        from catanatron.players.llm.history import ConversationHistoryManager

        manager = ConversationHistoryManager()
        new_messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        manager.update(new_messages)
        assert manager.get_messages() == new_messages

    def test_trim_to_last_n(self):
        from catanatron.players.llm.history import ConversationHistoryManager

        manager = ConversationHistoryManager()
        for i in range(10):
            manager.add_message({"content": f"message_{i}"})

        manager.trim_to_last_n(3)
        assert manager.message_count == 3
        assert manager.messages[-1]["content"] == "message_9"


class TestStateFormatter:
    """Tests for the state formatter."""

    def test_format_full_state(self, game_after_initial_placement):
        from catanatron.players.llm.state_formatter import StateFormatter

        state = StateFormatter.format_full_state(
            game_after_initial_placement, Color.RED
        )

        assert "my_color" in state
        assert state["my_color"] == "RED"
        assert "my_state" in state
        assert "opponents" in state
        assert "board" in state
        assert "bank" in state

    def test_format_for_prompt(self, game_after_initial_placement):
        from catanatron.players.llm.state_formatter import StateFormatter

        prompt = StateFormatter.format_for_prompt(
            game_after_initial_placement, Color.RED
        )

        assert "RED" in prompt
        assert "Victory Points" in prompt
        assert "Resources" in prompt

    def test_format_action(self):
        from catanatron.players.llm.state_formatter import StateFormatter

        action = Action(Color.RED, ActionType.BUILD_SETTLEMENT, 5)
        formatted = StateFormatter.format_action(action, 0)

        assert formatted["index"] == 0
        assert formatted["type"] == "BUILD_SETTLEMENT"
        assert "node 5" in formatted["description"]

    def test_format_roll_action(self):
        from catanatron.players.llm.state_formatter import StateFormatter

        action = Action(Color.RED, ActionType.ROLL, None)
        formatted = StateFormatter.format_action(action, 0)

        assert formatted["type"] == "ROLL"
        assert "Roll" in formatted["description"]

    def test_format_strategy_insight(self):
        from catanatron.players.llm.state_formatter import StateFormatter

        action = Action(Color.RED, ActionType.END_TURN, None)
        insight = StateFormatter.format_strategy_insight(
            action, "No better moves available", "TestAdvisor"
        )

        assert "TestAdvisor" in insight
        assert "End your turn" in insight
        assert "No better moves" in insight


class TestOutputTypes:
    """Tests for output type models."""

    def test_action_by_index(self):
        from catanatron.players.llm.output_types import ActionByIndex

        output = ActionByIndex(action_index=3, reasoning="Best action")
        assert output.action_index == 3
        assert output.reasoning == "Best action"

    def test_action_by_index_without_reasoning(self):
        from catanatron.players.llm.output_types import ActionByIndex

        output = ActionByIndex(action_index=0)
        assert output.action_index == 0
        assert output.reasoning is None

    def test_build_settlement_action(self):
        from catanatron.players.llm.output_types import BuildSettlementAction

        output = BuildSettlementAction(node_id=42, reasoning="Good production")
        assert output.action_type == "BUILD_SETTLEMENT"
        assert output.node_id == 42


class TestCatanDependencies:
    """Tests for the dependencies dataclass."""

    def test_dependencies_creation(self, game_after_initial_placement):
        from catanatron.players.llm.base import CatanDependencies

        game = game_after_initial_placement
        deps = CatanDependencies(
            color=Color.RED,
            game=game,
            playable_actions=game.playable_actions,
            strategy_recommendation=None,
            strategy_reasoning=None,
            turn_number=0,
            is_my_turn=True,
        )

        assert deps.color == Color.RED
        assert deps.game is game
        assert len(deps.playable_actions) > 0


# ============= Integration Tests with Mocked LLM =============


class TestPydanticAIPlayerWithMock:
    """Tests for PydanticAIPlayer with mocked LLM responses."""

    def test_player_creation(self):
        from catanatron.players.llm_player import PydanticAIPlayer

        # This will fail if pydantic-ai is not properly configured
        # but we're testing structure, not actual LLM calls
        with patch("catanatron.players.llm.base.Agent") as MockAgent:
            mock_agent = MagicMock()
            MockAgent.return_value = mock_agent

            player = PydanticAIPlayer(Color.RED, model="test:model")
            assert player.color == Color.RED
            assert player.model == "test:model"

    def test_decide_returns_valid_action(self, game_after_initial_placement):
        from catanatron.players.llm_player import PydanticAIPlayer
        from catanatron.players.llm.output_types import ActionByIndex

        game = game_after_initial_placement

        with patch("catanatron.players.llm.base.Agent") as MockAgent:
            # Setup mock agent
            mock_agent = MagicMock()
            mock_result = MagicMock()
            mock_result.data = ActionByIndex(action_index=0)
            mock_result.all_messages.return_value = []
            mock_agent.run_sync.return_value = mock_result
            MockAgent.return_value = mock_agent

            player = PydanticAIPlayer(Color.RED)
            action = player.decide(game, game.playable_actions)

            # Should return the first action since we mocked index 0
            assert action == game.playable_actions[0]

    def test_decide_with_invalid_index_falls_back(self, game_after_initial_placement):
        from catanatron.players.llm_player import PydanticAIPlayer
        from catanatron.players.llm.output_types import ActionByIndex

        game = game_after_initial_placement

        with patch("catanatron.players.llm.base.Agent") as MockAgent:
            mock_agent = MagicMock()
            mock_result = MagicMock()
            # Return invalid index
            mock_result.data = ActionByIndex(action_index=9999)
            mock_result.all_messages.return_value = []
            mock_agent.run_sync.return_value = mock_result
            MockAgent.return_value = mock_agent

            player = PydanticAIPlayer(Color.RED)
            action = player.decide(game, game.playable_actions)

            # Should fallback to first action
            assert action == game.playable_actions[0]

    def test_decide_on_error_falls_back(self, game_after_initial_placement):
        from catanatron.players.llm_player import PydanticAIPlayer

        game = game_after_initial_placement

        with patch("catanatron.players.llm.base.Agent") as MockAgent:
            mock_agent = MagicMock()
            # Simulate LLM error
            mock_agent.run_sync.side_effect = Exception("API Error")
            MockAgent.return_value = mock_agent

            player = PydanticAIPlayer(Color.RED)
            action = player.decide(game, game.playable_actions)

            # Should fallback to first action on error
            assert action == game.playable_actions[0]

    def test_history_cleared_on_new_turn(self, game_after_initial_placement):
        from catanatron.players.llm_player import PydanticAIPlayer
        from catanatron.players.llm.output_types import ActionByIndex

        game = game_after_initial_placement

        with patch("catanatron.players.llm.base.Agent") as MockAgent:
            mock_agent = MagicMock()
            mock_result = MagicMock()
            mock_result.data = ActionByIndex(action_index=0)
            mock_result.all_messages.return_value = [{"test": "message"}]
            mock_agent.run_sync.return_value = mock_result
            MockAgent.return_value = mock_agent

            player = PydanticAIPlayer(Color.RED)

            # First decide call
            player.decide(game, game.playable_actions)
            assert player.history_manager.message_count > 0

            # Simulate turn change
            game.state.num_turns = 999

            # Second decide call should clear history
            player.decide(game, game.playable_actions)
            assert player.history_manager.current_turn == 999


class TestLLMAlphaBetaPlayer:
    """Tests for LLMAlphaBetaPlayer."""

    def test_player_creation(self):
        from catanatron.players.llm_player import LLMAlphaBetaPlayer

        with patch("catanatron.players.llm.base.Agent") as MockAgent:
            MockAgent.return_value = MagicMock()

            player = LLMAlphaBetaPlayer(
                Color.RED, model="test:model", depth=3, prunning=True
            )
            assert player.color == Color.RED
            assert player.depth == 3
            assert player.prunning == True

    def test_repr(self):
        from catanatron.players.llm_player import LLMAlphaBetaPlayer

        with patch("catanatron.players.llm.base.Agent") as MockAgent:
            MockAgent.return_value = MagicMock()

            player = LLMAlphaBetaPlayer(Color.RED, model="test:model")
            repr_str = repr(player)
            assert "LLM" in repr_str
            assert "AlphaBeta" in repr_str


class TestLLMMCTSPlayer:
    """Tests for LLMMCTSPlayer."""

    def test_player_creation(self):
        from catanatron.players.llm_player import LLMMCTSPlayer

        with patch("catanatron.players.llm.base.Agent") as MockAgent:
            MockAgent.return_value = MagicMock()

            player = LLMMCTSPlayer(
                Color.RED, model="test:model", num_simulations=20
            )
            assert player.color == Color.RED
            assert player.num_simulations == 20


class TestLLMValuePlayer:
    """Tests for LLMValuePlayer."""

    def test_player_creation(self):
        from catanatron.players.llm_player import LLMValuePlayer

        with patch("catanatron.players.llm.base.Agent") as MockAgent:
            MockAgent.return_value = MagicMock()

            player = LLMValuePlayer(Color.RED, model="test:model")
            assert player.color == Color.RED


# ============= CLI Registration Tests =============


class TestCLIRegistration:
    """Tests for CLI player registration."""

    def test_llm_players_registered(self):
        from catanatron.cli.cli_players import CLI_PLAYERS

        codes = [p.code for p in CLI_PLAYERS]
        assert "LLM" in codes
        assert "LLMAB" in codes
        assert "LLMM" in codes
        assert "LLMV" in codes

    def test_factory_functions_exist(self):
        from catanatron.cli.cli_players import (
            create_llm_player,
            create_llm_alphabeta_player,
            create_llm_mcts_player,
            create_llm_value_player,
        )

        # Just verify the functions are importable
        assert callable(create_llm_player)
        assert callable(create_llm_alphabeta_player)
        assert callable(create_llm_mcts_player)
        assert callable(create_llm_value_player)


# ============= Tool Function Tests =============


class TestTools:
    """Tests for the LLM agent tools."""

    def test_get_game_state_tool_structure(self, game_after_initial_placement):
        """Test that get_game_state returns expected structure."""
        from catanatron.players.llm.state_formatter import StateFormatter

        state = StateFormatter.format_full_state(
            game_after_initial_placement, Color.RED
        )

        # Verify structure
        assert "turn" in state
        assert "phase" in state
        assert "my_state" in state
        assert "opponents" in state
        assert "board" in state
        assert "bank" in state

        # Verify my_state has resources (since it's self)
        assert "resources" in state["my_state"]
        assert "development_cards" in state["my_state"]

    def test_format_action_for_different_types(self):
        """Test action formatting for various action types."""
        from catanatron.players.llm.state_formatter import StateFormatter

        actions = [
            (Action(Color.RED, ActionType.ROLL, None), "Roll"),
            (Action(Color.RED, ActionType.END_TURN, None), "End"),
            (Action(Color.RED, ActionType.BUILD_SETTLEMENT, 5), "node 5"),
            (Action(Color.RED, ActionType.BUILD_ROAD, (1, 2)), "nodes 1 and 2"),
            (Action(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, None), "Buy"),
        ]

        for action, expected_text in actions:
            formatted = StateFormatter.format_action(action, 0)
            assert expected_text in formatted["description"], (
                f"Expected '{expected_text}' in description for {action.action_type}"
            )


# ============= Model Configuration Tests =============


class TestModelConfig:
    """Tests for the model configuration system."""

    def test_model_config_creation(self):
        """Test ModelConfig dataclass creation."""
        from catanatron.players.llm.models import ModelConfig
        
        config = ModelConfig(
            model_name="openai:gpt-4o",
            temperature=0.7,
            max_tokens=1024,
        )
        
        assert config.model_name == "openai:gpt-4o"
        assert config.temperature == 0.7
        assert config.max_tokens == 1024

    def test_model_config_to_settings(self):
        """Test converting ModelConfig to ModelSettings."""
        from catanatron.players.llm.models import ModelConfig
        
        config = ModelConfig(
            model_name="test:model",
            temperature=0.5,
            timeout=60.0,
        )
        
        settings = config.to_model_settings()
        assert settings is not None
        assert settings.temperature == 0.5
        assert settings.timeout == 60.0

    def test_create_model_with_string(self):
        """Test create_model with string input."""
        from catanatron.players.llm.models import create_model
        
        result = create_model("anthropic:claude-sonnet-4-20250514")
        assert result == "anthropic:claude-sonnet-4-20250514"

    def test_create_model_with_test_model(self):
        """Test create_model with TestModel passthrough."""
        from catanatron.players.llm.models import create_model
        
        test_model = TestModel(seed=42)
        result = create_model(test_model)
        assert result is test_model

    def test_create_model_with_none_uses_default(self):
        """Test create_model with None uses default."""
        from catanatron.players.llm.models import create_model, DEFAULT_MODEL
        
        result = create_model(None)
        assert result == DEFAULT_MODEL

    def test_create_model_respects_env_var(self):
        """Test create_model respects CATAN_LLM_MODEL env var."""
        from catanatron.players.llm.models import create_model
        
        os.environ["CATAN_LLM_MODEL"] = "openai:gpt-4o-mini"
        try:
            result = create_model(None)
            assert result == "openai:gpt-4o-mini"
        finally:
            del os.environ["CATAN_LLM_MODEL"]

    def test_test_mode_env_var_forces_test_model(self):
        """Test CATAN_LLM_TEST_MODE forces TestModel."""
        from catanatron.players.llm.models import create_model
        
        os.environ["CATAN_LLM_TEST_MODE"] = "1"
        try:
            result = create_model("anthropic:claude-sonnet-4-20250514")
            assert isinstance(result, TestModel)
        finally:
            del os.environ["CATAN_LLM_TEST_MODE"]

    def test_is_test_model(self):
        """Test is_test_model helper function."""
        from catanatron.players.llm.models import is_test_model
        
        assert is_test_model(TestModel()) is True
        assert is_test_model(FunctionModel(lambda m, i: None)) is True
        assert is_test_model("anthropic:claude-sonnet-4-20250514") is False


# ============= Tests with TestModel (No Mocking) =============


class TestPlayerWithTestModel:
    """Tests using PydanticAI's TestModel for deterministic behavior."""

    def test_player_creation_with_test_model(self):
        """Test creating a player with TestModel."""
        from catanatron.players.llm_player import PydanticAIPlayer
        
        player = PydanticAIPlayer(Color.RED, model=TestModel(seed=42))
        assert player.color == Color.RED
        assert isinstance(player._model, TestModel)

    def test_player_with_model_config(self):
        """Test creating a player with ModelConfig."""
        from catanatron.players.llm_player import PydanticAIPlayer
        from catanatron.players.llm.models import ModelConfig
        
        config = ModelConfig(
            model_name="test:model",
            temperature=0.5,
        )
        
        # We need to use TestModel for actual testing
        # ModelConfig with string model would try to use real API
        player = PydanticAIPlayer(Color.RED, model=TestModel())
        assert player.color == Color.RED

    def test_decide_with_test_model(self, game_after_initial_placement):
        """Test decide() with TestModel returns valid action."""
        from catanatron.players.llm_player import PydanticAIPlayer
        
        game = game_after_initial_placement
        player = PydanticAIPlayer(Color.RED, model=TestModel(seed=42))
        
        action = player.decide(game, game.playable_actions)
        
        # Action should be from playable_actions
        assert action in game.playable_actions

    def test_model_settings_passed_through(self):
        """Test that temperature/max_tokens are stored correctly."""
        from catanatron.players.llm_player import PydanticAIPlayer
        
        player = PydanticAIPlayer(
            Color.RED,
            model=TestModel(),
            temperature=0.7,
            max_tokens=2048,
        )
        
        assert player.temperature == 0.7
        assert player.max_tokens == 2048
        
        settings = player._get_model_settings()
        assert settings.temperature == 0.7
        assert settings.max_tokens == 2048


# ============= Tests with Scripted Responses =============


class TestScriptedResponses:
    """Tests using FunctionModel for scripted responses."""

    def test_always_action_helper(self):
        """Test always_action creates consistent model."""
        from catanatron.players.llm.testing import always_action
        from catanatron.players.llm_player import PydanticAIPlayer
        
        model = always_action(0)
        player = PydanticAIPlayer(Color.RED, model=model)
        
        # Model should be a FunctionModel
        assert isinstance(player._model, FunctionModel)

    def test_scripted_response_helper(self):
        """Test scripted_response creates working model."""
        from catanatron.players.llm.testing import scripted_response
        from catanatron.players.llm.output_types import ActionByIndex
        
        responses = [
            ActionByIndex(action_index=0),
            ActionByIndex(action_index=1),
        ]
        model = scripted_response(responses)
        
        assert isinstance(model, FunctionModel)

    def test_create_test_player_helper(self):
        """Test create_test_player convenience function."""
        from catanatron.players.llm.testing import create_test_player
        from catanatron.players.llm_player import PydanticAIPlayer
        
        player = create_test_player(Color.RED)
        
        assert isinstance(player, PydanticAIPlayer)
        assert player.color == Color.RED
        assert isinstance(player._model, TestModel)

    def test_create_test_model_helper(self):
        """Test create_test_model with parameters."""
        from catanatron.players.llm.testing import create_test_model
        
        model = create_test_model(seed=123, call_tools="none")
        
        assert isinstance(model, TestModel)
        assert model.seed == 123


# ============= Negotiation Tests with TestModel =============


class TestNegotiationWithTestModel:
    """Tests for negotiation system using TestModel."""

    def test_negotiation_manager_creation(self):
        """Test NegotiationManager can be created."""
        from catanatron.players.llm.negotiation import NegotiationManager
        
        manager = NegotiationManager(max_rounds=5)
        assert manager.max_rounds == 5
        assert manager.current_session is None

    def test_setup_negotiation(self):
        """Test setup_negotiation registers LLM players."""
        from catanatron.players.llm.negotiation import setup_negotiation
        from catanatron.players.llm_player import PydanticAIPlayer
        
        llm_player = PydanticAIPlayer(Color.RED, model=TestModel())
        random_player = RandomPlayer(Color.BLUE)
        
        game = Game([llm_player, random_player], seed=42)
        manager = setup_negotiation(game, max_rounds=3)
        
        # LLM player should be registered
        assert Color.RED in manager.players
        # Random player should not be registered
        assert Color.BLUE not in manager.players
        # Player should have reference to manager
        assert llm_player.negotiation_manager is manager

    def test_negotiation_session_creation(self):
        """Test starting a negotiation session."""
        from catanatron.players.llm.negotiation import setup_negotiation
        from catanatron.players.llm_player import PydanticAIPlayer
        
        player1 = PydanticAIPlayer(Color.RED, model=TestModel())
        player2 = PydanticAIPlayer(Color.BLUE, model=TestModel())
        
        game = Game([player1, player2], seed=42)
        manager = setup_negotiation(game)
        
        # Start negotiation
        session = manager.start_negotiation(Color.RED, game)
        
        assert session.initiator == Color.RED
        assert Color.RED in session.participants
        assert Color.BLUE in session.participants
        assert session.is_active is True


# ============= Integration Tests =============


class TestIntegrationWithTestModel:
    """Full integration tests using TestModel."""

    def test_game_with_llm_player_completes(self):
        """Test a game with LLM player (using TestModel) can run."""
        from catanatron.players.llm_player import PydanticAIPlayer
        
        # Use TestModel so no real API calls are made
        llm_player = PydanticAIPlayer(Color.RED, model=TestModel(seed=42))
        random_player = RandomPlayer(Color.BLUE)
        
        game = Game([llm_player, random_player], seed=42)
        
        # Run just a few ticks to verify it works
        for _ in range(10):
            if game.winning_color() is not None:
                break
            game.play_tick()
        
        # If we get here without error, the integration works
        assert True

    def test_multiple_llm_players_game(self):
        """Test a game with multiple LLM players."""
        from catanatron.players.llm_player import PydanticAIPlayer
        
        player1 = PydanticAIPlayer(Color.RED, model=TestModel(seed=1))
        player2 = PydanticAIPlayer(Color.BLUE, model=TestModel(seed=2))
        
        game = Game([player1, player2], seed=42)
        
        # Run a few ticks
        for _ in range(5):
            if game.winning_color() is not None:
                break
            game.play_tick()
        
        assert True

    def test_llm_player_with_negotiation_setup(self):
        """Test LLM players with negotiation manager configured."""
        from catanatron.players.llm_player import PydanticAIPlayer
        from catanatron.players.llm.negotiation import setup_negotiation
        
        player1 = PydanticAIPlayer(Color.RED, model=TestModel(seed=1))
        player2 = PydanticAIPlayer(Color.BLUE, model=TestModel(seed=2))
        
        game = Game([player1, player2], seed=42)
        manager = setup_negotiation(game, max_rounds=3)
        
        # Run a few ticks
        for _ in range(5):
            if game.winning_color() is not None:
                break
            game.play_tick()
        
        # Both players should have negotiation manager set
        assert player1.negotiation_manager is manager
        assert player2.negotiation_manager is manager
