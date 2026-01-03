"""
Integration tests for LLM negotiation interface.

These tests verify:
1. StrategicAdvisor correctly ranks actions
2. StateRenderer produces valid output
3. NegotiationProtocol works correctly
4. MockLLMPlayer can play a complete game
5. NegotiationManager coordinates conversations

For tests requiring actual LLM calls, use pytest.mark.slow
and ensure API keys are configured.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from catanatron.game import Game
from catanatron.models.player import Player, RandomPlayer, Color
from catanatron.models.enums import ActionType, RESOURCES

# Import the new LLM modules
from catanatron.players.llm.strategic_advisor import StrategicAdvisor, ActionRanking
from catanatron.players.llm.state_renderer import StateRenderer
from catanatron.players.llm.memory import (
    NegotiationMemory, 
    StatelessMemory, 
    PromiseType,
    Promise,
)
from catanatron.players.llm.providers import LLMConfig, LLMProvider
from catanatron.players.llm.prompts import (
    get_system_prompt,
    build_decision_prompt,
    PERSONAS,
)
from catanatron.players.llm.player import (
    LLMNegotiatingPlayer,
    LLMPlayerConfig,
    MockLLMPlayer,
    create_llm_player,
    create_mock_player,
)
from catanatron.negotiation.protocol import (
    NegotiationIntent,
    NegotiationMessage,
    TradeProposal,
    NegotiationSession,
    create_proposal_message,
    create_accept_message,
)
from catanatron.negotiation.manager import (
    NegotiationManager,
    NegotiationConfig,
    create_trade_action_from_proposal,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def two_player_game():
    """Create a simple 2-player game."""
    players = [
        RandomPlayer(Color.RED),
        RandomPlayer(Color.BLUE),
    ]
    game = Game(players, seed=42)
    return game


@pytest.fixture
def four_player_game():
    """Create a 4-player game."""
    players = [
        RandomPlayer(Color.RED),
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.ORANGE),
        RandomPlayer(Color.WHITE),
    ]
    game = Game(players, seed=42)
    return game


@pytest.fixture
def mid_game(two_player_game):
    """Advance game to mid-game state (past initial placement)."""
    game = two_player_game
    # Play some turns to get past initial placement
    for _ in range(50):
        if game.winning_color() is not None:
            break
        game.play_tick()
    return game


# =============================================================================
# STRATEGIC ADVISOR TESTS
# =============================================================================

class TestStrategicAdvisor:
    """Tests for the StrategicAdvisor class."""
    
    def test_advisor_creation(self):
        """Test advisor can be created."""
        advisor = StrategicAdvisor()
        assert advisor is not None
        assert advisor.value_fn is not None
    
    def test_rank_actions_returns_rankings(self, two_player_game):
        """Test that rank_actions returns ActionRanking objects."""
        advisor = StrategicAdvisor()
        game = two_player_game
        
        rankings = advisor.rank_actions(game, game.playable_actions, top_n=3)
        
        assert isinstance(rankings, list)
        assert len(rankings) <= 3
        
        for ranking in rankings:
            assert isinstance(ranking, ActionRanking)
            assert ranking.action is not None
            assert isinstance(ranking.value, float)
            assert isinstance(ranking.explanation, str)
            assert ranking.rank >= 1
    
    def test_rankings_are_sorted_by_value(self, mid_game):
        """Test that rankings are sorted descending by value."""
        advisor = StrategicAdvisor()
        
        rankings = advisor.rank_actions(mid_game, mid_game.playable_actions, top_n=5)
        
        if len(rankings) > 1:
            values = [r.value for r in rankings]
            assert values == sorted(values, reverse=True)
    
    def test_normalized_values_in_range(self, mid_game):
        """Test that normalized values are in [0, 1] range."""
        advisor = StrategicAdvisor()
        
        rankings = advisor.rank_actions(mid_game, mid_game.playable_actions, top_n=5)
        
        for ranking in rankings:
            assert 0 <= ranking.normalized_value <= 1
    
    def test_empty_actions_returns_empty_list(self, two_player_game):
        """Test that empty actions list returns empty rankings."""
        advisor = StrategicAdvisor()
        
        rankings = advisor.rank_actions(two_player_game, [], top_n=5)
        
        assert rankings == []
    
    def test_explanation_for_different_action_types(self, mid_game):
        """Test that explanations are generated for different action types."""
        advisor = StrategicAdvisor()
        
        rankings = advisor.rank_actions(mid_game, mid_game.playable_actions, top_n=10)
        
        # All should have explanations
        for ranking in rankings:
            assert ranking.explanation
            assert len(ranking.explanation) > 0


# =============================================================================
# STATE RENDERER TESTS
# =============================================================================

class TestStateRenderer:
    """Tests for the StateRenderer class."""
    
    def test_renderer_creation(self):
        """Test renderer can be created."""
        renderer = StateRenderer()
        assert renderer is not None
    
    def test_render_full_context(self, two_player_game):
        """Test that full context is rendered."""
        renderer = StateRenderer()
        game = two_player_game
        color = game.state.current_color()
        
        context = renderer.render_full_context(game, color)
        
        assert isinstance(context, str)
        assert len(context) > 0
        assert color.value in context
        assert "GAME STATE" in context
    
    def test_render_includes_resources(self, mid_game):
        """Test that rendered context includes resource information."""
        renderer = StateRenderer()
        color = mid_game.state.current_color()
        
        context = renderer.render_full_context(mid_game, color)
        
        assert "YOUR HAND" in context or "Resources" in context
    
    def test_render_includes_other_players(self, four_player_game):
        """Test that rendered context includes other players."""
        renderer = StateRenderer()
        color = four_player_game.state.current_color()
        
        context = renderer.render_full_context(four_player_game, color)
        
        assert "OTHER PLAYERS" in context
    
    def test_render_with_rankings(self, mid_game):
        """Test rendering with strategic rankings."""
        renderer = StateRenderer()
        advisor = StrategicAdvisor()
        
        color = mid_game.state.current_color()
        rankings = advisor.rank_actions(mid_game, mid_game.playable_actions, top_n=3)
        
        context = renderer.render_full_context(
            mid_game, 
            color, 
            rankings=rankings
        )
        
        assert "STRATEGIC RECOMMENDATIONS" in context


# =============================================================================
# MEMORY TESTS
# =============================================================================

class TestNegotiationMemory:
    """Tests for the NegotiationMemory class."""
    
    def test_memory_creation(self):
        """Test memory can be created."""
        memory = NegotiationMemory(Color.RED)
        assert memory is not None
        assert memory.my_color == Color.RED
    
    def test_record_promise(self):
        """Test recording a promise."""
        memory = NegotiationMemory(Color.RED)
        
        promise = memory.record_promise(
            maker=Color.BLUE,
            recipient=Color.RED,
            promise_type=PromiseType.WONT_ROBBER,
            description="Won't place robber on your tiles",
            current_turn=5,
            expiration_turn=10,
        )
        
        assert promise is not None
        assert promise.maker == Color.BLUE
        assert promise.recipient == Color.RED
        assert promise.is_active(current_turn=7)
        assert not promise.is_active(current_turn=11)
    
    def test_get_promises_to_me(self):
        """Test getting promises made to the player."""
        memory = NegotiationMemory(Color.RED)
        
        memory.record_promise(
            maker=Color.BLUE,
            recipient=Color.RED,
            promise_type=PromiseType.WILL_TRADE,
            description="Will trade wheat for ore",
            current_turn=5,
        )
        
        memory.record_promise(
            maker=Color.RED,
            recipient=Color.BLUE,
            promise_type=PromiseType.WONT_ROBBER,
            description="Won't robber you",
            current_turn=5,
        )
        
        promises_to_me = memory.get_promises_to_me(current_turn=5)
        
        assert len(promises_to_me) == 1
        assert promises_to_me[0].maker == Color.BLUE
    
    def test_reputation_tracking(self):
        """Test reputation score tracking."""
        memory = NegotiationMemory(Color.RED)
        
        memory._ensure_reputation(Color.BLUE)
        memory.reputation[Color.BLUE].trades_accepted = 3
        memory.reputation[Color.BLUE].trades_rejected = 1
        memory.reputation[Color.BLUE].promises_kept = 2
        memory.reputation[Color.BLUE].promises_broken = 0
        
        rep = memory.get_reputation(Color.BLUE)
        
        assert rep.trade_acceptance_rate == 0.75
        assert rep.promise_reliability == 1.0
        assert rep.overall_trust_score > 0.5
    
    def test_get_context_renders_text(self):
        """Test that get_context returns formatted text."""
        memory = NegotiationMemory(Color.RED)
        
        memory.record_promise(
            maker=Color.BLUE,
            recipient=Color.RED,
            promise_type=PromiseType.ALLIANCE,
            description="We'll work together against Orange",
            current_turn=5,
        )
        
        context = memory.get_context(current_turn=5)
        
        assert isinstance(context, str)
        assert "BLUE" in context
    
    def test_stateless_memory(self):
        """Test that StatelessMemory returns empty context."""
        memory = StatelessMemory(Color.RED)
        
        context = memory.get_context(current_turn=5)
        
        assert "stateless" in context.lower() or "no prior" in context.lower()


# =============================================================================
# LLM PROVIDER TESTS
# =============================================================================

class TestLLMProviders:
    """Tests for the LLM provider abstraction."""
    
    def test_config_creation(self):
        """Test LLMConfig can be created."""
        config = LLMConfig.openai("gpt-4")
        assert config.provider == LLMProvider.OPENAI
        assert config.model == "gpt-4"
    
    def test_anthropic_config(self):
        """Test Anthropic config creation."""
        config = LLMConfig.anthropic("claude-3-sonnet-20240229")
        assert config.provider == LLMProvider.ANTHROPIC
    
    def test_ollama_config(self):
        """Test Ollama config creation."""
        config = LLMConfig.ollama("llama3")
        assert config.provider == LLMProvider.OLLAMA
        assert config.base_url is not None
    
    def test_pydantic_ai_model_string(self):
        """Test model string generation."""
        config = LLMConfig.openai("gpt-4")
        assert config.get_pydantic_ai_model_string() == "openai:gpt-4"
        
        config = LLMConfig.anthropic("claude-3-sonnet")
        assert config.get_pydantic_ai_model_string() == "anthropic:claude-3-sonnet"


# =============================================================================
# PROMPT TESTS
# =============================================================================

class TestPrompts:
    """Tests for the prompt templates."""
    
    def test_system_prompt_generation(self):
        """Test system prompt can be generated."""
        prompt = get_system_prompt("balanced")
        
        assert isinstance(prompt, str)
        assert len(prompt) > 100
        assert "Catan" in prompt
    
    def test_personas_available(self):
        """Test all personas are available."""
        assert "cooperative" in PERSONAS
        assert "competitive" in PERSONAS
        assert "balanced" in PERSONAS
    
    def test_decision_prompt_building(self):
        """Test decision prompt building."""
        prompt = build_decision_prompt(
            game_state_text="Test game state",
            rankings_text="Test rankings",
            memory_text="Test memory",
        )
        
        assert "Test game state" in prompt
        assert "YOUR DECISION" in prompt


# =============================================================================
# LLM PLAYER TESTS
# =============================================================================

class TestMockLLMPlayer:
    """Tests for the MockLLMPlayer."""
    
    def test_mock_player_creation(self):
        """Test mock player can be created."""
        player = create_mock_player(Color.RED)
        assert player is not None
        assert player.color == Color.RED
    
    def test_mock_player_decide(self, two_player_game):
        """Test mock player can decide."""
        player = MockLLMPlayer(Color.RED, strategy="top_ranked")
        
        # Simulate being this player's turn
        game = two_player_game
        
        action = player.decide(game, game.playable_actions)
        
        assert action is not None
        assert action in game.playable_actions
    
    def test_mock_player_strategies(self, mid_game):
        """Test different mock player strategies."""
        strategies = ["top_ranked", "second_best", "random"]
        
        for strategy in strategies:
            player = MockLLMPlayer(Color.RED, strategy=strategy)
            action = player.decide(mid_game, mid_game.playable_actions)
            assert action is not None


class TestLLMNegotiatingPlayer:
    """Tests for the LLMNegotiatingPlayer (without actual LLM calls)."""
    
    def test_player_creation(self):
        """Test LLM player can be created."""
        config = LLMPlayerConfig(
            llm_config=LLMConfig.openai("gpt-4"),
            use_stateless_memory=True,
        )
        player = LLMNegotiatingPlayer(Color.RED, config)
        
        assert player is not None
        assert player.color == Color.RED
    
    def test_factory_function(self):
        """Test create_llm_player factory."""
        player = create_llm_player(
            Color.BLUE,
            provider="openai",
            model="gpt-4",
            persona="competitive"
        )
        
        assert player is not None
        assert player.color == Color.BLUE
        assert player.config.persona == "competitive"
    
    def test_trivial_decision(self, two_player_game):
        """Test that single-option decisions are handled."""
        config = LLMPlayerConfig(
            llm_config=LLMConfig.openai("gpt-4"),
            use_stateless_memory=True,
            fallback_to_random=True,
        )
        player = LLMNegotiatingPlayer(Color.RED, config)
        
        # Single action = trivial decision
        single_action = [two_player_game.playable_actions[0]]
        action = player.decide(two_player_game, single_action)
        
        assert action == single_action[0]


# =============================================================================
# NEGOTIATION PROTOCOL TESTS
# =============================================================================

class TestNegotiationProtocol:
    """Tests for the negotiation protocol."""
    
    def test_trade_proposal_creation(self):
        """Test TradeProposal creation."""
        proposal = TradeProposal(
            offering=(1, 0, 0, 0, 0),  # 1 wood
            asking=(0, 0, 1, 0, 0),     # 1 sheep
        )
        
        assert proposal.is_valid()
        assert "wood" in proposal.format_readable().lower()
        assert "sheep" in proposal.format_readable().lower()
    
    def test_invalid_proposal_detection(self):
        """Test invalid proposal detection."""
        # Offering nothing
        proposal1 = TradeProposal(
            offering=(0, 0, 0, 0, 0),
            asking=(0, 0, 1, 0, 0),
        )
        assert not proposal1.is_valid()
        
        # Asking nothing
        proposal2 = TradeProposal(
            offering=(1, 0, 0, 0, 0),
            asking=(0, 0, 0, 0, 0),
        )
        assert not proposal2.is_valid()
        
        # Trading same resource
        proposal3 = TradeProposal(
            offering=(1, 0, 0, 0, 0),
            asking=(1, 0, 0, 0, 0),
        )
        assert not proposal3.is_valid()
    
    def test_trade_tuple_conversion(self):
        """Test conversion to/from Catanatron trade tuple."""
        proposal = TradeProposal(
            offering=(1, 2, 0, 0, 0),
            asking=(0, 0, 0, 1, 1),
        )
        
        trade_tuple = proposal.to_trade_tuple()
        assert trade_tuple == (1, 2, 0, 0, 0, 0, 0, 0, 1, 1)
        
        recovered = TradeProposal.from_trade_tuple(trade_tuple)
        assert recovered.offering == proposal.offering
        assert recovered.asking == proposal.asking
    
    def test_negotiation_message(self):
        """Test NegotiationMessage creation."""
        message = NegotiationMessage(
            sender=Color.RED,
            content="I'd like to trade some wood for sheep",
            intent=NegotiationIntent.PROPOSE,
        )
        
        assert message.is_broadcast()
        assert "RED" in message.format_for_log()
    
    def test_negotiation_session(self):
        """Test NegotiationSession management."""
        session = NegotiationSession(
            session_id="test123",
            initiator=Color.RED,
            participants=[Color.RED, Color.BLUE, Color.ORANGE],
            start_turn=10,
        )
        
        # Add opening message
        session.add_message(NegotiationMessage(
            sender=Color.RED,
            content="Anyone want to trade?",
            intent=NegotiationIntent.GREETING,
        ))
        
        assert len(session.rounds) == 1
        assert len(session.get_all_messages()) == 1
        
        # Add response
        session.add_message(NegotiationMessage(
            sender=Color.BLUE,
            content="What do you have?",
            intent=NegotiationIntent.QUESTION,
        ))
        
        assert len(session.get_all_messages()) == 2
        
        # Conclude
        session.conclude(outcome="no_agreement")
        assert not session.is_active


# =============================================================================
# NEGOTIATION MANAGER TESTS
# =============================================================================

class TestNegotiationManager:
    """Tests for the NegotiationManager."""
    
    def test_manager_creation(self):
        """Test manager can be created."""
        players = {
            Color.RED: RandomPlayer(Color.RED),
            Color.BLUE: RandomPlayer(Color.BLUE),
        }
        manager = NegotiationManager(players)
        
        assert manager is not None
    
    def test_start_negotiation(self, two_player_game):
        """Test starting a negotiation session."""
        game = two_player_game
        players = {p.color: p for p in game.state.players}
        manager = NegotiationManager(players)
        
        session = manager.start_negotiation(
            game=game,
            initiator=Color.RED,
            opening_message="Looking to trade wood for sheep",
            initial_proposal=TradeProposal(
                offering=(1, 0, 0, 0, 0),
                asking=(0, 0, 1, 0, 0),
            ),
        )
        
        assert session is not None
        assert session.initiator == Color.RED
        assert session.is_active
        assert len(session.get_all_messages()) == 1
    
    def test_intent_detection(self):
        """Test intent detection from message text."""
        players = {Color.RED: RandomPlayer(Color.RED)}
        manager = NegotiationManager(players)
        
        assert manager._detect_intent("Yes, I accept the deal") == NegotiationIntent.ACCEPT
        assert manager._detect_intent("No, I don't want that") == NegotiationIntent.REJECT
        assert manager._detect_intent("How about 2 wood instead?") == NegotiationIntent.COUNTER
        assert manager._detect_intent("What resources do you have?") == NegotiationIntent.QUESTION
    
    def test_create_trade_action_from_proposal(self):
        """Test creating trade action from proposal."""
        proposal = TradeProposal(
            offering=(1, 0, 0, 0, 0),
            asking=(0, 0, 1, 0, 0),
        )
        
        action = create_trade_action_from_proposal(Color.RED, proposal)
        
        assert action.color == Color.RED
        assert action.action_type == ActionType.OFFER_TRADE
        assert action.value == (1, 0, 0, 0, 0, 0, 0, 1, 0, 0)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the complete LLM negotiation system."""
    
    def test_mock_player_plays_full_game(self):
        """Test that mock LLM players can play a complete game."""
        players = [
            MockLLMPlayer(Color.RED, strategy="top_ranked"),
            MockLLMPlayer(Color.BLUE, strategy="top_ranked"),
        ]
        
        game = Game(players, seed=42)
        
        # Play up to 200 turns (should be enough to finish or timeout)
        for _ in range(200):
            if game.winning_color() is not None:
                break
            game.play_tick()
        
        # Game should have progressed
        assert game.state.num_turns > 0
    
    def test_strategic_advisor_with_renderer(self, mid_game):
        """Test strategic advisor and renderer work together."""
        advisor = StrategicAdvisor()
        renderer = StateRenderer()
        
        color = mid_game.state.current_color()
        rankings = advisor.rank_actions(mid_game, mid_game.playable_actions, top_n=5)
        
        context = renderer.render_full_context(
            mid_game,
            color,
            rankings=rankings,
        )
        
        # Should produce valid context with rankings
        assert "STRATEGIC RECOMMENDATIONS" in context
        assert len(context) > 500  # Reasonable length
    
    def test_negotiation_flow(self, four_player_game):
        """Test complete negotiation flow."""
        game = four_player_game
        players = {p.color: p for p in game.state.players}
        manager = NegotiationManager(players)
        
        # Start negotiation
        proposal = TradeProposal(
            offering=(2, 0, 0, 0, 0),
            asking=(0, 0, 0, 1, 0),
        )
        
        session = manager.start_negotiation(
            game=game,
            initiator=Color.RED,
            opening_message="Offering 2 wood for 1 wheat. Good deal!",
            initial_proposal=proposal,
        )
        
        # Simulate acceptance response
        accept_message = create_accept_message(
            sender=Color.BLUE,
            message="Sounds good, I'll take that deal.",
            proposer=Color.RED,
            accepted_proposal=proposal,
        )
        session.add_message(accept_message)
        
        # Check session state
        assert session.current_round().has_acceptance()
        
        # Conclude with agreement
        session.conclude(
            outcome="agreement",
            agreement=proposal,
            agreed_parties=[Color.RED, Color.BLUE]
        )
        
        assert not session.is_active
        assert session.outcome == "agreement"
        assert session.final_agreement == proposal


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

