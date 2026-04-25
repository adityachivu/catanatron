"""
Tests for the FairTrade accumulator.

Validates that the accumulator:
  - Ignores non-trade actions
  - Correctly captures CONFIRM_TRADE actions and runs MCTS analysis
  - Produces valid JSON output with the expected schema
"""

import json
import os
import shutil
import tempfile

import pytest

from catanatron.game import Game
from catanatron.models.player import Color, RandomPlayer
from catanatron.models.enums import Action, ActionType
from catanatron.cli.fair_trade_accumulator import FairTradeAccumulator


@pytest.fixture
def output_dir(tmp_path):
    """Provide a temporary directory for test outputs."""
    d = tmp_path / "fair_trade_test_logs"
    d.mkdir()
    return str(d)


@pytest.fixture
def two_player_game():
    """Create a simple 2-player game for testing."""
    players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
    game = Game(players, seed=42)
    return game


class TestFairTradeAccumulator:
    """Unit tests for FairTradeAccumulator."""

    def test_init_defaults(self):
        """Test default initialization values."""
        acc = FairTradeAccumulator()
        assert acc.num_simulations == 100
        assert acc.output_dir == "fair_trade_logs"
        assert acc.trade_records == []
        assert acc.trade_count == 0

    def test_init_custom(self, output_dir):
        """Test custom initialization values."""
        acc = FairTradeAccumulator(num_simulations=50, output_dir=output_dir)
        assert acc.num_simulations == 50
        assert acc.output_dir == output_dir

    def test_before_resets_state(self, two_player_game, output_dir):
        """Test that before() resets per-game state."""
        acc = FairTradeAccumulator(output_dir=output_dir)
        # Simulate some leftover state
        acc.trade_records = [{"fake": "data"}]
        acc.trade_count = 5

        acc.before(two_player_game)

        assert acc.game_id == two_player_game.id
        assert acc.trade_records == []
        assert acc.trade_count == 0

    def test_step_ignores_non_trade_actions(self, two_player_game, output_dir):
        """Test that step() ignores actions other than CONFIRM_TRADE."""
        acc = FairTradeAccumulator(num_simulations=10, output_dir=output_dir)
        acc.before(two_player_game)

        # Feed it a non-trade action (e.g., END_TURN)
        non_trade_action = Action(Color.RED, ActionType.END_TURN, None)
        acc.step(two_player_game, non_trade_action)

        assert acc.trade_count == 0
        assert acc.trade_records == []

    def test_step_captures_confirm_trade(self, two_player_game, output_dir):
        """Test that step() correctly processes a CONFIRM_TRADE action.

        We need to set up the game state so that a CONFIRM_TRADE is valid.
        This means we need to go through the proper trade flow.
        """
        acc = FairTradeAccumulator(num_simulations=10, output_dir=output_dir)
        acc.before(two_player_game)

        game = two_player_game
        # Manually set up trade state: give both players resources
        from catanatron.state_functions import player_key, player_deck_replenish

        red_key = player_key(game.state, Color.RED)
        blue_key = player_key(game.state, Color.BLUE)

        # Give RED wood and brick, BLUE wheat
        game.state.player_state[f"{red_key}_WOOD_IN_HAND"] = 3
        game.state.player_state[f"{red_key}_BRICK_IN_HAND"] = 3
        game.state.player_state[f"{blue_key}_WHEAT_IN_HAND"] = 3

        # Manually set up trade resolution state
        game.state.is_resolving_trade = True
        # offering: 1 wood, 0 brick, 0 sheep, 0 wheat, 0 ore
        # asking: 0 wood, 0 brick, 0 sheep, 1 wheat, 0 ore
        trade_value = (1, 0, 0, 0, 0, 0, 0, 0, 1, 0)
        game.state.current_trade = (*trade_value, game.state.colors.index(Color.RED))
        game.state.acceptees = (False, True)  # BLUE accepted

        # Create the CONFIRM_TRADE action
        # value = offering[5] + asking[5] + partner_color
        confirm_value = (1, 0, 0, 0, 0, 0, 0, 0, 1, 0, Color.BLUE)
        confirm_action = Action(Color.RED, ActionType.CONFIRM_TRADE, confirm_value)
        game.playable_actions = [confirm_action]

        acc.step(game, confirm_action)

        assert acc.trade_count == 1
        assert len(acc.trade_records) == 1

        record = acc.trade_records[0]
        assert record["trade_number"] == 1
        assert record["initiator"] == "RED"
        assert record["partner"] == "BLUE"
        assert record["offering"] == {"wood": 1}
        assert record["asking"] == {"wheat": 1}
        assert "pre_trade_probabilities" in record
        assert "post_trade_probabilities" in record
        assert "probability_deltas" in record
        assert "fairness_score" in record

    def test_after_skips_when_no_trades(self, two_player_game, output_dir):
        """Test that after() does not write a file when there were no trades."""
        acc = FairTradeAccumulator(output_dir=output_dir)
        acc.before(two_player_game)
        acc.after(two_player_game)

        # Directory should be empty (no file written)
        files = os.listdir(output_dir)
        assert len(files) == 0

    def test_after_writes_json(self, two_player_game, output_dir):
        """Test that after() writes a valid JSON file when trades were recorded."""
        acc = FairTradeAccumulator(num_simulations=10, output_dir=output_dir)
        acc.before(two_player_game)

        # Inject a fake trade record to test JSON writing
        acc.trade_records.append({
            "trade_number": 1,
            "turn": 10,
            "initiator": "RED",
            "partner": "BLUE",
            "offering": {"wood": 1},
            "asking": {"wheat": 1},
            "pre_trade_probabilities": {"RED": 50.0, "BLUE": 50.0},
            "post_trade_probabilities": {"RED": 55.0, "BLUE": 45.0},
            "probability_deltas": {"RED": 5.0, "BLUE": -5.0},
            "fairness_score": 0.0,
        })

        acc.after(two_player_game)

        # Check that the file was created
        files = os.listdir(output_dir)
        assert len(files) == 1
        assert files[0].endswith("_fair_trade.json")

        # Parse and validate JSON
        filepath = os.path.join(output_dir, files[0])
        with open(filepath, "r") as f:
            data = json.load(f)

        assert data["game_id"] == two_player_game.id
        assert data["num_trades"] == 1
        assert len(data["trades"]) == 1

        trade = data["trades"][0]
        assert trade["trade_number"] == 1
        assert trade["initiator"] == "RED"
        assert trade["partner"] == "BLUE"
        assert "offering" in trade
        assert "asking" in trade
        assert "fairness_score" in trade

    def test_json_schema_completeness(self, output_dir):
        """Test that the JSON output has all required fields."""
        required_trade_fields = {
            "trade_number",
            "turn",
            "initiator",
            "partner",
            "offering",
            "asking",
            "pre_trade_probabilities",
            "post_trade_probabilities",
            "probability_deltas",
            "fairness_score",
        }

        acc = FairTradeAccumulator(num_simulations=10, output_dir=output_dir)
        players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
        game = Game(players, seed=123)
        acc.before(game)

        # Inject a complete trade record
        acc.trade_records.append({
            "trade_number": 1,
            "turn": 5,
            "initiator": "RED",
            "partner": "BLUE",
            "offering": {"brick": 2},
            "asking": {"ore": 1},
            "pre_trade_probabilities": {"RED": 40.0, "BLUE": 60.0},
            "post_trade_probabilities": {"RED": 45.0, "BLUE": 55.0},
            "probability_deltas": {"RED": 5.0, "BLUE": -5.0},
            "fairness_score": 0.0,
        })

        acc.after(game)

        filepath = os.path.join(output_dir, f"{game.id}_fair_trade.json")
        with open(filepath, "r") as f:
            data = json.load(f)

        assert "game_id" in data
        assert "num_trades" in data
        assert "trades" in data

        for trade in data["trades"]:
            assert required_trade_fields.issubset(trade.keys()), (
                f"Missing fields: {required_trade_fields - trade.keys()}"
            )
