"""
FairTrade Accumulator — measures the "fairness" of player-to-player trades.

For every CONFIRM_TRADE action, this accumulator:
  1. Runs MCTS win-probability analysis on the pre-trade game state
  2. Simulates the trade on a game copy to obtain post-trade probabilities
  3. Computes the per-player probability delta and a fairness score
  4. Logs everything into a per-game JSON file

The fairness_score is defined as initiator_delta + partner_delta:
  - Near 0  → the trade was roughly zero-sum between the two parties
  - Positive → both players gained (at the expense of the other players)
  - Negative → one or both players lost net probability

Usage:
    from catanatron.cli.fair_trade_accumulator import FairTradeAccumulator

    accumulator = FairTradeAccumulator(num_simulations=100)
    game.play(accumulators=[accumulator])
"""

import json
import os

from catanatron.game import GameAccumulator, Game
from catanatron.models.enums import ActionType
from catanatron.players.mcts import StateNode

# Resource names in the same order as the engine's 5-element frequency decks
RESOURCE_NAMES = ["wood", "brick", "sheep", "wheat", "ore"]


def _analyze_win_probabilities(game, num_simulations):
    """Run MCTS simulations on a game state and return per-player win probabilities.

    This is a standalone reimplementation of the logic in
    ``catanatron.web.mcts_analysis.GameAnalyzer`` so that we do **not** need
    to import the ``catanatron.web`` package (which pulls in Flask).

    Args:
        game: A ``Game`` instance (will be copied internally).
        num_simulations: Number of MCTS rollouts to run.

    Returns:
        dict mapping color value strings to win percentages (0–100).
    """
    if game.winning_color() is not None:
        winner = game.winning_color()
        return {
            winner.value: 100.0,
            **{c.value: 0.0 for c in game.state.colors if c != winner},
        }

    # Create root node and run simulations
    root = StateNode(
        game.state.current_color(), game.copy(), None, prunning=True
    )
    for _ in range(num_simulations):
        root.run_simulation()

    # Calculate probabilities using MCTS statistics
    probabilities = {}
    for color in game.state.colors:
        if color == root.color:
            win_ratio = root.wins / root.visits if root.visits > 0 else 0
        else:
            # StateNode only tracks wins for the root color; distribute the
            # remaining wins evenly among the other players.
            remaining_wins = root.visits - root.wins
            num_other_players = len(game.state.colors) - 1
            win_ratio = (
                (remaining_wins / num_other_players) / root.visits
                if root.visits > 0
                else 0
            )

        probabilities[color.value] = round(win_ratio * 100, 1)

    return probabilities


class FairTradeAccumulator(GameAccumulator):
    """Tracks win-probability shifts around player-to-player trades."""

    def __init__(self, num_simulations=100, output_dir="fair_trade_logs"):
        """
        Args:
            num_simulations: Number of MCTS rollouts per analysis (run twice
                per trade — before and after).
            output_dir: Directory where per-game JSON logs are written.
        """
        self.num_simulations = num_simulations
        self.output_dir = output_dir

        # Per-game state (reset in ``before``)
        self.game_id = None
        self.trade_records = []
        self.trade_count = 0

    # ------------------------------------------------------------------
    # GameAccumulator lifecycle
    # ------------------------------------------------------------------

    def before(self, game: Game):
        """Called once when the game starts."""
        self.game_id = game.id
        self.trade_records = []
        self.trade_count = 0

    def step(self, game_before_action: Game, action):
        """Called before every action is applied to the game state.

        We intercept CONFIRM_TRADE actions to measure the win-probability
        shift caused by the trade.
        """
        if action.action_type != ActionType.CONFIRM_TRADE:
            return

        self.trade_count += 1

        # --- Parse trade details from the action value ---
        # CONFIRM_TRADE value is an 11-tuple:
        #   [0:5]  = offering resources (wood, brick, sheep, wheat, ore)
        #   [5:10] = asking resources
        #   [10]   = color of the accepting player (partner)
        offering_values = action.value[:5]
        asking_values = action.value[5:10]
        partner_color = action.value[10]
        initiator_color = action.color

        offering = {
            RESOURCE_NAMES[i]: v
            for i, v in enumerate(offering_values)
            if v > 0
        }
        asking = {
            RESOURCE_NAMES[i]: v
            for i, v in enumerate(asking_values)
            if v > 0
        }

        # --- 1. Pre-trade win probabilities ---
        pre_trade_probs = _analyze_win_probabilities(
            game_before_action, self.num_simulations
        )

        # --- 2. Simulate the trade on a copy ---
        game_copy = game_before_action.copy()
        game_copy.execute(action, validate_action=True)
        post_trade_probs = _analyze_win_probabilities(
            game_copy, self.num_simulations
        )

        # --- 3. Compute deltas ---
        probability_deltas = {}
        for color_value in pre_trade_probs:
            pre = pre_trade_probs.get(color_value, 0.0)
            post = post_trade_probs.get(color_value, 0.0)
            probability_deltas[color_value] = round(post - pre, 2)

        initiator_delta = probability_deltas.get(initiator_color.value, 0.0)
        partner_delta = probability_deltas.get(partner_color.value, 0.0)
        fairness_score = round(initiator_delta + partner_delta, 2)

        # --- 4. Build record ---
        record = {
            "trade_number": self.trade_count,
            "turn": game_before_action.state.num_turns,
            "initiator": initiator_color.value,
            "partner": partner_color.value,
            "offering": offering,
            "asking": asking,
            "pre_trade_probabilities": pre_trade_probs,
            "post_trade_probabilities": post_trade_probs,
            "probability_deltas": probability_deltas,
            "fairness_score": fairness_score,
        }
        self.trade_records.append(record)

    def after(self, game: Game):
        """Called when the game ends. Writes the trade log to disk."""
        if not self.trade_records:
            return  # nothing to write if no player trades happened

        os.makedirs(self.output_dir, exist_ok=True)

        log = {
            "game_id": self.game_id,
            "num_trades": len(self.trade_records),
            "trades": self.trade_records,
        }

        filepath = os.path.join(
            self.output_dir, f"{self.game_id}_fair_trade.json"
        )
        with open(filepath, "w") as f:
            json.dump(log, f, indent=4)

        print(f"📊 FairTrade log saved to {filepath}")
