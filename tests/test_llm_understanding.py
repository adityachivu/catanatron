"""
LLM Understanding Harness
=========================
Tests the LLM agent's comprehension of game and board state through natural-language
questions rather than action selection.

Each `UnderstandingScenario` constructs a deterministic game state, formats it
using the same `StateFormatter` pipeline as the live game, then asks an open-ended
question.  The agent uses a custom per-scenario system prompt and returns a plain
string — no action indices, no tools.

Scenarios are organised into four categories:
  GAME_STATE  – pure game-logic reasoning (VP calc, dev cards, largest army)
  TRADE       – trade evaluation and defensive reasoning
  BOARD_STATE – spatial / topological reasoning (adjacency, ports, robber hex)
  JOINT       – combined game + board reasoning (robber targeting, road races)

Usage
-----
# Run all scenarios with real API calls (requires CATAN_LLM_MODEL env var):
    python tests/test_llm_understanding.py

# Smoke-test without API calls (uses TestModel):
    CATAN_LLM_TEST_MODE=1 pytest tests/test_llm_understanding.py -v

# Run against a specific model:
    CATAN_LLM_MODEL=anthropic:claude-sonnet-4-20250514 python tests/test_llm_understanding.py

Adding new scenarios
--------------------
1. Write a `setup_*()` function that returns a fully-configured `Game`.
2. Create an `UnderstandingScenario(...)` instance with the fields below.
3. Append it to `SCENARIOS`.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# sys.path bootstrap — makes the file runnable both via pytest and directly:
#   pytest: adds project root + catanatron/ via pytest.ini pythonpath setting
#   script: neither is on sys.path by default, so we add them here
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CATANATRON_SRC = _PROJECT_ROOT / "catanatron"
for _p in [str(_PROJECT_ROOT), str(_CATANATRON_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest
from pydantic_ai import Agent

from catanatron.game import Game
from catanatron.models.enums import Action, ActionType
from catanatron.models.player import Color, SimplePlayer
from catanatron.state_functions import (
    player_freqdeck_add,
    player_key,
    get_visible_victory_points,
    get_actual_victory_points,
)

from catanatron.players.llm.models import ModelInput, create_model
from catanatron.players.llm.state_formatter import StateFormatter

from tests.utils import advance_to_play_turn, build_initial_placements


# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------

CATEGORIES = ("GAME_STATE", "TRADE", "BOARD_STATE", "JOINT")


@dataclass
class UnderstandingScenario:
    """
    A single comprehension test for the LLM.

    Attributes
    ----------
    name:
        Short identifier used in output/test names.
    category:
        One of GAME_STATE, TRADE, BOARD_STATE, JOINT.
    description:
        What cognitive skill or knowledge is being tested.
    setup:
        Callable that returns a freshly-built, deterministic `Game` instance.
        Must be reproducible (use fixed seeds / explicit actions).
    perspective_color:
        The player colour from whose perspective the state is formatted.
    system_prompt:
        Task-specific instruction to the agent (replaces `CATAN_SYSTEM_PROMPT`).
    question:
        The free-form question asked after presenting the game state.
    """

    name: str
    category: str
    description: str
    setup: Callable[[], Game]
    perspective_color: Color
    system_prompt: str
    question: str


# ---------------------------------------------------------------------------
# Scenario helper: shared base game
# ---------------------------------------------------------------------------

def _make_base_game(seed: int = 42) -> Game:
    """
    Create a standard 2-player game, run through initial placements and
    advance into the first proper play turn (dice already rolled).

    Board seed 42 is used throughout so node/hex positions are stable.
    Player 0 = RED,  Player 1 = BLUE.
    Initial placements (from tests/utils defaults):
      RED  → settlements at nodes 0 and 2, roads on edges (0,1) and (1,2)
      BLUE → settlements at nodes 24 and 26, roads on edges (24,25) and (25,26)
    """
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players, seed=seed)
    build_initial_placements(game)   # default node/edge positions from tests/utils
    advance_to_play_turn(game)       # roll dice, resolve DISCARD/MOVE_ROBBER if needed
    return game


# ═══════════════════════════════════════════════════════════════════════════
# CATEGORY: GAME_STATE — Pure Game Logic
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# GS-1: Victory Point Calculation
# ---------------------------------------------------------------------------

def _setup_vp_calculation() -> Game:
    """
    BLUE has: 3 settlements, 1 city, Longest Road (2 VP), and high visible VP.
    We build this by giving BLUE extra settlements/city and enough roads for
    Longest Road.

    Seed-42 board topology used.
    RED  → settlements at nodes 0 and 2, roads (0,1) (1,2)
    BLUE → settlements at nodes 24, 26, 28 + city at node 24
             roads (24,25) (25,26) (26,27) (27,28) (28,29) → road length 5

    BLUE visible VP breakdown (what RED can see):
      2 settlements (nodes 26, 28) = 2 VP  (node 24 upgraded to city)
      1 city (node 24) = 2 VP
      Longest Road   = 2 VP
      Total visible  = 6 VP
    """
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players, seed=42)

    # Custom placement: RED defaults, BLUE gets more roads + extra settlement
    build_initial_placements(
        game,
        p0_actions=[0, (0, 1), 2, (1, 2)],       # RED: standard
        p1_actions=[24, (24, 25), 26, (25, 26)],  # BLUE: standard
    )
    advance_to_play_turn(game)

    state = game.state
    blue_key = player_key(state, Color.BLUE)

    # Give BLUE resources for additional construction
    # Build extra roads for BLUE to get Longest Road
    player_freqdeck_add(state, Color.BLUE, [5, 5, 5, 5, 5])  # plenty of resources

    # Build roads: (26,27), (27,28), (28,29)
    state.board.build_road(Color.BLUE, (26, 27))
    state.player_state[f"{blue_key}_ROADS_AVAILABLE"] -= 1
    state.buildings_by_color[Color.BLUE]["ROAD"].append((26, 27))

    state.board.build_road(Color.BLUE, (27, 28))
    state.player_state[f"{blue_key}_ROADS_AVAILABLE"] -= 1
    state.buildings_by_color[Color.BLUE]["ROAD"].append((27, 28))

    state.board.build_road(Color.BLUE, (28, 29))
    state.player_state[f"{blue_key}_ROADS_AVAILABLE"] -= 1
    state.buildings_by_color[Color.BLUE]["ROAD"].append((28, 29))

    # Build settlement at node 28 (connected to BLUE's road network)
    state.board.build_settlement(Color.BLUE, 28)
    state.player_state[f"{blue_key}_SETTLEMENTS_AVAILABLE"] -= 1
    state.player_state[f"{blue_key}_VICTORY_POINTS"] += 1
    state.player_state[f"{blue_key}_ACTUAL_VICTORY_POINTS"] += 1
    state.buildings_by_color[Color.BLUE]["SETTLEMENT"].append(28)

    # Upgrade node 24 to city
    state.board.build_city(Color.BLUE, 24)
    state.buildings_by_color[Color.BLUE]["SETTLEMENT"].remove(24)
    state.buildings_by_color[Color.BLUE]["CITY"].append(24)
    state.player_state[f"{blue_key}_SETTLEMENTS_AVAILABLE"] += 1
    state.player_state[f"{blue_key}_CITIES_AVAILABLE"] -= 1
    state.player_state[f"{blue_key}_VICTORY_POINTS"] += 1
    state.player_state[f"{blue_key}_ACTUAL_VICTORY_POINTS"] += 1

    # Set Longest Road for BLUE
    state.player_state[f"{blue_key}_HAS_ROAD"] = True
    state.player_state[f"{blue_key}_VICTORY_POINTS"] += 2
    state.player_state[f"{blue_key}_ACTUAL_VICTORY_POINTS"] += 2
    state.player_state[f"{blue_key}_LONGEST_ROAD_LENGTH"] = 5

    # Clean up BLUE's resources so the state is tidy
    player_freqdeck_add(state, Color.BLUE, [-5, -5, -5, -5, -5])

    return game


SCENARIO_VP_CALCULATION = UnderstandingScenario(
    name="victory_point_calculation",
    category="GAME_STATE",
    description=(
        "Tests: VP counting from visible information. "
        "BLUE has settlements, a city, and Longest Road."
    ),
    setup=_setup_vp_calculation,
    perspective_color=Color.RED,
    system_prompt=(
        "You are evaluating a Catan game state. "
        "Answer concisely, showing a clear numerical breakdown."
    ),
    question=(
        "Calculate Player BLUE's visible Victory Points. "
        "Show the breakdown: how many points come from settlements, "
        "cities, and any special achievements?"
    ),
)


# ---------------------------------------------------------------------------
# GS-2: Development Card Logic
# ---------------------------------------------------------------------------

def _setup_dev_card_logic() -> Game:
    """
    RED has 3 Ore, 1 Sheep, 1 Wheat (can afford dev card purchase).
    RED has 9 actual VP (7 visible + 2 hidden VP dev cards).
    Board arranged so buildable_node_ids returns empty for RED.
    RED has no resources for a city (needs 2 Wheat + 3 Ore, only has 1 Wheat).

    The only way to reach 10 VP this turn is buying a dev card and hoping
    for a VP card, since no settlements can be built and no city can be afforded.
    """
    game = _make_base_game()
    state = game.state
    red_key = player_key(state, Color.RED)

    # Set RED's VP: 2 settlements = 2 VP visible already
    # Add 5 more visible VP (pretend RED built more earlier)
    state.player_state[f"{red_key}_VICTORY_POINTS"] = 7
    state.player_state[f"{red_key}_ACTUAL_VICTORY_POINTS"] = 9

    # Give RED 2 VP dev cards in hand (hidden, add +2 to actual only)
    state.player_state[f"{red_key}_VICTORY_POINT_IN_HAND"] = 2

    # Give RED resources: 3 Ore, 1 Sheep, 1 Wheat (can buy dev card)
    player_freqdeck_add(state, Color.RED, [0, 0, 1, 1, 3])

    # Ensure no buildable nodes for RED — occupy all adjacent nodes
    # RED's road network reaches nodes reachable from 0, 1, 2
    # Buildable edges for RED (from inspection): (0,5), (0,20), (1,6), (2,3), (2,9)
    # We need to block all buildable nodes. RED currently has no buildable nodes
    # (confirmed by inspection), so this is already the case with the default layout.

    return game


SCENARIO_DEV_CARD_LOGIC = UnderstandingScenario(
    name="development_card_logic",
    category="GAME_STATE",
    description=(
        "Tests: recognising that buying a dev card is the only path to victory "
        "when no settlements/cities can be built."
    ),
    setup=_setup_dev_card_logic,
    perspective_color=Color.RED,
    system_prompt=(
        "You are a Catan strategy advisor. "
        "Analyse the player's resources, board position, and VP situation. "
        "Answer in 2-4 sentences."
    ),
    question=(
        "Given your current resources and board position, "
        "what action should you take to try to win the game this turn?"
    ),
)


# ---------------------------------------------------------------------------
# GS-3: Largest Army Tracking
# ---------------------------------------------------------------------------

def _setup_largest_army() -> Game:
    """
    BLUE has played 3 Knights → holds Largest Army.
    RED has played 2 Knights, has 1 Knight in hand.
    If RED plays the Knight, RED will have 3 played = ties BLUE.
    Tie does NOT transfer Largest Army — current holder keeps it.
    """
    game = _make_base_game()
    state = game.state
    red_key = player_key(state, Color.RED)
    blue_key = player_key(state, Color.BLUE)

    # BLUE: 3 played knights, has Largest Army
    state.player_state[f"{blue_key}_PLAYED_KNIGHT"] = 3
    state.player_state[f"{blue_key}_HAS_ARMY"] = True
    state.player_state[f"{blue_key}_VICTORY_POINTS"] += 2
    state.player_state[f"{blue_key}_ACTUAL_VICTORY_POINTS"] += 2

    # RED: 2 played knights, 1 knight in hand
    state.player_state[f"{red_key}_PLAYED_KNIGHT"] = 2
    state.player_state[f"{red_key}_KNIGHT_IN_HAND"] = 1
    state.player_state[f"{red_key}_HAS_ARMY"] = False

    return game


SCENARIO_LARGEST_ARMY = UnderstandingScenario(
    name="largest_army_tracking",
    category="GAME_STATE",
    description=(
        "Tests: understanding Largest Army rules — ties do NOT transfer the card."
    ),
    setup=_setup_largest_army,
    perspective_color=Color.RED,
    system_prompt=(
        "You are a Catan rules expert. "
        "Answer precisely, citing the relevant rule about ties."
    ),
    question=(
        "If you play your Knight card this turn, who will hold the Largest Army card? "
        "Explain the tie-breaking rule."
    ),
)


# ═══════════════════════════════════════════════════════════════════════════
# CATEGORY: TRADE — Trade Evaluation Logic
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# TR-1: Trade Evaluation (Math/Logic)
# ---------------------------------------------------------------------------

def _setup_trade_evaluation() -> Game:
    """
    RED has: 2 Sheep, 0 Ore, 1 Wheat, 1 Wood, 0 Brick.
    City costs 2 Wheat + 3 Ore.
    Trade on the table: BLUE offers 1 Ore for RED's 1 Sheep + 1 Wood.

    After trade: 1 Sheep, 1 Ore, 1 Wheat, 0 Wood, 0 Brick.
    Still needs 1 more Wheat + 2 more Ore for a City → trade helps but
    doesn't get RED all the way there.
    """
    game = _make_base_game()
    state = game.state

    # Give RED specific resources
    player_freqdeck_add(state, Color.RED, [1, 0, 2, 1, 0])  # wood=1,brick=0,sheep=2,wheat=1,ore=0

    # Set up trade state: BLUE offers 1 Ore, asks for 1 Sheep + 1 Wood
    # Trade tuple: [offer_wood, offer_brick, offer_sheep, offer_wheat, offer_ore,
    #               ask_wood, ask_brick, ask_sheep, ask_wheat, ask_ore]
    state.is_resolving_trade = True
    state.current_trade = (0, 0, 0, 0, 1, 1, 0, 1, 0, 0)  # offering ore, asking wood+sheep

    return game


SCENARIO_TRADE_EVALUATION = UnderstandingScenario(
    name="trade_evaluation",
    category="TRADE",
    description=(
        "Tests: resource math before/after trade, assessing whether a trade "
        "helps progress toward a City (2 Wheat + 3 Ore)."
    ),
    setup=_setup_trade_evaluation,
    perspective_color=Color.RED,
    system_prompt=(
        "You are a Catan trade advisor. "
        "Show explicit resource counts before and after the trade. "
        "Answer in 3-5 sentences."
    ),
    question=(
        "There is an active trade offer: BLUE offers you 1 Ore in exchange for "
        "1 Sheep and 1 Wood. Analyze your resources before and after this trade, "
        "and evaluate whether accepting helps you progress toward building a City "
        "(which costs 2 Wheat and 3 Ore)."
    ),
)


# ---------------------------------------------------------------------------
# TR-2: Defensive Resource Holding
# ---------------------------------------------------------------------------

def _setup_defensive_trade() -> Game:
    """
    BLUE has 9 visible VP — very close to winning.
    BLUE offers a generous trade (3 Wood + 2 Brick for 1 Wheat).
    RED has 2 Wheat. Giving Wheat to BLUE when they're about to win is risky.
    """
    game = _make_base_game()
    state = game.state
    blue_key = player_key(state, Color.BLUE)

    # Set BLUE to 9 visible VP (near winning)
    state.player_state[f"{blue_key}_VICTORY_POINTS"] = 9
    state.player_state[f"{blue_key}_ACTUAL_VICTORY_POINTS"] = 9

    # Give RED 2 wheat
    player_freqdeck_add(state, Color.RED, [0, 0, 0, 2, 0])

    # Set up trade: BLUE offers 3 Wood + 2 Brick for 1 Wheat
    state.is_resolving_trade = True
    state.current_trade = (3, 2, 0, 0, 0, 0, 0, 0, 1, 0)  # offering wood+brick, asking wheat

    return game


SCENARIO_DEFENSIVE_TRADE = UnderstandingScenario(
    name="defensive_resource_holding",
    category="TRADE",
    description=(
        "Tests: recognising that giving resources to a player at 9 VP "
        "is dangerous regardless of how generous the trade looks."
    ),
    setup=_setup_defensive_trade,
    perspective_color=Color.RED,
    system_prompt=(
        "You are a Catan strategy advisor. "
        "Consider both the raw resource value AND the strategic implications. "
        "Answer in 2-4 sentences."
    ),
    question=(
        "There is an active trade offer from BLUE. "
        "Evaluate this trade from a defensive perspective. "
        "Should you accept it? Why or why not?"
    ),
)


# ═══════════════════════════════════════════════════════════════════════════
# CATEGORY: BOARD_STATE — Spatial / Topological Reasoning
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# BS-1: Initial Placement Synergy
# ---------------------------------------------------------------------------

def _setup_initial_placement() -> Game:
    """
    Present an initial build phase where RED must choose between two
    intersections with different resource profiles.

    We use a fresh game (no initial placements yet).
    Board seed 42 topology:
      Node 2: touches (Brick-8, Wood-8, Wheat-6) → good for expansion (wood/brick)
      Node 9: touches (Wood-8, Ore-10, Wheat-6)  → good for City/DevCard (ore/wheat)
    """
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players, seed=42)
    # Don't build initial placements — we want the initial placement phase
    return game


SCENARIO_INITIAL_PLACEMENT = UnderstandingScenario(
    name="initial_placement_synergy",
    category="BOARD_STATE",
    description=(
        "Tests: understanding resource synergy at settlement locations. "
        "Node 2 (Brick-8, Wood-8, Wheat-6) for expansion vs "
        "Node 9 (Wood-8, Ore-10, Wheat-6) for City/DevCard strategy."
    ),
    setup=_setup_initial_placement,
    perspective_color=Color.RED,
    system_prompt=(
        "You are evaluating initial settlement placement in Catan. "
        "Consider resource synergy, probability, and long-term strategy. "
        "Answer in 3-5 sentences."
    ),
    question=(
        "You are choosing your initial settlement location. "
        "Compare Node 2 and Node 9 by looking at their adjacent resource hexes "
        "in the board state. Which node better supports a City and Development Card "
        "strategy, and which better supports early road and settlement expansion? "
        "Which would you choose?"
    ),
)


# ---------------------------------------------------------------------------
# BS-2: Distance Rule / Blocking
# ---------------------------------------------------------------------------

def _setup_distance_rule() -> Game:
    """
    After initial placements, node 1 is adjacent to RED's settlement at node 0
    and node 2. Node 1 should NOT appear in buildable_node_ids because of the
    distance rule (cannot build within 1 edge of existing settlement).

    Node 1's neighbors: 0, 2, 6 — nodes 0 and 2 have RED settlements.
    """
    game = _make_base_game()
    return game


SCENARIO_DISTANCE_RULE = UnderstandingScenario(
    name="distance_rule_blocking",
    category="BOARD_STATE",
    description=(
        "Tests: understanding why certain nodes are not buildable (distance rule). "
        "Node 1 is between RED's settlements at nodes 0 and 2."
    ),
    setup=_setup_distance_rule,
    perspective_color=Color.RED,
    system_prompt=(
        "You are a Catan rules expert. "
        "Explain the distance rule clearly, referencing specific node IDs. "
        "Answer in 2-3 sentences."
    ),
    question=(
        "Node 1 is directly connected to both Node 0 and Node 2, where you have "
        "settlements. Looking at the buildable settlement nodes in the board state, "
        "explain why Node 1 does not appear as a valid building location."
    ),
)


# ---------------------------------------------------------------------------
# BS-3: Cutting Off Longest Road
# ---------------------------------------------------------------------------

def _setup_road_cutting() -> Game:
    """
    BLUE has roads (24,25), (25,26), (26,27).
    RED builds a settlement on node 26 (blocking BLUE's road).
    This breaks BLUE's continuous road — it's now split into
    (24,25)-(25,26) on one side and (26,27) on the other,
    but RED's settlement on 26 breaks the continuity.
    """
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players, seed=42)

    # Place BLUE with roads along 24->25->26->27
    # RED at nodes far from BLUE initially, then we manually place on node 26
    build_initial_placements(
        game,
        p0_actions=[0, (0, 1), 2, (1, 2)],
        p1_actions=[24, (24, 25), 27, (26, 27)],  # BLUE: 24 and 27 settlements
    )
    advance_to_play_turn(game)

    state = game.state
    red_key = player_key(state, Color.RED)

    # Give RED resources for a settlement and build on node 26
    # (node 26 was BLUE's settlement in default, but we changed the placement above)
    # Actually with the above placements, BLUE has settlements at 24 and 27.
    # BLUE has roads: (24,25), (26,27)
    # We need road (25,26) too for BLUE — let's add it
    blue_key = player_key(state, Color.BLUE)
    state.board.build_road(Color.BLUE, (25, 26))
    state.player_state[f"{blue_key}_ROADS_AVAILABLE"] -= 1
    state.buildings_by_color[Color.BLUE]["ROAD"].append((25, 26))

    # Now BLUE has roads: (24,25), (25,26), (26,27) — road length 3
    # RED needs to build a settlement at node 25 or 26 to cut the road.
    # Node 25 touches only sheep-12 tile. Let's put RED on node 9 first via road.
    # Actually, let's just directly place RED's settlement on a node that cuts BLUE's road.
    # Node 25: neighbors are 24, 26, and 54 — it's on BLUE's road path.
    # But node 25 is adjacent to BLUE's settlement at 24 (distance rule).
    # Node 26: neighbors are 25, 27, 57 — adjacent to BLUE's settlement at 27.
    # Neither works with distance rule. Let's rethink.

    # With settlements at 24 and 27, the road (24,25)-(25,26)-(26,27) connects them.
    # An enemy settlement on 25 or 26 would break the road.
    # But distance rule prevents building adjacent to existing settlements.
    # Node 25: adjacent to 24 (BLUE) → blocked.
    # Node 26: adjacent to 27 (BLUE) → blocked.

    # Let's instead: place BLUE's settlements at 24 and 28, roads at (24,25)(25,26)(26,27)(27,28).
    # Then RED can settle on node 26 (not adjacent to 24 or 28 — node 26 neighbors: 25,27,57)
    # Node 26 adjacent to 27, node 27 adjacent to 28. But 26 is not adjacent to 24 or 28. Check:
    # Node 26 neighbors: 25, 27, 57. Not adjacent to 24 or 28.
    # But we also need to ensure 26 is not adjacent to RED's own settlements at 0, 2.
    # Node 26 neighbors: 25, 27, 57. None of these are 0 or 2. Good.
    # HOWEVER, RED needs road connectivity to build there (not initial phase).

    # This scenario is getting complex for the distance rule. Let's simplify:
    # Just describe the situation in the state and ask the question. The board edges
    # and buildings are all in the formatted state.

    return game


SCENARIO_ROAD_CUTTING = UnderstandingScenario(
    name="cutting_off_longest_road",
    category="BOARD_STATE",
    description=(
        "Tests: understanding that an enemy settlement on a road path breaks "
        "road continuity."
    ),
    setup=_setup_road_cutting,
    perspective_color=Color.RED,
    system_prompt=(
        "You are a Catan rules expert. "
        "Answer precisely about road continuity rules. "
        "Answer in 2-3 sentences."
    ),
    question=(
        "BLUE has roads on edges (24,25), (25,26), and (26,27). "
        "If you were to build a settlement on node 26 (hypothetically), "
        "would BLUE's road still count as a continuous road of length 3? "
        "Explain the rule about enemy settlements breaking road continuity."
    ),
)


# ---------------------------------------------------------------------------
# BS-4: Robber Placement — Hex Evaluation
# ---------------------------------------------------------------------------

def _setup_robber_hex_eval() -> Game:
    """
    BLUE (leader) has a city at node 24 which touches:
      - Tile 7: Sheep-12 (probability 1/36)
      - Tile 18: Wood-6 (probability 5/36)
    And a settlement at node 26 touching:
      - Tile 7: Sheep-12 (probability 1/36)

    The best robber placement is on tile 18 (Wood-6) since 6 is the
    second most probable roll (5/36), much better than 12 (1/36).

    We'll also give BLUE another settlement at node 9 touching:
      - Tile 1: Wood-8 (probability 5/36)
      - Tile 2: Wheat-6 (probability 5/36)
      - Tile 8: Ore-10 (probability 3/36)

    The clearly best hex to rob is one touching BLUE's city (doubled production)
    with high probability: tile 18 (Wood-6, 5/36) at the city on node 24.
    """
    game = _make_base_game()
    state = game.state
    blue_key = player_key(state, Color.BLUE)

    # Upgrade BLUE's settlement at node 24 to a city
    state.board.build_city(Color.BLUE, 24)
    state.buildings_by_color[Color.BLUE]["SETTLEMENT"].remove(24)
    state.buildings_by_color[Color.BLUE]["CITY"].append(24)
    state.player_state[f"{blue_key}_SETTLEMENTS_AVAILABLE"] += 1
    state.player_state[f"{blue_key}_CITIES_AVAILABLE"] -= 1
    state.player_state[f"{blue_key}_VICTORY_POINTS"] += 1
    state.player_state[f"{blue_key}_ACTUAL_VICTORY_POINTS"] += 1

    # Set BLUE as leader
    state.player_state[f"{blue_key}_VICTORY_POINTS"] = 6
    state.player_state[f"{blue_key}_ACTUAL_VICTORY_POINTS"] = 6

    return game


SCENARIO_ROBBER_HEX_EVAL = UnderstandingScenario(
    name="robber_hex_evaluation",
    category="BOARD_STATE",
    description=(
        "Tests: understanding dice probability to choose optimal robber placement. "
        "BLUE has a city touching a high-probability hex."
    ),
    setup=_setup_robber_hex_eval,
    perspective_color=Color.RED,
    system_prompt=(
        "You are a Catan strategy advisor analysing robber placement. "
        "Consider dice probability and building types (city = 2x production). "
        "Answer in 3-5 sentences."
    ),
    question=(
        "You need to place the robber. BLUE is the leader. "
        "Looking at the hexes adjacent to BLUE's buildings, "
        "which specific hex should you place the robber on to maximise "
        "damage to BLUE's resource income? Reference the dice numbers and probabilities."
    ),
)


# ---------------------------------------------------------------------------
# BS-5: City Upgrade Targeting
# ---------------------------------------------------------------------------

def _setup_city_upgrade() -> Game:
    """
    RED has two settlements:
      Node 0: touches (Brick-8, Brick-3, Ore-11)
      Node 2: touches (Brick-8, Wood-8, Wheat-6)

    RED has resources for one city upgrade (2 Wheat + 3 Ore).
    Node 2 (Brick-8, Wood-8, Wheat-6) produces higher-value resources
    at better probability. Upgrading = doubled income from those hexes.
    """
    game = _make_base_game()
    state = game.state

    # Give RED resources for a city: 3 ore, 2 wheat
    player_freqdeck_add(state, Color.RED, [0, 0, 0, 2, 3])

    return game


SCENARIO_CITY_UPGRADE = UnderstandingScenario(
    name="city_upgrade_targeting",
    category="BOARD_STATE",
    description=(
        "Tests: evaluating which settlement to upgrade based on adjacent hex "
        "resources and probabilities."
    ),
    setup=_setup_city_upgrade,
    perspective_color=Color.RED,
    system_prompt=(
        "You are a Catan strategy advisor. "
        "Consider resource diversity, probability, and strategic value. "
        "Answer in 3-5 sentences."
    ),
    question=(
        "You have the resources to upgrade one of your settlements to a city. "
        "Compare the resource hexes adjacent to each of your settlements. "
        "Which settlement should you upgrade, and why?"
    ),
)


# ---------------------------------------------------------------------------
# BS-6: Trapped Settlement
# ---------------------------------------------------------------------------

def _setup_trapped_settlement() -> Game:
    """
    RED has a settlement at node 2.
    Node 2's edges: (1,2), (2,3), (2,9).
    RED owns road (1,2). BLUE owns roads on (2,3) and (2,9).
    RED cannot build any NEW road from node 2 since all edges are taken.

    We directly insert BLUE roads into the board data (bypassing connectivity
    validation) to simulate a mid-game state where BLUE expanded into this area.
    """
    game = _make_base_game()
    state = game.state
    blue_key = player_key(state, Color.BLUE)

    # Directly insert BLUE roads on edges adjacent to RED's node 2
    # (bypass build_road validation which requires connectivity to BLUE's network)
    for edge in [(2, 3), (2, 9)]:
        state.board.roads[edge] = Color.BLUE
        state.board.roads[(edge[1], edge[0])] = Color.BLUE
        state.player_state[f"{blue_key}_ROADS_AVAILABLE"] -= 1
        state.buildings_by_color[Color.BLUE]["ROAD"].append(edge)

    # Clear buildable edges cache so it recalculates
    state.board.buildable_edges_cache = {}

    return game


SCENARIO_TRAPPED_SETTLEMENT = UnderstandingScenario(
    name="trapped_settlement",
    category="BOARD_STATE",
    description=(
        "Tests: recognising when a settlement has no buildable road options "
        "due to opponent roads on all adjacent edges."
    ),
    setup=_setup_trapped_settlement,
    perspective_color=Color.RED,
    system_prompt=(
        "You are analysing a Catan board position. "
        "Focus on road building constraints from a specific settlement. "
        "Answer in 2-3 sentences."
    ),
    question=(
        "Look at your settlement on Node 2. Examine the edges connected to Node 2 "
        "and who owns roads on them. Can you build a new road from this settlement? "
        "Explain based on the road data in the board state."
    ),
)


# ---------------------------------------------------------------------------
# BS-7: Port Value / Building Direction
# ---------------------------------------------------------------------------

def _setup_port_direction() -> Game:
    """
    RED has settlements producing lots of Sheep:
      Node 0: (Brick-8, Brick-3, Ore-11)
      Node 2: (Brick-8, Wood-8, Wheat-6)

    We'll re-place RED to sheep-heavy nodes instead.
    Node 32: (Wheat-9, Wood-11, Sheep port nearby at 32,33)
    Node 11: (Wheat-6, Wheat-9, Wood-11)

    Actually, let's use a simpler approach: place RED at nodes that produce sheep,
    and have two expansion directions — one toward a 2:1 Sheep port, another elsewhere.

    From seed-42: 2:1 Sheep port is at nodes 32, 33.
    Node 11 touches: (Wheat-6, Wheat-9, Wood-11) — not sheep. 
    
    Let's place RED at nodes that give sheep production.
    Node 26 touches: Sheep-12 only. Node 25 touches: Sheep-12 only.
    
    Better approach: use default placements and just ask about expansion
    direction given the buildable edges and nearby ports.
    """
    game = _make_base_game()
    state = game.state

    # Give RED resources for roads
    player_freqdeck_add(state, Color.RED, [3, 3, 0, 0, 0])

    return game


SCENARIO_PORT_DIRECTION = UnderstandingScenario(
    name="port_building_direction",
    category="BOARD_STATE",
    description=(
        "Tests: evaluating expansion direction considering ports "
        "and current resource production."
    ),
    setup=_setup_port_direction,
    perspective_color=Color.RED,
    system_prompt=(
        "You are a Catan strategy advisor. "
        "Consider port locations, current production, and expansion cost. "
        "Answer in 3-5 sentences."
    ),
    question=(
        "You have resources to build roads. Looking at your buildable edges, "
        "the port locations, and your current resource production, "
        "where should you expand your road network next? "
        "Explain your reasoning."
    ),
)


# ---------------------------------------------------------------------------
# BS-8: Coordinate Adjacency (Catanatron specific)
# ---------------------------------------------------------------------------

def _setup_coordinate_adjacency() -> Game:
    """
    RED has a settlement at node 0. 
    Node 0 connects to edges: (0,1), (0,5), (0,20).
    RED already has road on (0,1).
    Buildable edges from node 0: (0,5) and (0,20).
    """
    game = _make_base_game()
    return game


SCENARIO_COORDINATE_ADJACENCY = UnderstandingScenario(
    name="coordinate_adjacency",
    category="BOARD_STATE",
    description=(
        "Tests: reading Catanatron topology data to identify valid edge IDs "
        "for road placement from a specific node."
    ),
    setup=_setup_coordinate_adjacency,
    perspective_color=Color.RED,
    system_prompt=(
        "You are analysing Catanatron board topology. "
        "Be precise about node and edge IDs. "
        "Answer concisely."
    ),
    question=(
        "Your settlement is on Node 0. Looking at the board state data "
        "(tiles, roads, and buildable edges), list the edges where you can "
        "build a road from Node 0. Explain how you identified them."
    ),
)


# ---------------------------------------------------------------------------
# BS-9: Connecting Roads (Longest Road specific edge)
# ---------------------------------------------------------------------------

def _setup_connecting_roads() -> Game:
    """
    RED has two separate road segments:
    Segment 1: (0, 1), (1, 2)
    Segment 2: (9, 10), (10, 11)
    Missing edge to connect them: (2, 9)

    If RED builds on (2, 9), length becomes 5, giving them Longest Road.
    """
    game = _make_base_game()
    state = game.state
    red_key = player_key(state, Color.RED)

    # RED already has (0, 1) and (1, 2).
    # Give RED roads (9, 10) and (10, 11). We bypass build_road validation.
    state.board.roads[(9, 10)] = Color.RED
    state.board.roads[(10, 9)] = Color.RED
    state.board.roads[(10, 11)] = Color.RED
    state.board.roads[(11, 10)] = Color.RED
    
    state.player_state[f"{red_key}_ROADS_AVAILABLE"] -= 2
    state.buildings_by_color[Color.RED]["ROAD"].extend([(9, 10), (10, 11)])

    # Give RED resources for 1 road
    player_freqdeck_add(state, Color.RED, [1, 1, 0, 0, 0])

    state.board.buildable_edges_cache = {}
    return game


SCENARIO_CONNECTING_ROADS = UnderstandingScenario(
    name="connecting_roads_longest_road",
    category="BOARD_STATE",
    description=(
        "Tests: identifying a single edge that connects two separate road networks "
        "to achieve Longest Road."
    ),
    setup=_setup_connecting_roads,
    perspective_color=Color.RED,
    system_prompt=(
        "You are a Catan spatial reasoning expert. "
        "Look closely at the edges in the player's road network. "
        "Answer in 2-3 sentences."
    ),
    question=(
        "You currently have two separate road segments. Looking at your road "
        "coordinates and the available buildable edges, is there a single edge you "
        "can build a road on to connect these two segments? What specific edge ID "
        "is it, and what would your total connected road length become?"
    ),
)


# ---------------------------------------------------------------------------
# BS-10: Spatial Blocking of Opponent
# ---------------------------------------------------------------------------

def _setup_spatial_blocking() -> Game:
    """
    BLUE has settlements at 32 (Port) and 37 (Inland).
    BLUE roads: (32, 33), (37, 36), (36, 35).
    Path is: 32 - 33 - 34 - 35 - 36 - 37
    Missing edges for BLUE: (33, 34) and (34, 35).
    RED has a settlement at 13, and a road on (13, 34).
    RED can build a road on (34, 33) or (34, 35) to permanently block BLUE.
    """
    game = _make_base_game()
    state = game.state
    red_key = player_key(state, Color.RED)
    blue_key = player_key(state, Color.BLUE)

    # Reset BLUE's standard buildings to avoid clutter
    state.buildings_by_color[Color.BLUE]["SETTLEMENT"] = []
    state.buildings_by_color[Color.BLUE]["ROAD"] = []
    
    # Remove old BLUE buildings from board dict
    for node in [24, 26]:
        if node in state.board.buildings:
            del state.board.buildings[node]
    for edge in [(24, 25), (25, 26)]:
        if edge in state.board.roads:
            del state.board.roads[edge]
            del state.board.roads[(edge[1], edge[0])]

    # Add new BLUE settlements
    state.board.buildings[32] = (Color.BLUE, "SETTLEMENT")
    state.board.buildings[37] = (Color.BLUE, "SETTLEMENT")
    state.buildings_by_color[Color.BLUE]["SETTLEMENT"].extend([32, 37])
    
    # Add new BLUE roads
    for edge in [(32, 33), (37, 36), (36, 35)]:
        state.board.roads[edge] = Color.BLUE
        state.board.roads[(edge[1], edge[0])] = Color.BLUE
        state.buildings_by_color[Color.BLUE]["ROAD"].append(edge)

    # Add RED settlement at 13
    state.board.buildings[13] = (Color.RED, "SETTLEMENT")
    state.buildings_by_color[Color.RED]["SETTLEMENT"].append(13)

    # Add RED road at (13, 34)
    state.board.roads[(13, 34)] = Color.RED
    state.board.roads[(34, 13)] = Color.RED
    state.buildings_by_color[Color.RED]["ROAD"].append((13, 34))

    # Give RED resources for 1 road
    player_freqdeck_add(state, Color.RED, [1, 1, 0, 0, 0])

    state.board.buildable_edges_cache = {}
    return game


SCENARIO_SPATIAL_BLOCKING = UnderstandingScenario(
    name="spatial_blocking_opponent",
    category="BOARD_STATE",
    description=(
        "Tests: using a road to cut off an opponent from connecting their "
        "two settlements."
    ),
    setup=_setup_spatial_blocking,
    perspective_color=Color.RED,
    system_prompt=(
        "You are a Catan spatial reasoning expert. "
        "Look closely at the edges, the opponent's road network, and your network. "
        "Answer in 2-4 sentences."
    ),
    question=(
        "BLUE has settlements at Node 32 and Node 37, and is trying to connect "
        "them by building roads towards Node 34. You (RED) have a settlement at "
        "Node 13 and a road on edge (13, 34). Look at your buildable edges. "
        "What specific action can you take right now to permanently cut off BLUE "
        "from connecting their two settlements? Analyze the edge IDs to explain your move."
    ),
)


SCENARIO_CONNECTING_ROADS_UNHINTED = UnderstandingScenario(
    name="connecting_roads_unhinted",
    category="BOARD_STATE",
    description=(
        "Same setup as connecting_roads_longest_road but with absolutely no hints "
        "about connecting segments or Longest Road."
    ),
    setup=_setup_connecting_roads,
    perspective_color=Color.RED,
    system_prompt=(
        "You are a Catan strategy advisor. "
        "Analyze the player's road network. "
        "Answer concisely."
    ),
    question=(
        "Looking at your current road network and available buildable edges, "
        "where specifically would you build your next road? Explain your strategic reasoning."
    ),
)


SCENARIO_SPATIAL_BLOCKING_UNHINTED = UnderstandingScenario(
    name="spatial_blocking_unhinted",
    category="BOARD_STATE",
    description=(
        "Same as spatial_blocking_opponent but with no hints about blocking BLUE."
    ),
    setup=_setup_spatial_blocking,
    perspective_color=Color.RED,
    system_prompt=(
        "You are a Catan strategy advisor. "
        "Analyze the board state, including the opponent's settlements and roads. "
        "Answer concisely."
    ),
    question=(
        "BLUE has settlements at Node 32 and Node 37. You (RED) have a settlement "
        "at Node 13 and a road on edge (13, 34). Where should you prioritize building "
        "your next road? Explain the strategic value of this placement."
    ),
)


# ═══════════════════════════════════════════════════════════════════════════
# CATEGORY: JOINT — Combined Board + Game State Reasoning
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# JT-1: Robber Targeting (Strategic Priorities)
# ---------------------------------------------------------------------------

def _setup_robber_targeting() -> Game:
    """
    RED rolled a 7 and must move the robber.
    BLUE has 8 visible VP and 4 resource cards.
    
    In a 2-player game there's only one opponent, so the question
    asks about the trade-off between VP-based targeting and resource-based targeting.
    We'll give BLUE high VP + moderate hand to make this interesting.
    """
    game = _make_base_game()
    state = game.state
    blue_key = player_key(state, Color.BLUE)

    # Set BLUE to 8 VP with moderate hand
    state.player_state[f"{blue_key}_VICTORY_POINTS"] = 8
    state.player_state[f"{blue_key}_ACTUAL_VICTORY_POINTS"] = 8
    player_freqdeck_add(state, Color.BLUE, [1, 1, 1, 1, 0])  # 4 resource cards

    return game


SCENARIO_ROBBER_TARGETING = UnderstandingScenario(
    name="robber_targeting",
    category="JOINT",
    description=(
        "Tests: balancing VP-threat targeting vs resource-wealth targeting "
        "when deciding robber placement."
    ),
    setup=_setup_robber_targeting,
    perspective_color=Color.RED,
    system_prompt=(
        "You are a Catan strategy advisor. "
        "Consider both the defensive (blocking leader) and offensive (stealing resources) "
        "aspects of robber placement. Answer in 3-5 sentences."
    ),
    question=(
        "You rolled a 7 and must move the robber and steal from an opponent. "
        "Looking at your opponent's VP count, resource card count, and their "
        "settlement/city positions on the board, decide where to place the robber "
        "and explain the strategic trade-offs."
    ),
)


# ---------------------------------------------------------------------------
# JT-2: Longest Road Race Inference
# ---------------------------------------------------------------------------

def _setup_longest_road_race() -> Game:
    """
    RED holds Longest Road with length 5.
    BLUE has road length 4 and just acquired 2 Wood + 2 Brick
    (enough for 2 roads, which would give them length 6 → overtake RED).

    We set this up by giving RED 5 roads and BLUE 4 roads + resources.
    """
    game = _make_base_game()
    state = game.state
    red_key = player_key(state, Color.RED)
    blue_key = player_key(state, Color.BLUE)

    # Build extra roads for RED: (0,5), (5,4), (4,3) → total 5 roads
    state.board.build_road(Color.RED, (0, 5))
    state.player_state[f"{red_key}_ROADS_AVAILABLE"] -= 1
    state.buildings_by_color[Color.RED]["ROAD"].append((0, 5))

    state.board.build_road(Color.RED, (4, 5))
    state.player_state[f"{red_key}_ROADS_AVAILABLE"] -= 1
    state.buildings_by_color[Color.RED]["ROAD"].append((4, 5))

    state.board.build_road(Color.RED, (3, 4))
    state.player_state[f"{red_key}_ROADS_AVAILABLE"] -= 1
    state.buildings_by_color[Color.RED]["ROAD"].append((3, 4))

    # Set RED's Longest Road
    state.player_state[f"{red_key}_HAS_ROAD"] = True
    state.player_state[f"{red_key}_VICTORY_POINTS"] += 2
    state.player_state[f"{red_key}_ACTUAL_VICTORY_POINTS"] += 2
    state.player_state[f"{red_key}_LONGEST_ROAD_LENGTH"] = 5

    # Build extra roads for BLUE: (26,27), (27,28) → total 4 roads
    state.board.build_road(Color.BLUE, (26, 27))
    state.player_state[f"{blue_key}_ROADS_AVAILABLE"] -= 1
    state.buildings_by_color[Color.BLUE]["ROAD"].append((26, 27))

    state.board.build_road(Color.BLUE, (27, 28))
    state.player_state[f"{blue_key}_ROADS_AVAILABLE"] -= 1
    state.buildings_by_color[Color.BLUE]["ROAD"].append((27, 28))

    state.player_state[f"{blue_key}_LONGEST_ROAD_LENGTH"] = 4

    # Give BLUE exactly 2 Wood + 2 Brick (enough for 2 roads)
    player_freqdeck_add(state, Color.BLUE, [2, 2, 0, 0, 0])

    return game


SCENARIO_LONGEST_ROAD_RACE = UnderstandingScenario(
    name="longest_road_race",
    category="JOINT",
    description=(
        "Tests: predicting opponent's next move from their resources and board state. "
        "BLUE has road length 4 and resources for 2 more roads."
    ),
    setup=_setup_longest_road_race,
    perspective_color=Color.RED,
    system_prompt=(
        "You are a Catan strategy advisor. "
        "Analyse your opponent's likely next move based on their resources, "
        "board position, and the current game situation. "
        "Answer in 3-5 sentences."
    ),
    question=(
        "Look at BLUE's current board position, road length, and resource cards. "
        "What do you predict BLUE's next strategic move will be, and how might "
        "it affect the game standings?"
    ),
)


# ---------------------------------------------------------------------------
# Scenario for existing backward compatibility: Resource Awareness
# ---------------------------------------------------------------------------

def _setup_scenario_resource_awareness() -> Game:
    """
    RED has exactly the resources needed to build one settlement
    (1 wood, 1 brick, 1 sheep, 1 wheat) and nothing else.
    """
    game = _make_base_game()
    player_freqdeck_add(game.state, Color.RED, [1, 1, 1, 1, 0])
    return game


SCENARIO_RESOURCE_AWARENESS = UnderstandingScenario(
    name="resource_awareness",
    category="GAME_STATE",
    description=(
        "Tests: knowledge of settlement building cost + positional reasoning. "
        "RED holds exactly 1 wood, 1 brick, 1 sheep, 1 wheat."
    ),
    setup=_setup_scenario_resource_awareness,
    perspective_color=Color.RED,
    system_prompt=(
        "You are evaluating a Catan player's game state. "
        "Answer concisely in 2–4 sentences. "
        "Focus on what the player CAN do right now and whether it is strategically sound."
    ),
    question=(
        "RED has exactly one settlement's worth of resources "
        "(1 wood, 1 brick, 1 sheep, 1 wheat). "
        "Based on RED's current board position and the available buildable nodes shown "
        "in the state, should RED build a settlement immediately? "
        "Identify the most strategically valuable buildable node, if any, and explain why."
    ),
)


# ---------------------------------------------------------------------------
# Scenario for existing backward compatibility: Production Odds
# ---------------------------------------------------------------------------

def _setup_scenario_production_odds() -> Game:
    """
    RED is placed on specific nodes; question is about production potential.
    """
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players, seed=42)
    build_initial_placements(
        game,
        p0_actions=[8, (8, 9), 45, (45, 46)],
        p1_actions=[24, (24, 25), 26, (25, 26)],
    )
    advance_to_play_turn(game)
    player_freqdeck_add(game.state, Color.RED, [0, 0, 0, 0, 0])
    return game


SCENARIO_PRODUCTION_ODDS = UnderstandingScenario(
    name="production_odds",
    category="BOARD_STATE",
    description=(
        "Tests: hex probability awareness and port recognition. "
        "RED has 0 resources; question asks purely about expected income and trading."
    ),
    setup=_setup_scenario_production_odds,
    perspective_color=Color.RED,
    system_prompt=(
        "You are analysing a Catan board position. "
        "Answer in 3–5 sentences. "
        "Be specific about roll numbers, resource types, and expected frequency."
    ),
    question=(
        "Based solely on RED's current settlement locations shown in the game state, "
        "rank RED's expected resource income from most to least frequent "
        "(reference the dice roll numbers on adjacent hexes). "
        "Does RED have access to any ports, and if so, how does that affect their "
        "trading strategy in the early game?"
    ),
)


# ---------------------------------------------------------------------------
# Registered scenarios by category
# ---------------------------------------------------------------------------

SCENARIOS: list[UnderstandingScenario] = [
    # GAME_STATE
    SCENARIO_RESOURCE_AWARENESS,
    SCENARIO_VP_CALCULATION,
    SCENARIO_DEV_CARD_LOGIC,
    SCENARIO_LARGEST_ARMY,
    # TRADE
    SCENARIO_TRADE_EVALUATION,
    SCENARIO_DEFENSIVE_TRADE,
    # BOARD_STATE
    SCENARIO_PRODUCTION_ODDS,
    SCENARIO_INITIAL_PLACEMENT,
    SCENARIO_DISTANCE_RULE,
    SCENARIO_ROAD_CUTTING,
    SCENARIO_ROBBER_HEX_EVAL,
    SCENARIO_CITY_UPGRADE,
    SCENARIO_TRAPPED_SETTLEMENT,
    SCENARIO_PORT_DIRECTION,
    SCENARIO_COORDINATE_ADJACENCY,
    SCENARIO_CONNECTING_ROADS,
    SCENARIO_CONNECTING_ROADS_UNHINTED,
    SCENARIO_SPATIAL_BLOCKING,
    SCENARIO_SPATIAL_BLOCKING_UNHINTED,
    # JOINT
    SCENARIO_ROBBER_TARGETING,
    SCENARIO_LONGEST_ROAD_RACE,
]


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_scenario(scenario: UnderstandingScenario, model: ModelInput = None) -> str:
    """
    Execute a single understanding scenario and return the agent's answer.

    Mirrors the agent-call style used in BaseLLMPlayer (base.py).
    No tools are registered — this tests pure reasoning from the state JSON.
    """
    game = scenario.setup()

    # Format state identically to how the live player does it (base.py)
    state_dict = StateFormatter.format_full_state(game, scenario.perspective_color)
    state_json = json.dumps(state_dict, indent=2, default=str)

    # Build the user prompt: structured state block + open-ended question
    prompt_parts = [
        "=== CATAN GAME STATE ===",
        state_json,
        "=== END GAME STATE ===",
        "",
        "=== QUESTION ===",
        scenario.question,
        "=== END QUESTION ===",
    ]
    user_prompt = "\n".join(prompt_parts)

    # Create a fresh agent with the scenario's custom system prompt.
    resolved_model = create_model(model)
    agent: Agent[None, str] = Agent(
        resolved_model,
        system_prompt=scenario.system_prompt,
        output_type=str,
    )

    result = agent.run_sync(user_prompt)
    return result.output


# ---------------------------------------------------------------------------
# Results file helpers
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent / "understanding_results"


def _write_categorised_results(
    results: list[tuple[UnderstandingScenario, str]],
    model_label: str,
) -> Path:
    """
    Write all scenario results to a single timestamped file, organised by category.
    Returns the path written to.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = RESULTS_DIR / f"{timestamp}_understanding_results.txt"

    lines = [
        "=" * 72,
        "  LLM UNDERSTANDING EVALUATION RESULTS",
        "=" * 72,
        f"  Timestamp : {datetime.now().isoformat()}",
        f"  Model     : {model_label}",
        f"  Scenarios : {len(results)}",
        "=" * 72,
        "",
    ]

    # Group results by category
    by_category: dict[str, list[tuple[UnderstandingScenario, str]]] = {}
    for scenario, answer in results:
        by_category.setdefault(scenario.category, []).append((scenario, answer))

    # Write each category
    category_order = ["GAME_STATE", "TRADE", "BOARD_STATE", "JOINT"]
    for cat in category_order:
        if cat not in by_category:
            continue
        lines.append("─" * 72)
        lines.append(f"  CATEGORY: {cat}")
        lines.append("─" * 72)
        lines.append("")

        for scenario, answer in by_category[cat]:
            lines.append(f"  [{scenario.name}]")
            lines.append(f"  Testing: {scenario.description}")
            lines.append(f"  Perspective: {scenario.perspective_color.value}")
            lines.append("")
            lines.append("  QUESTION:")
            for qline in scenario.question.splitlines():
                lines.append(f"    {qline}")
            lines.append("")
            lines.append("  ANSWER:")
            for aline in answer.strip().splitlines():
                lines.append(f"    {aline}")
            lines.append("")
            lines.append("  " + "·" * 68)
            lines.append("")

    lines.append("=" * 72)
    lines.append("  END OF RESULTS")
    lines.append("=" * 72)

    content = "\n".join(lines)
    filename.write_text(content, encoding="utf-8")
    return filename


def _write_single_result(scenario: UnderstandingScenario, answer: str, model_label: str) -> Path:
    """
    Write a single scenario result (used by pytest individual tests).
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = RESULTS_DIR / f"{timestamp}_{scenario.name}.txt"
    content = "\n".join([
        f"Scenario : {scenario.name}",
        f"Category : {scenario.category}",
        f"Timestamp: {datetime.now().isoformat()}",
        f"Model    : {model_label}",
        f"Testing  : {scenario.description}",
        f"Perspective: {scenario.perspective_color.value}",
        "",
        "QUESTION:",
        scenario.question,
        "",
        "ANSWER:",
        answer.strip(),
        "",
    ])
    filename.write_text(content, encoding="utf-8")
    return filename


# ---------------------------------------------------------------------------
# Pytest tests (smoke: response is non-empty string)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "scenario",
    SCENARIOS,
    ids=[s.name for s in SCENARIOS],
)
def test_scenario(scenario: UnderstandingScenario):
    """
    Smoke test: agent produces a non-empty answer for each scenario.
    Run with CATAN_LLM_TEST_MODE=1 to use TestModel (no API calls).
    (Actual results generation should be run via python CLI, not pytest).
    """
    answer = run_scenario(scenario)
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0, "Expected a non-empty answer from the agent"
    print(f"\n\n{'─' * 60}")
    print(f"Scenario : {scenario.name} [{scenario.category}]")
    print(f"\nANSWER:\n{answer.strip()}")
    print(f"{'─' * 60}")


# ---------------------------------------------------------------------------
# CLI runner — runs all scenarios and writes one categorised output file
# ---------------------------------------------------------------------------

def _separator(char: str = "─", width: int = 72) -> str:
    return char * width


def main():
    """
    Run scenarios with real API calls, print answers to stdout,
    and save all results to one categorised file in understanding_results/.
    Pass scenario names as command-line arguments to run only those specific tests.
    """
    args = sys.argv[1:]
    
    scenarios_to_run = SCENARIOS
    if args:
        scenarios_to_run = [s for s in SCENARIOS if s.name in args]
        if not scenarios_to_run:
            print(f"Error: No scenarios match the provided arguments: {args}")
            print("Available scenarios:", [s.name for s in SCENARIOS])
            return

    model = os.environ.get("CATAN_LLM_MODEL") or None
    model_label = str(create_model(model))
    RESULTS_DIR.mkdir(exist_ok=True)

    print(_separator("═"))
    print("  LLM UNDERSTANDING HARNESS")
    print(_separator("═"))
    print(f"  Model    : {model_label}")
    print(f"  Scenarios: {len(scenarios_to_run)} (of {len(SCENARIOS)} total)")
    print(f"  Output   : {RESULTS_DIR}/")
    print(_separator("═"))

    all_results: list[tuple[UnderstandingScenario, str]] = []

    for i, scenario in enumerate(scenarios_to_run, 1):
        print(f"\n[{i}/{len(scenarios_to_run)}]  [{scenario.category}] {scenario.name.upper()}")
        print(_separator())
        print(f"  Testing : {scenario.description}")
        print(f"  Perspective: {scenario.perspective_color.value}")
        print(_separator("·"))
        print(f"  QUESTION:\n  {scenario.question}")
        print(_separator("·"))
        print("  Running scenario...", end="", flush=True)

        try:
            answer = run_scenario(scenario, model=model)
            all_results.append((scenario, answer))
            print(f"\r  ANSWER:")
            print()
            for line in answer.strip().splitlines():
                print(f"    {line}")
        except Exception as exc:
            error_msg = f"ERROR: {exc}"
            all_results.append((scenario, error_msg))
            print(f"\r  {error_msg}")

        print(_separator())

    # Write all results to one categorised file
    out_path = _write_categorised_results(all_results, model_label)
    print(f"\n{'═' * 72}")
    print(f"  All results saved to: {out_path}")
    print(f"{'═' * 72}")


if __name__ == "__main__":
    main()
