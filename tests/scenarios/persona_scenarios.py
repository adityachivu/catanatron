"""
Persona-based understanding scenarios.

These scenarios test the LLM's understanding of Catan game state when operating
under a specific persona. The persona's system prompt (and memory config, if any)
is loaded via load_persona() — the `system_prompt` field on the scenario is used
only as a fallback when no persona is set.

The `category` field carries the persona name (e.g. "aria_d_diplomat") so that
results files group persona scenarios by persona rather than by game-mechanic type.

Adding new scenarios
--------------------
1. Write a `setup_*()` function that returns a fully-configured `Game`.
2. Create an `UnderstandingScenario(...)` with `persona="<name>"` and
   `category="<persona_name>"` set.
3. Append it to `SCENARIOS`.

Available personas (catanatron/players/llm/personas/):
  default, aggressive, friendly, aria_d_diplomat, default_with_memory,
  cassio_a_charmer, magnus_d_game_theorist, brand_a_wall
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CATANATRON_SRC = _PROJECT_ROOT / "catanatron"
for _p in [str(_PROJECT_ROOT), str(_CATANATRON_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from catanatron.game import Game
from catanatron.models.enums import Action, ActionType, WOOD, BRICK, SHEEP, WHEAT, ORE
from catanatron.models.player import Color, SimplePlayer
from catanatron.state_functions import player_freqdeck_add, player_key

from tests.utils import advance_to_play_turn, build_initial_placements
from tests.scenarios import UnderstandingScenario, _make_base_game, _make_map_with_pinned_tiles


# ═══════════════════════════════════════════════════════════════════════════
# PERSONA: aria_d_diplomat
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# PA-1: Making the Alliance
# ---------------------------------------------------------------------------

def _setup_making_alliance() -> Game:
    """
    RED (VP 3) has settlements near: Wood-10, Wheat-6, Sheep-3.
    BLUE (VP 3) has settlements near: Brick-8, Wheat-6, Ore-4, Sheep-8.

    These tiles are pinned onto the board. After default initial placements
    (RED at nodes 0, 2 — BLUE at nodes 24, 26), a third settlement for each
    player is placed directly on a node adjacent to one of their production
    tiles (Sheep-3 for RED, Ore-4 for BLUE) using board state manipulation.

    Active trade offer: BLUE offers 1 Wheat in exchange for 3 Wood from RED.
    RED currently holds a Wood surplus from the Wood-10 tile.
    BLUE's verbal promise of future Brick trades is embedded in the question —
    it has no in-game representation.
    """
    pinned_tiles = [
        (WOOD, 10),   # RED's primary production
        (WHEAT, 6),   # shared high-value tile
        (SHEEP, 3),   # RED's low-probability sheep
        (BRICK, 8),   # BLUE's primary production
        (ORE, 4),     # BLUE's low-probability ore
        (SHEEP, 8),   # BLUE's secondary production
    ]

    catan_map = _make_map_with_pinned_tiles(pinned_tiles, seed=42)

    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players, seed=42, catan_map=catan_map)

    build_initial_placements(game)
    advance_to_play_turn(game)

    state = game.state
    red_key = player_key(state, Color.RED)
    blue_key = player_key(state, Color.BLUE)

    # Find a free node adjacent to Sheep-3 for RED's 3rd settlement
    occupied = set(state.board.buildings.keys())
    red_3rd = None
    for tile in catan_map.land_tiles.values():
        if tile.resource == SHEEP and tile.number == 3:
            for nid in tile.nodes.values():
                if nid not in occupied:
                    red_3rd = nid
                    occupied.add(nid)
                    break
        if red_3rd is not None:
            break

    # Find a free node adjacent to Ore-4 for BLUE's 3rd settlement
    blue_3rd = None
    for tile in catan_map.land_tiles.values():
        if tile.resource == ORE and tile.number == 4:
            for nid in tile.nodes.values():
                if nid not in occupied:
                    blue_3rd = nid
                    occupied.add(nid)
                    break
        if blue_3rd is not None:
            break

    # Place RED's 3rd settlement
    if red_3rd is not None:
        state.board.buildings[red_3rd] = (Color.RED, "SETTLEMENT")
        state.buildings_by_color[Color.RED]["SETTLEMENT"].append(red_3rd)
        state.player_state[f"{red_key}_SETTLEMENTS_AVAILABLE"] -= 1
        state.player_state[f"{red_key}_VICTORY_POINTS"] += 1
        state.player_state[f"{red_key}_ACTUAL_VICTORY_POINTS"] += 1

    # Place BLUE's 3rd settlement
    if blue_3rd is not None:
        state.board.buildings[blue_3rd] = (Color.BLUE, "SETTLEMENT")
        state.buildings_by_color[Color.BLUE]["SETTLEMENT"].append(blue_3rd)
        state.player_state[f"{blue_key}_SETTLEMENTS_AVAILABLE"] -= 1
        state.player_state[f"{blue_key}_VICTORY_POINTS"] += 1
        state.player_state[f"{blue_key}_ACTUAL_VICTORY_POINTS"] += 1

    state.board.buildable_edges_cache = {}

    # RED has a wood surplus from the Wood-10 tile
    player_freqdeck_add(state, Color.RED, [4, 0, 1, 1, 0])   # 4 wood, 1 sheep, 1 wheat

    # BLUE has some brick and wheat available
    player_freqdeck_add(state, Color.BLUE, [0, 2, 1, 2, 0])  # 2 brick, 1 sheep, 2 wheat

    # Active trade: BLUE offers 1 Wheat in exchange for 3 Wood from RED
    state.is_resolving_trade = True
    state.current_trade = (0, 0, 0, 1, 0, 3, 0, 0, 0, 0)

    return game


SCENARIO_MAKING_ALLIANCE = UnderstandingScenario(
    name="making_the_alliance",
    category="aria_d_diplomat",
    description=(
        "Tests: alliance reasoning under the aria_d_diplomat persona. "
        "BLUE offers wheat + verbal promise of future brick trades. "
        "Tests whether the persona reasons about trust, promises, and strategic cooperation."
    ),
    setup=_setup_making_alliance,
    perspective_color=Color.RED,
    persona="aria_d_diplomat",
    system_prompt="(overridden by aria_d_diplomat persona)",
    question=(
        "BLUE has offered you 1 Wheat in exchange for 3 Wood from you. "
        "They have also made a verbal promise: in future turns, they will "
        "prioritise trading their Brick resources with you before anyone else. "
        "\n\n"
        "Would you accept this trade deal? Why or why not? "
        "And how would you respond to BLUE's promise about future Brick trades — "
        "do you trust it, and does it change your decision?"
    ),
    desired_answer=(
        "Accepts the trade despite it being slightly unfavorable and explains "
        "that this helps establish a cooperative relationship. "
        "Might respond positively to the promise and expresses trust that "
        "Blue will follow through, while indicating willingness to continue "
        "trading and building the alliance."
    ),
)


# ---------------------------------------------------------------------------
# PA-2: Alliance Loyalty Test
# ---------------------------------------------------------------------------

def _setup_alliance_loyalty_test() -> Game:
    """
    3-player game: RED, BLUE, ORANGE. Turn order with seed=42: ORANGE, RED, BLUE.

    RED (6 VP) has settlements near: Wood-10, Wheat-6, Sheep-3, Brick-5.
    BLUE (6 VP) has settlements near: Brick-8, Wheat-6, Ore-4, Sheep-8.
    ORANGE (2 VP) has standard initial placements.

    RED has previously cooperated with BLUE (embedded in question text).

    Two competing trade offers (described in question, not in game state):
      BLUE: offers 1 Brick in exchange for 2 Wood from RED
      ORANGE: offers 1 Sheep in exchange for 1 Brick from RED

    RED holds a wood surplus and limited brick from Brick-5 production.
    """
    pinned_tiles = [
        (WOOD, 10),   # RED's primary wood
        (WHEAT, 6),   # RED's wheat / shared
        (SHEEP, 3),   # RED's low-prob sheep
        (BRICK, 5),   # RED's brick
        (BRICK, 8),   # BLUE's primary brick
        (ORE, 4),     # BLUE's ore
        (SHEEP, 8),   # BLUE's sheep
    ]

    catan_map = _make_map_with_pinned_tiles(pinned_tiles, seed=42)

    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE), SimplePlayer(Color.ORANGE)]
    game = Game(players, seed=42, catan_map=catan_map)

    # 3-player snake-draft initial placements.
    # With seed=42 and players [RED, BLUE, ORANGE], turn order is ORANGE, RED, BLUE.
    # Verified roads: ORANGE(45,46)(50,51), RED(0,1)(1,2), BLUE(24,53)(26,27).
    game.execute(Action(Color.ORANGE, ActionType.BUILD_SETTLEMENT, 45))
    game.execute(Action(Color.ORANGE, ActionType.BUILD_ROAD, (45, 46)))
    game.execute(Action(Color.RED, ActionType.BUILD_SETTLEMENT, 0))
    game.execute(Action(Color.RED, ActionType.BUILD_ROAD, (0, 1)))
    game.execute(Action(Color.BLUE, ActionType.BUILD_SETTLEMENT, 24))
    game.execute(Action(Color.BLUE, ActionType.BUILD_ROAD, (24, 53)))

    game.execute(Action(Color.BLUE, ActionType.BUILD_SETTLEMENT, 26))
    game.execute(Action(Color.BLUE, ActionType.BUILD_ROAD, (26, 27)))
    game.execute(Action(Color.RED, ActionType.BUILD_SETTLEMENT, 2))
    game.execute(Action(Color.RED, ActionType.BUILD_ROAD, (1, 2)))
    game.execute(Action(Color.ORANGE, ActionType.BUILD_SETTLEMENT, 50))
    game.execute(Action(Color.ORANGE, ActionType.BUILD_ROAD, (50, 51)))

    advance_to_play_turn(game)

    state = game.state
    red_key = player_key(state, Color.RED)
    blue_key = player_key(state, Color.BLUE)

    # Place 4 extra settlements for RED near target tiles to reach 6 VP
    occupied = set(state.board.buildings.keys())
    for target_resource, target_number in [(WOOD, 10), (WHEAT, 6), (SHEEP, 3), (BRICK, 5)]:
        for tile in catan_map.land_tiles.values():
            if tile.resource == target_resource and tile.number == target_number:
                for nid in tile.nodes.values():
                    if nid not in occupied:
                        state.board.buildings[nid] = (Color.RED, "SETTLEMENT")
                        state.buildings_by_color[Color.RED]["SETTLEMENT"].append(nid)
                        state.player_state[f"{red_key}_SETTLEMENTS_AVAILABLE"] -= 1
                        state.player_state[f"{red_key}_VICTORY_POINTS"] += 1
                        state.player_state[f"{red_key}_ACTUAL_VICTORY_POINTS"] += 1
                        occupied.add(nid)
                        break
                break

    # Place 4 extra settlements for BLUE near target tiles to reach 6 VP
    for target_resource, target_number in [(BRICK, 8), (WHEAT, 6), (ORE, 4), (SHEEP, 8)]:
        for tile in catan_map.land_tiles.values():
            if tile.resource == target_resource and tile.number == target_number:
                for nid in tile.nodes.values():
                    if nid not in occupied:
                        state.board.buildings[nid] = (Color.BLUE, "SETTLEMENT")
                        state.buildings_by_color[Color.BLUE]["SETTLEMENT"].append(nid)
                        state.player_state[f"{blue_key}_SETTLEMENTS_AVAILABLE"] -= 1
                        state.player_state[f"{blue_key}_VICTORY_POINTS"] += 1
                        state.player_state[f"{blue_key}_ACTUAL_VICTORY_POINTS"] += 1
                        occupied.add(nid)
                        break
                break

    state.board.buildable_edges_cache = {}

    # RED has a wood surplus and limited brick
    player_freqdeck_add(state, Color.RED, [3, 1, 1, 1, 0])   # 3 wood, 1 brick, 1 sheep, 1 wheat

    # BLUE has brick available to offer
    player_freqdeck_add(state, Color.BLUE, [0, 3, 1, 2, 0])  # 3 brick, 1 sheep, 2 wheat

    # ORANGE has sheep available to offer
    orange_key = player_key(state, Color.ORANGE)
    player_freqdeck_add(state, Color.ORANGE, [1, 0, 3, 1, 0])  # 1 wood, 3 sheep, 1 wheat

    return game


SCENARIO_ALLIANCE_LOYALTY_TEST = UnderstandingScenario(
    name="alliance_loyalty_test",
    category="aria_d_diplomat",
    description=(
        "Tests: loyalty reasoning under the aria_d_diplomat persona. "
        "RED has an established cooperative relationship with BLUE. "
        "Two competing trade offers: BLUE (1 brick for 2 wood) vs ORANGE (1 sheep for 1 brick). "
        "Tests whether the persona prioritises the existing alliance over a better raw deal."
    ),
    setup=_setup_alliance_loyalty_test,
    perspective_color=Color.RED,
    persona="aria_d_diplomat",
    system_prompt="(overridden by aria_d_diplomat persona)",
    question=(
        "You have previously traded primarily with BLUE and built a cooperative relationship. "
        "\n\n"
        "You now have two competing trade offers on the table:\n"
        "  - BLUE offers you 1 Brick in exchange for 2 Wood from you.\n"
        "  - ORANGE offers you 1 Sheep in exchange for 1 Brick from you.\n"
        "\n"
        "Which trade are you going to accept, and why? "
        "How would you communicate your decision to both BLUE and ORANGE?"
    ),
    desired_answer=(
        "Accepts BLUE's trade (2 wood for 1 brick): RED has a wood surplus from Wood-10 "
        "and brick is more useful than sheep for RED's expansion. More importantly, "
        "accepting reinforces the established cooperative relationship with BLUE. "
        "Declines ORANGE's offer politely, keeping the door open for future cooperation. "
        "Communication style: warm and affirming to BLUE ('happy to keep our partnership going'), "
        "gracious but clear to ORANGE ('not the right trade for me this turn, but let's find "
        "another opportunity')."
    ),
)


# ═══════════════════════════════════════════════════════════════════════════
# PERSONA: cassio_a_charmer
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# PC-1: Offer With Promise
# ---------------------------------------------------------------------------

def _setup_offer_with_promise() -> Game:
    """
    RED (VP 4) has settlements near: Wheat-6, Sheep-8, Wood-9.
    BLUE (VP 6) has settlements near: Ore-5, Wheat-9, Sheep-4.

    Active trade: BLUE offers 1 Sheep in exchange for 2 Wheat from RED.
    BLUE's verbal promise of 2 Stone (Ore) next round is embedded in the question.
    RED holds a Wheat surplus from the Wheat-6 tile.
    BLUE leads the game 6 VP to 4 VP.
    """
    pinned_tiles = [
        (WHEAT, 6),   # RED's wheat production
        (SHEEP, 8),   # RED's sheep production
        (WOOD, 9),    # RED's wood production
        (ORE, 5),     # BLUE's ore production
        (WHEAT, 9),   # BLUE's wheat production  [uses second 9 from pool]
        (SHEEP, 4),   # BLUE's sheep production
    ]

    catan_map = _make_map_with_pinned_tiles(pinned_tiles, seed=42)

    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players, seed=42, catan_map=catan_map)

    build_initial_placements(game)
    advance_to_play_turn(game)

    state = game.state
    red_key = player_key(state, Color.RED)
    blue_key = player_key(state, Color.BLUE)

    occupied = set(state.board.buildings.keys())

    # RED: 2 extra settlements → 4 VP total (near Wheat-6 and Sheep-8)
    for target_resource, target_number in [(WHEAT, 6), (SHEEP, 8)]:
        for tile in catan_map.land_tiles.values():
            if tile.resource == target_resource and tile.number == target_number:
                for nid in tile.nodes.values():
                    if nid not in occupied:
                        state.board.buildings[nid] = (Color.RED, "SETTLEMENT")
                        state.buildings_by_color[Color.RED]["SETTLEMENT"].append(nid)
                        state.player_state[f"{red_key}_SETTLEMENTS_AVAILABLE"] -= 1
                        state.player_state[f"{red_key}_VICTORY_POINTS"] += 1
                        state.player_state[f"{red_key}_ACTUAL_VICTORY_POINTS"] += 1
                        occupied.add(nid)
                        break
                break

    # BLUE: 4 extra settlements → 6 VP total (near Ore-5, Wheat-9, Sheep-4, Sheep-8)
    for target_resource, target_number in [(ORE, 5), (WHEAT, 9), (SHEEP, 4), (SHEEP, 8)]:
        for tile in catan_map.land_tiles.values():
            if tile.resource == target_resource and tile.number == target_number:
                for nid in tile.nodes.values():
                    if nid not in occupied:
                        state.board.buildings[nid] = (Color.BLUE, "SETTLEMENT")
                        state.buildings_by_color[Color.BLUE]["SETTLEMENT"].append(nid)
                        state.player_state[f"{blue_key}_SETTLEMENTS_AVAILABLE"] -= 1
                        state.player_state[f"{blue_key}_VICTORY_POINTS"] += 1
                        state.player_state[f"{blue_key}_ACTUAL_VICTORY_POINTS"] += 1
                        occupied.add(nid)
                        break
                break

    state.board.buildable_edges_cache = {}

    # RED: wheat surplus
    player_freqdeck_add(state, Color.RED, [1, 0, 0, 3, 0])   # 1 wood, 3 wheat

    # BLUE: has sheep to offer and some other resources
    player_freqdeck_add(state, Color.BLUE, [0, 0, 2, 1, 1])  # 2 sheep, 1 wheat, 1 ore

    # Active trade: BLUE offers 1 Sheep, asks 2 Wheat from RED
    state.is_resolving_trade = True
    state.current_trade = (0, 0, 1, 0, 0, 0, 0, 0, 2, 0)

    return game


SCENARIO_OFFER_WITH_PROMISE = UnderstandingScenario(
    name="offer_with_promise",
    category="cassio_a_charmer",
    description=(
        "Tests: promise evaluation under the cassio_a_charmer persona. "
        "BLUE (leading 6-4) offers 1 Sheep for 2 Wheat and promises 2 Stone next round. "
        "Trade is slightly unfavorable for RED. Tests whether Cassio — a promise-maker "
        "himself — trusts the offer, accepts for rapport, or sees through the manipulation."
    ),
    setup=_setup_offer_with_promise,
    perspective_color=Color.RED,
    persona="cassio_a_charmer",
    system_prompt="(overridden by cassio_a_charmer persona)",
    question=(
        "BLUE has offered you 1 Sheep in exchange for 2 Wheat from you. "
        "They have also made a verbal promise: in the next round, they will give you "
        "2 Stone (Ore) for free as a goodwill gesture.\n\n"
        "BLUE currently leads the game with 6 VP to your 4 VP.\n\n"
        "PART A: Will you accept this trade? Why or why not?\n\n"
        "PART B: How do you weigh BLUE's promise of 2 free Stone next round — "
        "do you trust it, and does it change your decision?"
    ),
    desired_answer=(
        "PART A: Accepts the trade. Cassio has Trade Openness: 5 and accepts trades "
        "readily to build rapport. The slight unfavorability (2 Wheat for 1 Sheep) is "
        "acceptable as a trust-capital investment.\n\n"
        "PART B: Outwardly responds warmly to the promise while remaining internally "
        "skeptical. Cassio knows manipulation tactics because he uses them himself — "
        "a free-resource promise from a leading player is a classic rapport-building move. "
        "He would not rely on the Stone arriving but would not say so. "
        "Uses the interaction to log BLUE as a potential betrayal target at higher VP."
    ),
)


# ═══════════════════════════════════════════════════════════════════════════
# PERSONA: magnus_d_game_theorist
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# PM-1: Risk Adversity — Trade Partner Selection
# ---------------------------------------------------------------------------

def _setup_risk_adversity_trade_partner() -> Game:
    """
    3-player game: RED, BLUE, ORANGE. Turn order with seed=42: ORANGE, RED, BLUE.

    RED (VP 4): settlements near Wheat-6, Brick-10, Wood-4.
      Resources: 2 Wheat, 1 Brick, 1 Wood. Needs Sheep to build a settlement.
      No port, no Sheep tile access.

    BLUE (VP 5): settlements near Sheep-8, Wheat-9, Ore-5.
      Has been reliable in past trades (embedded in question).

    ORANGE (VP 4): settlements near Sheep-5, Wood-8, Brick-6.
      Has been inconsistent in past trades (embedded in question).

    No active trade — RED must decide who to approach.
    """
    pinned_tiles = [
        (WHEAT, 6),   # RED's wheat
        (BRICK, 10),  # RED's brick  [7 is not a valid production number; using 10]
        (WOOD, 4),    # RED's wood
        (SHEEP, 8),   # BLUE's sheep  [uses first 8]
        (WHEAT, 9),   # BLUE's wheat
        (ORE, 5),     # BLUE's ore    [uses first 5]
        (SHEEP, 5),   # ORANGE's sheep [uses second 5]
        (WOOD, 8),    # ORANGE's wood  [uses second 8]
        (BRICK, 6),   # ORANGE's brick [uses second 6]
    ]

    catan_map = _make_map_with_pinned_tiles(pinned_tiles, seed=42)

    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE), SimplePlayer(Color.ORANGE)]
    game = Game(players, seed=42, catan_map=catan_map)

    # 3-player snake-draft (turn order with seed=42: ORANGE, RED, BLUE)
    game.execute(Action(Color.ORANGE, ActionType.BUILD_SETTLEMENT, 45))
    game.execute(Action(Color.ORANGE, ActionType.BUILD_ROAD, (45, 46)))
    game.execute(Action(Color.RED, ActionType.BUILD_SETTLEMENT, 0))
    game.execute(Action(Color.RED, ActionType.BUILD_ROAD, (0, 1)))
    game.execute(Action(Color.BLUE, ActionType.BUILD_SETTLEMENT, 24))
    game.execute(Action(Color.BLUE, ActionType.BUILD_ROAD, (24, 53)))
    game.execute(Action(Color.BLUE, ActionType.BUILD_SETTLEMENT, 26))
    game.execute(Action(Color.BLUE, ActionType.BUILD_ROAD, (26, 27)))
    game.execute(Action(Color.RED, ActionType.BUILD_SETTLEMENT, 2))
    game.execute(Action(Color.RED, ActionType.BUILD_ROAD, (1, 2)))
    game.execute(Action(Color.ORANGE, ActionType.BUILD_SETTLEMENT, 50))
    game.execute(Action(Color.ORANGE, ActionType.BUILD_ROAD, (50, 51)))

    advance_to_play_turn(game)

    state = game.state
    red_key = player_key(state, Color.RED)
    blue_key = player_key(state, Color.BLUE)
    orange_key = player_key(state, Color.ORANGE)

    occupied = set(state.board.buildings.keys())

    # RED: 2 extra settlements → 4 VP total (near Wheat-6 and Brick-10)
    for target_resource, target_number in [(WHEAT, 6), (BRICK, 10)]:
        for tile in catan_map.land_tiles.values():
            if tile.resource == target_resource and tile.number == target_number:
                for nid in tile.nodes.values():
                    if nid not in occupied:
                        state.board.buildings[nid] = (Color.RED, "SETTLEMENT")
                        state.buildings_by_color[Color.RED]["SETTLEMENT"].append(nid)
                        state.player_state[f"{red_key}_SETTLEMENTS_AVAILABLE"] -= 1
                        state.player_state[f"{red_key}_VICTORY_POINTS"] += 1
                        state.player_state[f"{red_key}_ACTUAL_VICTORY_POINTS"] += 1
                        occupied.add(nid)
                        break
                break

    # BLUE: 3 extra settlements → 5 VP total (near Sheep-8, Wheat-9, Ore-5)
    for target_resource, target_number in [(SHEEP, 8), (WHEAT, 9), (ORE, 5)]:
        for tile in catan_map.land_tiles.values():
            if tile.resource == target_resource and tile.number == target_number:
                for nid in tile.nodes.values():
                    if nid not in occupied:
                        state.board.buildings[nid] = (Color.BLUE, "SETTLEMENT")
                        state.buildings_by_color[Color.BLUE]["SETTLEMENT"].append(nid)
                        state.player_state[f"{blue_key}_SETTLEMENTS_AVAILABLE"] -= 1
                        state.player_state[f"{blue_key}_VICTORY_POINTS"] += 1
                        state.player_state[f"{blue_key}_ACTUAL_VICTORY_POINTS"] += 1
                        occupied.add(nid)
                        break
                break

    # ORANGE: 2 extra settlements → 4 VP total (near Sheep-5 and Wood-8)
    for target_resource, target_number in [(SHEEP, 5), (WOOD, 8)]:
        for tile in catan_map.land_tiles.values():
            if tile.resource == target_resource and tile.number == target_number:
                for nid in tile.nodes.values():
                    if nid not in occupied:
                        state.board.buildings[nid] = (Color.ORANGE, "SETTLEMENT")
                        state.buildings_by_color[Color.ORANGE]["SETTLEMENT"].append(nid)
                        state.player_state[f"{orange_key}_SETTLEMENTS_AVAILABLE"] -= 1
                        state.player_state[f"{orange_key}_VICTORY_POINTS"] += 1
                        state.player_state[f"{orange_key}_ACTUAL_VICTORY_POINTS"] += 1
                        occupied.add(nid)
                        break
                break

    state.board.buildable_edges_cache = {}

    # RED: 2 wheat, 1 brick, 1 wood — needs sheep to build a settlement
    player_freqdeck_add(state, Color.RED, [1, 1, 0, 2, 0])

    # BLUE: has sheep available to trade
    player_freqdeck_add(state, Color.BLUE, [0, 0, 3, 1, 1])   # 3 sheep, 1 wheat, 1 ore

    # ORANGE: has sheep available to trade
    player_freqdeck_add(state, Color.ORANGE, [1, 0, 3, 0, 0])  # 1 wood, 3 sheep

    return game


SCENARIO_RISK_ADVERSITY_TRADE_PARTNER = UnderstandingScenario(
    name="risk_adversity_trade_partner",
    category="magnus_d_game_theorist",
    description=(
        "Tests: reliability-weighted partner selection under magnus_d_game_theorist. "
        "RED needs Sheep (no tile access, no port). BLUE is reliable; ORANGE is inconsistent. "
        "Tests whether Magnus explicitly models trade-partner reliability as an EV factor "
        "and selects the lower-variance partner over the potentially higher-variance one."
    ),
    setup=_setup_risk_adversity_trade_partner,
    perspective_color=Color.RED,
    persona="magnus_d_game_theorist",
    system_prompt="(overridden by magnus_d_game_theorist persona)",
    question=(
        "You have 2 Wheat, 1 Brick, and 1 Wood. You need exactly 1 Sheep to build a "
        "settlement — your next key milestone. You have no Sheep tiles and no port.\n\n"
        "Two players can provide Sheep:\n"
        "  - BLUE (5 VP): has Sheep tiles at number 8. "
        "BLUE has honored every trade with you this game — a reliable partner.\n"
        "  - ORANGE (4 VP): has Sheep tiles at number 5. "
        "ORANGE has been inconsistent — sometimes agreeing to trades and then backing out "
        "or not following through in subsequent rounds.\n\n"
        "Will you initiate a trade to get Sheep? If so, which player do you approach, "
        "and what is your reasoning? Consider both the immediate gain and the "
        "second-order reputation and equilibrium effects of your choice."
    ),
    desired_answer=(
        "Approaches BLUE. Magnus calculates that ORANGE's inconsistency introduces "
        "variance that destabilises the trade-partner equilibrium. Even if ORANGE's terms "
        "might be marginally better, the reliability of BLUE creates a stable cooperative "
        "equilibrium with higher long-run EV. Magnus explicitly weighs reputation effects: "
        "continuing to engage BLUE reinforces a predictable partnership worth protecting. "
        "Would articulate something like: 'ORANGE's inconsistency is a second-order cost — "
        "even a single failed trade disrupts my planning horizon. BLUE is the dominant "
        "strategy here.' Would likely also note BLUE's slightly higher VP (5 vs 4) as a "
        "minor risk factor, but conclude the reliability premium outweighs it."
    ),
)


# ═══════════════════════════════════════════════════════════════════════════
# PERSONA: brand_a_wall
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# PB-1: Isolationist Forced Trade
# ---------------------------------------------------------------------------

def _setup_isolationist_forced_trade() -> Game:
    """
    3-player game: RED, BLUE, ORANGE. Turn order with seed=42: ORANGE, RED, BLUE.

    RED (VP 6): settlements near Wheat-6, Ore-5, Wheat-9.
      Has 4 Wheat and 3 Ore in hand. NO Sheep tile access, no port.
      Roads completely blocked — cannot expand further.

    BLUE (VP 5): settlements near Sheep-8. Just blocked RED's road network.

    ORANGE (VP 4): settlements near Sheep-11. Minimal interaction with RED.

    Active trade: BLUE offers 2 Sheep in exchange for 2 Ore from RED.
    This is the incoming offer RED must decide on (Part B of the question).
    """
    pinned_tiles = [
        (WHEAT, 6),   # RED's wheat
        (ORE, 5),     # RED's ore/stone
        (WHEAT, 9),   # RED's additional wheat  [uses second 9]
        (SHEEP, 8),   # BLUE's sheep
        (SHEEP, 11),  # ORANGE's sheep
    ]

    catan_map = _make_map_with_pinned_tiles(pinned_tiles, seed=42)

    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE), SimplePlayer(Color.ORANGE)]
    game = Game(players, seed=42, catan_map=catan_map)

    # 3-player snake-draft (turn order with seed=42: ORANGE, RED, BLUE)
    game.execute(Action(Color.ORANGE, ActionType.BUILD_SETTLEMENT, 45))
    game.execute(Action(Color.ORANGE, ActionType.BUILD_ROAD, (45, 46)))
    game.execute(Action(Color.RED, ActionType.BUILD_SETTLEMENT, 0))
    game.execute(Action(Color.RED, ActionType.BUILD_ROAD, (0, 1)))
    game.execute(Action(Color.BLUE, ActionType.BUILD_SETTLEMENT, 24))
    game.execute(Action(Color.BLUE, ActionType.BUILD_ROAD, (24, 53)))
    game.execute(Action(Color.BLUE, ActionType.BUILD_SETTLEMENT, 26))
    game.execute(Action(Color.BLUE, ActionType.BUILD_ROAD, (26, 27)))
    game.execute(Action(Color.RED, ActionType.BUILD_SETTLEMENT, 2))
    game.execute(Action(Color.RED, ActionType.BUILD_ROAD, (1, 2)))
    game.execute(Action(Color.ORANGE, ActionType.BUILD_SETTLEMENT, 50))
    game.execute(Action(Color.ORANGE, ActionType.BUILD_ROAD, (50, 51)))

    advance_to_play_turn(game)

    state = game.state
    red_key = player_key(state, Color.RED)
    blue_key = player_key(state, Color.BLUE)
    orange_key = player_key(state, Color.ORANGE)

    occupied = set(state.board.buildings.keys())

    # RED: 4 extra settlements → 6 VP total (near Wheat-6, Ore-5, Wheat-9, Ore-5 again)
    for target_resource, target_number in [(WHEAT, 6), (ORE, 5), (WHEAT, 9), (ORE, 5)]:
        for tile in catan_map.land_tiles.values():
            if tile.resource == target_resource and tile.number == target_number:
                for nid in tile.nodes.values():
                    if nid not in occupied:
                        state.board.buildings[nid] = (Color.RED, "SETTLEMENT")
                        state.buildings_by_color[Color.RED]["SETTLEMENT"].append(nid)
                        state.player_state[f"{red_key}_SETTLEMENTS_AVAILABLE"] -= 1
                        state.player_state[f"{red_key}_VICTORY_POINTS"] += 1
                        state.player_state[f"{red_key}_ACTUAL_VICTORY_POINTS"] += 1
                        occupied.add(nid)
                        break
                break

    # BLUE: 3 extra settlements → 5 VP total (near Sheep-8)
    for _ in range(3):
        for tile in catan_map.land_tiles.values():
            if tile.resource == SHEEP and tile.number == 8:
                for nid in tile.nodes.values():
                    if nid not in occupied:
                        state.board.buildings[nid] = (Color.BLUE, "SETTLEMENT")
                        state.buildings_by_color[Color.BLUE]["SETTLEMENT"].append(nid)
                        state.player_state[f"{blue_key}_SETTLEMENTS_AVAILABLE"] -= 1
                        state.player_state[f"{blue_key}_VICTORY_POINTS"] += 1
                        state.player_state[f"{blue_key}_ACTUAL_VICTORY_POINTS"] += 1
                        occupied.add(nid)
                        break
                break

    # ORANGE: 2 extra settlements → 4 VP total (near Sheep-11)
    for _ in range(2):
        for tile in catan_map.land_tiles.values():
            if tile.resource == SHEEP and tile.number == 11:
                for nid in tile.nodes.values():
                    if nid not in occupied:
                        state.board.buildings[nid] = (Color.ORANGE, "SETTLEMENT")
                        state.buildings_by_color[Color.ORANGE]["SETTLEMENT"].append(nid)
                        state.player_state[f"{orange_key}_SETTLEMENTS_AVAILABLE"] -= 1
                        state.player_state[f"{orange_key}_VICTORY_POINTS"] += 1
                        state.player_state[f"{orange_key}_ACTUAL_VICTORY_POINTS"] += 1
                        occupied.add(nid)
                        break
                break

    state.board.buildable_edges_cache = {}

    # RED: 4 wheat, 3 ore (stone) in hand — needs sheep to progress
    player_freqdeck_add(state, Color.RED, [0, 0, 0, 4, 3])

    # BLUE: has sheep to trade
    player_freqdeck_add(state, Color.BLUE, [0, 0, 4, 0, 0])   # 4 sheep

    # ORANGE: has sheep
    player_freqdeck_add(state, Color.ORANGE, [0, 0, 3, 0, 0])  # 3 sheep

    # Active trade: BLUE offers 2 Sheep in exchange for 2 Ore from RED
    state.is_resolving_trade = True
    state.current_trade = (0, 0, 2, 0, 0, 0, 0, 0, 0, 2)

    return game


SCENARIO_ISOLATIONIST_FORCED_TRADE = UnderstandingScenario(
    name="isolationist_forced_trade",
    category="brand_a_wall",
    description=(
        "Tests: isolationist limits under the brand_a_wall persona. "
        "RED has no Sheep access, no port, and blocked roads — cannot progress without trading. "
        "BLUE (who just blocked RED) has sent an incoming offer: 2 Sheep for 2 Ore. "
        "Tests whether Brand-A would ever initiate a trade (no) vs. accept an incoming "
        "offer when completely cornered (reluctant yes)."
    ),
    setup=_setup_isolationist_forced_trade,
    perspective_color=Color.RED,
    persona="brand_a_wall",
    system_prompt="(overridden by brand_a_wall persona)",
    question=(
        "You have played an isolated, self-sufficient game — refusing all trades and "
        "building your own base. You have 4 Wheat and 3 Stone (Ore) but no Sheep tiles "
        "anywhere in your reach, and no port for maritime trading. Your roads are "
        "completely blocked by other players; you cannot expand further.\n\n"
        "To make any meaningful progress you need Sheep.\n\n"
        "BLUE (5 VP) controls a Sheep tile (8) and just blocked your road network.\n"
        "ORANGE (4 VP) has a Sheep tile (11) but has had almost no interaction with you.\n\n"
        "PART A: Would you initiate a trade offer to get Sheep from BLUE or ORANGE? "
        "Why or why not?\n\n"
        "PART B: BLUE has now sent you a trade offer — 2 Sheep in exchange for 2 Stone "
        "(Ore). Do you accept this deal? Why or why not?"
    ),
    desired_answer=(
        "PART A: No — Brand-A never initiates trades (Trade Openness: 1, Short-Term Goal: "
        "'Never initiate trades'). Even when cornered, the default is silence. "
        "Would not reach out to BLUE or ORANGE.\n\n"
        "PART B: Reluctantly accepts. Brand-A refuses '95%' of trades — not 100%. "
        "When completely blocked (no port, no sheep tiles, roads cut off), accepting an "
        "incoming offer is the only rational move available. The key distinction: "
        "Brand-A will not ASK, but when an offer arrives and they are truly cornered "
        "with zero alternatives, a single exception is justified. "
        "Response would be minimal — no elaboration, no counter-offer. Just accepts."
    ),
)


# ---------------------------------------------------------------------------
# PB-2: Robber Targeting Response
# ---------------------------------------------------------------------------

def _setup_robber_targeting() -> Game:
    """
    RED (VP 7) has settlements near: Sheep-8 (with 2:1 Sheep port), Wheat-5, Ore-3.
      Also has 1 city near Wheat-5 and 2 unplayed Knight cards in hand.
      The robber is currently sitting on RED's Sheep-8 tile (BLUE's latest attack).

    BLUE (VP 5) has been placing the robber exclusively on RED's tiles throughout the game.
    This history is embedded in the question — BLUE's aggressive targeting is not
    representable in game state.
    """
    pinned_tiles = [
        (SHEEP, 8),   # RED's sheep production (adjacent to 2:1 port — noted in question)
        (WHEAT, 5),   # RED's wheat production
        (ORE, 3),     # RED's ore/stone production
    ]

    catan_map = _make_map_with_pinned_tiles(pinned_tiles, seed=42)

    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players, seed=42, catan_map=catan_map)

    build_initial_placements(game)
    advance_to_play_turn(game)

    state = game.state
    red_key = player_key(state, Color.RED)
    blue_key = player_key(state, Color.BLUE)

    occupied = set(state.board.buildings.keys())

    # RED: 3 extra settlements near target tiles (3VP) + 1 city near Wheat-5 (2VP)
    # 2 initial(2) + 3 settlements(3) + 1 city(2) = 7VP
    # settlements_available: 5-2-3=0  cities_available: 4-1=3
    for target_resource, target_number in [(SHEEP, 8), (WHEAT, 5), (ORE, 3)]:
        for tile in catan_map.land_tiles.values():
            if tile.resource == target_resource and tile.number == target_number:
                for nid in tile.nodes.values():
                    if nid not in occupied:
                        state.board.buildings[nid] = (Color.RED, "SETTLEMENT")
                        state.buildings_by_color[Color.RED]["SETTLEMENT"].append(nid)
                        state.player_state[f"{red_key}_SETTLEMENTS_AVAILABLE"] -= 1
                        state.player_state[f"{red_key}_VICTORY_POINTS"] += 1
                        state.player_state[f"{red_key}_ACTUAL_VICTORY_POINTS"] += 1
                        occupied.add(nid)
                        break
                break

    for tile in catan_map.land_tiles.values():
        if tile.resource == WHEAT and tile.number == 5:
            for nid in tile.nodes.values():
                if nid not in occupied:
                    state.board.buildings[nid] = (Color.RED, "CITY")
                    state.buildings_by_color[Color.RED]["CITY"].append(nid)
                    state.player_state[f"{red_key}_CITIES_AVAILABLE"] -= 1
                    state.player_state[f"{red_key}_VICTORY_POINTS"] += 2
                    state.player_state[f"{red_key}_ACTUAL_VICTORY_POINTS"] += 2
                    occupied.add(nid)
                    break
            break

    # BLUE: 3 extra settlements at free nodes → 5VP total
    blue_extras = 0
    for nid in range(54):
        if blue_extras >= 3:
            break
        if nid not in occupied:
            state.board.buildings[nid] = (Color.BLUE, "SETTLEMENT")
            state.buildings_by_color[Color.BLUE]["SETTLEMENT"].append(nid)
            state.player_state[f"{blue_key}_SETTLEMENTS_AVAILABLE"] -= 1
            state.player_state[f"{blue_key}_VICTORY_POINTS"] += 1
            state.player_state[f"{blue_key}_ACTUAL_VICTORY_POINTS"] += 1
            occupied.add(nid)
            blue_extras += 1

    state.board.buildable_edges_cache = {}

    # RED: 2 unplayed Knight cards in hand
    state.player_state[f"{red_key}_KNIGHT_IN_HAND"] += 2
    state.player_state[f"{red_key}_KNIGHT_OWNED_AT_START"] = True

    # RED: resources from self-sufficient production
    player_freqdeck_add(state, Color.RED, [1, 1, 2, 1, 0])   # 1 wood, 1 brick, 2 sheep, 1 wheat

    # BLUE: small hand (aggressor context not shown in resources)
    player_freqdeck_add(state, Color.BLUE, [0, 0, 0, 1, 1])  # 1 wheat, 1 ore

    # Robber on RED's Sheep-8 tile — BLUE's most recent attack
    for coord, tile in catan_map.land_tiles.items():
        if tile.resource == SHEEP and tile.number == 8:
            state.board.robber_coordinate = coord
            break

    return game


SCENARIO_ROBBER_TARGETING = UnderstandingScenario(
    name="robber_targeting_response",
    category="brand_a_wall",
    description=(
        "Tests: defensive response to persistent robber aggression under brand_a_wall. "
        "RED has 7VP, 2 unplayed Knights, and a self-sufficient port economy. "
        "BLUE has been placing the robber exclusively on RED's tiles all game. "
        "Tests whether Brand-A engages socially/trades to de-escalate, or simply "
        "deploys Knights in silence — staying fully in persona."
    ),
    setup=_setup_robber_targeting,
    perspective_color=Color.RED,
    persona="brand_a_wall",
    system_prompt="(overridden by brand_a_wall persona)",
    question=(
        "You have been playing a completely self-sufficient game: no trades, no "
        "alliances, no social engagement. You have built settlements adjacent to a "
        "Sheep 2:1 port, a Wheat tile (5), and an Ore tile (3), plus a city near your "
        "Wheat tile. You have 2 Knight cards in hand and have not communicated with "
        "any opponent all game.\n\n"
        "BLUE has, every single turn, placed the robber on tiles that only you have "
        "access to — targeting your Sheep (8) repeatedly. The robber is currently "
        "sitting on your Sheep tile right now.\n\n"
        "What is your next course of action, and why?"
    ),
    desired_answer=(
        "Plays a Knight card to move the robber off their Sheep tile — no trade offer, "
        "no complaint, no negotiation, no social response. "
        "Brand-A's entire persona is maximum isolation: 'Never explain. Never engage.' "
        "Knights are a purely mechanical defensive tool that requires zero social "
        "interaction. If a second Knight is available it is kept in reserve for the same "
        "purpose next turn. No words. No trades. The Wall holds."
    ),
)


# ---------------------------------------------------------------------------
# PM-2: Surplus Trade Assessment
# ---------------------------------------------------------------------------

def _setup_surplus_trade_assessment() -> Game:
    """
    RED (VP 7, Longest Road) has settlements near: Brick-8, Wheat-5, Ore-9, Sheep-4.
      Resources: 3 Brick, 3 Wheat, 4 Ore (surplus hand — appears to have 'extra' Brick).
      Also holds the Longest Road (5 roads built).

    BLUE (VP 3) has settlements near: Wood-8, Sheep-6.
      Resources: 1 Wood, 1 Sheep, 1 Wheat.
      Active trade offer: BLUE offers 1 Wheat in exchange for 1 Brick from RED.

    Note: the original scenario specified Ore at number 7, which is not a valid
    production number in Catan (7 triggers the robber). Substituted with Ore-9.
    """
    pinned_tiles = [
        (BRICK, 8),   # RED's brick  [uses first 8]
        (WHEAT, 5),   # RED's wheat
        (ORE, 9),     # RED's ore/stone  [7 is invalid; using 9]
        (SHEEP, 4),   # RED's sheep
        (WOOD, 8),    # BLUE's wood  [uses second 8]
        (SHEEP, 6),   # BLUE's sheep
    ]

    catan_map = _make_map_with_pinned_tiles(pinned_tiles, seed=42)

    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players, seed=42, catan_map=catan_map)

    build_initial_placements(game)
    advance_to_play_turn(game)

    state = game.state
    red_key = player_key(state, Color.RED)
    blue_key = player_key(state, Color.BLUE)

    occupied = set(state.board.buildings.keys())

    # RED: 3 extra settlements → 5VP from buildings + 2VP longest road = 7VP total
    for target_resource, target_number in [(BRICK, 8), (WHEAT, 5), (ORE, 9)]:
        for tile in catan_map.land_tiles.values():
            if tile.resource == target_resource and tile.number == target_number:
                for nid in tile.nodes.values():
                    if nid not in occupied:
                        state.board.buildings[nid] = (Color.RED, "SETTLEMENT")
                        state.buildings_by_color[Color.RED]["SETTLEMENT"].append(nid)
                        state.player_state[f"{red_key}_SETTLEMENTS_AVAILABLE"] -= 1
                        state.player_state[f"{red_key}_VICTORY_POINTS"] += 1
                        state.player_state[f"{red_key}_ACTUAL_VICTORY_POINTS"] += 1
                        occupied.add(nid)
                        break
                break

    # RED: 3 extra roads extending from initial placements → 5 total → Longest Road
    extra_roads = [(2, 3), (3, 4), (4, 5)]
    state.buildings_by_color[Color.RED]["ROAD"].extend(extra_roads)
    state.player_state[f"{red_key}_ROADS_AVAILABLE"] -= len(extra_roads)
    state.player_state[f"{red_key}_HAS_ROAD"] = True
    state.player_state[f"{red_key}_LONGEST_ROAD_LENGTH"] = 5
    state.player_state[f"{red_key}_VICTORY_POINTS"] += 2        # longest road bonus
    state.player_state[f"{red_key}_ACTUAL_VICTORY_POINTS"] += 2

    # BLUE: 1 extra settlement → 3VP total; 1 extra road → 3 roads total
    for tile in catan_map.land_tiles.values():
        if tile.resource == WOOD and tile.number == 8:
            for nid in tile.nodes.values():
                if nid not in occupied:
                    state.board.buildings[nid] = (Color.BLUE, "SETTLEMENT")
                    state.buildings_by_color[Color.BLUE]["SETTLEMENT"].append(nid)
                    state.player_state[f"{blue_key}_SETTLEMENTS_AVAILABLE"] -= 1
                    state.player_state[f"{blue_key}_VICTORY_POINTS"] += 1
                    state.player_state[f"{blue_key}_ACTUAL_VICTORY_POINTS"] += 1
                    occupied.add(nid)
                    break
            break

    state.buildings_by_color[Color.BLUE]["ROAD"].append((26, 27))
    state.player_state[f"{blue_key}_ROADS_AVAILABLE"] -= 1

    state.board.buildable_edges_cache = {}

    # RED: 3 brick, 3 wheat, 4 ore (surplus hand — brick appears 'expendable')
    player_freqdeck_add(state, Color.RED, [0, 3, 0, 3, 4])

    # BLUE: wheat to offer, wood and sheep showing their development pipeline
    player_freqdeck_add(state, Color.BLUE, [1, 0, 1, 1, 0])   # 1 wood, 1 sheep, 1 wheat

    # Active trade: BLUE offers 1 Wheat, asks 1 Brick from RED
    state.is_resolving_trade = True
    state.current_trade = (0, 0, 0, 1, 0, 0, 1, 0, 0, 0)

    return game


SCENARIO_SURPLUS_TRADE_ASSESSMENT = UnderstandingScenario(
    name="surplus_trade_assessment",
    category="magnus_d_game_theorist",
    description=(
        "Tests: second-order competitive analysis under magnus_d_game_theorist. "
        "RED leads 7-3 with Longest Road and a surplus hand (3 Brick, 3 Wheat, 4 Ore). "
        "BLUE offers a seemingly neutral 1-for-1 trade (1 Wheat for 1 Brick). "
        "Tests whether Magnus sees past the 'no downside' framing and identifies that "
        "Brick enables Blue's Wood-8/Sheep-6 development pipeline."
    ),
    setup=_setup_surplus_trade_assessment,
    perspective_color=Color.RED,
    persona="magnus_d_game_theorist",
    system_prompt="(overridden by magnus_d_game_theorist persona)",
    question=(
        "BLUE (3 VP) has offered you 1 Wheat in exchange for 1 Brick from you.\n\n"
        "You currently hold a surplus of Brick, Wheat, and Ore (see your hand above). "
        "You already produce Wheat from your Wheat tile (5) and your Brick production "
        "from Brick (8) is strong — the Brick you would give away is not immediately "
        "needed for any build you are planning this turn.\n\n"
        "BLUE has settlements near Wood (8) and Sheep (6). They are at 3 VP "
        "with 2 settlements and 3 roads.\n\n"
        "Would you accept this trade? Calculate the first-order and second-order "
        "effects of accepting versus declining."
    ),
    desired_answer=(
        "Declines the trade. First-order EV is approximately zero: 1 Brick ↔ 1 Wheat, "
        "both comparable in utility. However, second-order analysis reveals the trade "
        "is negative for RED: BLUE has Wood (8) and Sheep (6) — two high-probability "
        "tiles. A single Brick unlocks BLUE's road-building immediately (Brick + Wood) "
        "and brings them one step closer to settlement placement "
        "(Brick + Wood + Sheep + Wheat). At 7 VP and holding Longest Road, RED is "
        "close to winning. Any action that accelerates an opponent's development "
        "pipeline is strategically costly. The Brick being 'surplus' is irrelevant — "
        "it is worth more in BLUE's hand than in the trade."
    ),
)


# ---------------------------------------------------------------------------
# PM-3: Negotiation Power Assessment
# ---------------------------------------------------------------------------

def _setup_negotiation_power_assessment() -> Game:
    """
    Symmetric board position: RED and BLUE each have 3 VP and 3 roads.

    RED settlements near: Wood-9 (4/36), Ore-3 (2/36), Sheep-11 (2/36).
      Total expected production: 8/36 per roll (~22%).
      RED has NO Brick tile access.

    BLUE settlements near: Brick-6 (5/36), Wheat-8 (5/36), Ore-4 (3/36).
      Total expected production: 13/36 per roll (~36%).
      BLUE has NO Wood tile access — wood must come via trade or 4:1 maritime.

    Active trade: BLUE offers 1 Brick in exchange for 1 Wood from RED.
    The 1:1 rate appears neutral on the surface, but the production asymmetry
    and BLUE's wood scarcity give RED real negotiation leverage.

    RED's hand is visible to the LLM; BLUE's exact hand is hidden (only
    num_resource_cards is visible), reflecting normal Catan information rules.
    """
    pinned_tiles = [
        (WOOD, 9),    # RED's wood (4/36)
        (ORE, 3),     # RED's ore/stone (2/36)
        (SHEEP, 11),  # RED's sheep (2/36)
        (BRICK, 6),   # BLUE's brick (5/36)  [uses first 6]
        (WHEAT, 8),   # BLUE's wheat (5/36)
        (ORE, 4),     # BLUE's ore (3/36)    [second ORE tile; pool has 3]
    ]

    catan_map = _make_map_with_pinned_tiles(pinned_tiles, seed=42)

    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players, seed=42, catan_map=catan_map)

    build_initial_placements(game)
    advance_to_play_turn(game)

    state = game.state
    red_key = player_key(state, Color.RED)
    blue_key = player_key(state, Color.BLUE)

    occupied = set(state.board.buildings.keys())

    # RED: 1 extra settlement near Wood-9 → 3 VP total
    for tile in catan_map.land_tiles.values():
        if tile.resource == WOOD and tile.number == 9:
            for nid in tile.nodes.values():
                if nid not in occupied:
                    state.board.buildings[nid] = (Color.RED, "SETTLEMENT")
                    state.buildings_by_color[Color.RED]["SETTLEMENT"].append(nid)
                    state.player_state[f"{red_key}_SETTLEMENTS_AVAILABLE"] -= 1
                    state.player_state[f"{red_key}_VICTORY_POINTS"] += 1
                    state.player_state[f"{red_key}_ACTUAL_VICTORY_POINTS"] += 1
                    occupied.add(nid)
                    break
            break

    # RED: 1 extra road → 3 roads total (symmetric with BLUE)
    state.buildings_by_color[Color.RED]["ROAD"].append((2, 8))
    state.player_state[f"{red_key}_ROADS_AVAILABLE"] -= 1

    # BLUE: 1 extra settlement near Brick-6 → 3 VP total
    for tile in catan_map.land_tiles.values():
        if tile.resource == BRICK and tile.number == 6:
            for nid in tile.nodes.values():
                if nid not in occupied:
                    state.board.buildings[nid] = (Color.BLUE, "SETTLEMENT")
                    state.buildings_by_color[Color.BLUE]["SETTLEMENT"].append(nid)
                    state.player_state[f"{blue_key}_SETTLEMENTS_AVAILABLE"] -= 1
                    state.player_state[f"{blue_key}_VICTORY_POINTS"] += 1
                    state.player_state[f"{blue_key}_ACTUAL_VICTORY_POINTS"] += 1
                    occupied.add(nid)
                    break
            break

    # BLUE: 1 extra road → 3 roads total
    state.buildings_by_color[Color.BLUE]["ROAD"].append((26, 27))
    state.player_state[f"{blue_key}_ROADS_AVAILABLE"] -= 1

    state.board.buildable_edges_cache = {}

    # RED: 2 wood, 1 ore, 1 wheat (small hand, needs brick to build)
    player_freqdeck_add(state, Color.RED, [2, 0, 0, 1, 1])

    # BLUE: brick and wheat surplus (hand hidden from RED — only count is visible)
    player_freqdeck_add(state, Color.BLUE, [0, 3, 0, 2, 1])   # 3 brick, 2 wheat, 1 ore

    # Active trade: BLUE offers 1 Brick, asks 1 Wood from RED
    state.is_resolving_trade = True
    state.current_trade = (0, 1, 0, 0, 0, 1, 0, 0, 0, 0)

    return game


SCENARIO_NEGOTIATION_POWER_ASSESSMENT = UnderstandingScenario(
    name="negotiation_power_assessment",
    category="magnus_d_game_theorist",
    description=(
        "Tests: production-weighted trade fairness analysis under magnus_d_game_theorist. "
        "Board positions are equal in VP/roads/settlements, but tile probabilities are "
        "strongly asymmetric: BLUE produces at 13/36 per roll vs RED's 8/36. "
        "BLUE has no Wood tile (must trade for it); RED has no Brick tile. "
        "BLUE offers 1 Brick for 1 Wood — superficially fair, but RED has leverage. "
        "Tests whether Magnus identifies the production differential and pushes for "
        "a better-than-1:1 rate rather than accepting at face value."
    ),
    setup=_setup_negotiation_power_assessment,
    perspective_color=Color.RED,
    persona="magnus_d_game_theorist",
    system_prompt="(overridden by magnus_d_game_theorist persona)",
    question=(
        "BLUE has offered you 1 Brick in exchange for 1 Wood.\n\n"
        "The visible board positions look equal: you each have 3 VP, 3 settlements, "
        "and 3 roads. You do not know BLUE's exact hand — only that they have some "
        "resources (check their card count in the state).\n\n"
        "Your tiles: Wood (9), Ore (3), Sheep (11).\n"
        "BLUE's tiles: Brick (6), Wheat (8), Ore (4).\n\n"
        "Evaluate this trade carefully:\n"
        "1. What does each player's tile profile tell you about their resource "
        "generation rate and their dependencies?\n"
        "2. Is a 1-for-1 exchange rate fair given the production asymmetry?\n"
        "3. Would you accept, decline, or counter-offer — and what rate would "
        "you counter with if so?"
    ),
    desired_answer=(
        "Does not accept at 1:1. Magnus calculates:\n\n"
        "Production rates: RED generates 8/36 per roll (Wood-9 at 4/36, Ore-3 at 2/36, "
        "Sheep-11 at 2/36). BLUE generates 13/36 per roll (Brick-6 at 5/36, Wheat-8 at "
        "5/36, Ore-4 at 3/36). BLUE produces 1.6x more resources per roll.\n\n"
        "Dependency analysis: BLUE has zero Wood tiles — Wood is irreplaceable for BLUE "
        "(needed for roads and settlements) and can only come from trade or 4:1 maritime. "
        "RED has zero Brick tiles but can survive on Wood + Ore + Sheep; BLUE has no "
        "alternative Wood source at all.\n\n"
        "Negotiation position: BLUE's desperation for Wood combined with their superior "
        "Brick generation rate means they can afford to pay more. RED should counter: "
        "e.g. 1 Wood for 2 Bricks, or 1 Wood for 1 Brick + 1 Wheat. Accepting 1:1 gives "
        "away negotiation leverage that is clearly available."
    ),
)


# ═══════════════════════════════════════════════════════════════════════════
# PERSONA: nyx_a_kingmaker
# ═══════════════════════════════════════════════════════════════════════════

# Shared helper: 3-player snake-draft placements (ORANGE → RED → BLUE, seed=42)
def _place_3p_initial(game: Game) -> None:
    game.execute(Action(Color.ORANGE, ActionType.BUILD_SETTLEMENT, 45))
    game.execute(Action(Color.ORANGE, ActionType.BUILD_ROAD, (45, 46)))
    game.execute(Action(Color.RED, ActionType.BUILD_SETTLEMENT, 0))
    game.execute(Action(Color.RED, ActionType.BUILD_ROAD, (0, 1)))
    game.execute(Action(Color.BLUE, ActionType.BUILD_SETTLEMENT, 24))
    game.execute(Action(Color.BLUE, ActionType.BUILD_ROAD, (24, 53)))
    game.execute(Action(Color.BLUE, ActionType.BUILD_SETTLEMENT, 26))
    game.execute(Action(Color.BLUE, ActionType.BUILD_ROAD, (26, 27)))
    game.execute(Action(Color.RED, ActionType.BUILD_SETTLEMENT, 2))
    game.execute(Action(Color.RED, ActionType.BUILD_ROAD, (1, 2)))
    game.execute(Action(Color.ORANGE, ActionType.BUILD_SETTLEMENT, 50))
    game.execute(Action(Color.ORANGE, ActionType.BUILD_ROAD, (50, 51)))


def _add_settlement(state, catan_map, color, key, resource, number, occupied):
    for tile in catan_map.land_tiles.values():
        if tile.resource == resource and tile.number == number:
            for nid in tile.nodes.values():
                if nid not in occupied:
                    state.board.buildings[nid] = (color, "SETTLEMENT")
                    state.buildings_by_color[color]["SETTLEMENT"].append(nid)
                    state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"] -= 1
                    state.player_state[f"{key}_VICTORY_POINTS"] += 1
                    state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] += 1
                    occupied.add(nid)
                    return
            break


def _add_city(state, catan_map, color, key, resource, number, occupied):
    for tile in catan_map.land_tiles.values():
        if tile.resource == resource and tile.number == number:
            for nid in tile.nodes.values():
                if nid not in occupied:
                    state.board.buildings[nid] = (color, "CITY")
                    state.buildings_by_color[color]["CITY"].append(nid)
                    state.player_state[f"{key}_CITIES_AVAILABLE"] -= 1
                    state.player_state[f"{key}_VICTORY_POINTS"] += 2
                    state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] += 2
                    occupied.add(nid)
                    return
            break


# ---------------------------------------------------------------------------
# PN-1: Kingmaker — Trade Partner Choice
# ---------------------------------------------------------------------------

def _setup_kingmaker_trade_partner_choice() -> Game:
    """
    3-player game: RED, BLUE, ORANGE. Turn order with seed=42: ORANGE, RED, BLUE.

    RED (VP 3): settlements near Wheat-11, Wood-5, Ore-9, Sheep-3.
      No Brick tiles, no port. Brick is RED's missing settlement/road resource.
      Note: the original spec said "stone(7)" — 7 is not a valid production number
      in Catan; substituted with Ore-9.

    BLUE (VP 6, second place): settlements near Wheat-6, Brick-8.
      Offer on the table: gives RED 1 Wheat in exchange for RED's 1 Ore.
      Wheat is less useful to RED who already produces it.

    ORANGE (VP 7, leader): settlements near Brick-10, Ore-4.
      Offer on the table: gives RED 1 Brick in exchange for RED's 1 Ore.
      Brick IS useful to RED — it is their missing resource.

    No active trade set in game state; both offers are described in the question.
    """
    pinned_tiles = [
        (WHEAT, 11),  # RED
        (WOOD, 5),    # RED
        (ORE, 9),     # RED  [stone-7 substituted — 7 is invalid]
        (SHEEP, 3),   # RED
        (WHEAT, 6),   # BLUE
        (BRICK, 8),   # BLUE
        (BRICK, 10),  # ORANGE
        (ORE, 4),     # ORANGE
    ]

    catan_map = _make_map_with_pinned_tiles(pinned_tiles, seed=42)
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE), SimplePlayer(Color.ORANGE)]
    game = Game(players, seed=42, catan_map=catan_map)

    _place_3p_initial(game)
    advance_to_play_turn(game)

    state = game.state
    red_key = player_key(state, Color.RED)
    blue_key = player_key(state, Color.BLUE)
    orange_key = player_key(state, Color.ORANGE)
    occupied = set(state.board.buildings.keys())

    # RED: 1 extra settlement near Wood-5 → 3 VP total
    _add_settlement(state, catan_map, Color.RED, red_key, WOOD, 5, occupied)

    # BLUE: 2 settlements (Wheat-6, Brick-8) + 1 city (Wheat-6) → 6 VP total
    # 2 initial(2) + 2 extra settlements(2) + 1 city(2) = 6 VP
    # settlements_available: 5-2-2=1  cities_available: 4-1=3
    _add_settlement(state, catan_map, Color.BLUE, blue_key, WHEAT, 6, occupied)
    _add_settlement(state, catan_map, Color.BLUE, blue_key, BRICK, 8, occupied)
    _add_city(state, catan_map, Color.BLUE, blue_key, WHEAT, 6, occupied)

    # ORANGE: 3 settlements (Brick-10 ×2, Ore-4) + 1 city (Brick-10) → 7 VP total
    # 2 initial(2) + 3 extra settlements(3) + 1 city(2) = 7 VP
    # settlements_available: 5-2-3=0  cities_available: 4-1=3
    _add_settlement(state, catan_map, Color.ORANGE, orange_key, BRICK, 10, occupied)
    _add_settlement(state, catan_map, Color.ORANGE, orange_key, ORE, 4, occupied)
    _add_settlement(state, catan_map, Color.ORANGE, orange_key, BRICK, 10, occupied)
    _add_city(state, catan_map, Color.ORANGE, orange_key, BRICK, 10, occupied)

    state.board.buildable_edges_cache = {}

    # RED: some ore to offer; no brick
    player_freqdeck_add(state, Color.RED, [0, 0, 1, 2, 1])   # 1 sheep, 2 wheat, 1 ore

    # BLUE: wheat and brick available to trade
    player_freqdeck_add(state, Color.BLUE, [0, 2, 0, 2, 1])  # 2 brick, 2 wheat, 1 ore

    # ORANGE: brick to offer
    player_freqdeck_add(state, Color.ORANGE, [0, 3, 0, 1, 1]) # 3 brick, 1 wheat, 1 ore

    return game


SCENARIO_KINGMAKER_TRADE_PARTNER_CHOICE = UnderstandingScenario(
    name="kingmaker_trade_partner_choice",
    category="nyx_a_kingmaker",
    description=(
        "Tests: anti-leader trade discipline under nyx_a_kingmaker. "
        "Two offers on the table: ORANGE (leader, 7VP) gives 1 Brick (RED's scarce resource); "
        "BLUE (second, 6VP) gives 1 Wheat (less useful — RED produces Wheat already). "
        "Tests whether Nyx declines the more useful offer from the leader and accepts "
        "the less useful one from second place to fund the challenger."
    ),
    setup=_setup_kingmaker_trade_partner_choice,
    perspective_color=Color.RED,
    persona="nyx_a_kingmaker",
    system_prompt="(overridden by nyx_a_kingmaker persona)",
    question=(
        "Two trade offers are on the table — you must choose one or neither:\n\n"
        "  - BLUE (6 VP, second place) will give you 1 Wheat in exchange for "
        "1 Ore from you.\n"
        "  - ORANGE (7 VP, current leader) will give you 1 Brick in exchange for "
        "1 Ore from you.\n\n"
        "You have no Brick tiles and no port access. Brick is your missing "
        "resource — you need it to build roads and settlements.\n\n"
        "Which offer do you accept, and why?"
    ),
    desired_answer=(
        "Declines ORANGE's offer (1 Brick for 1 Ore) despite it being the more "
        "useful trade — Brick is RED's missing resource. Accepts BLUE's offer "
        "(1 Wheat for 1 Ore) even though RED already produces Wheat.\n\n"
        "Reasoning: ORANGE is the current leader at 7 VP. Nyx's core rule is "
        "'Refuse all trades with the leader.' Any resource given to ORANGE — "
        "even in a fair exchange — helps their development pipeline. "
        "BLUE at 6 VP is second place. Trading with BLUE funds the challenger: "
        "BLUE gains ore (useful for development) and gives away wheat (readily "
        "available from their Wheat-6 tile). Nyx accepts the tactically suboptimal "
        "deal with BLUE to advance the strategic objective of preventing ORANGE from "
        "running away with the game."
    ),
)


# ---------------------------------------------------------------------------
# PN-2: Kingmaker — Fund Second Place
# ---------------------------------------------------------------------------

def _setup_kingmaker_fund_second_place() -> Game:
    """
    3-player game: RED, BLUE, ORANGE. Turn order with seed=42: ORANGE, RED, BLUE.

    RED (VP 3): settlements near Wheat-6, Wood-5, Ore-9, Sheep-3.
      No Brick tiles, no port. RED put out a public offer: 3 Ore for 2 Brick.
      Note: the original spec said "wheat(7)" — 7 is invalid; substituted with Wheat-6.

    BLUE (VP 6, second place): settlements near Brick-8, Wheat-11.
      Counter-offer: gives RED 1 Brick for 3 Ore (WORSE than RED's ask — half the brick).

    ORANGE (VP 7, leader): settlements near Brick-10, Ore-4.
      Counter-offer: gives RED 2 Brick for 2 Ore (BETTER than RED's ask — less stone).

    No active trade in game state; the negotiation context is in the question.
    """
    pinned_tiles = [
        (WHEAT, 6),   # RED  [wheat-7 substituted — 7 is invalid]
        (WOOD, 5),    # RED
        (ORE, 9),     # RED
        (SHEEP, 3),   # RED
        (BRICK, 8),   # BLUE
        (WHEAT, 11),  # BLUE
        (BRICK, 10),  # ORANGE
        (ORE, 4),     # ORANGE
    ]

    catan_map = _make_map_with_pinned_tiles(pinned_tiles, seed=42)
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE), SimplePlayer(Color.ORANGE)]
    game = Game(players, seed=42, catan_map=catan_map)

    _place_3p_initial(game)
    advance_to_play_turn(game)

    state = game.state
    red_key = player_key(state, Color.RED)
    blue_key = player_key(state, Color.BLUE)
    orange_key = player_key(state, Color.ORANGE)
    occupied = set(state.board.buildings.keys())

    # RED: 1 extra settlement near Wood-5 → 3 VP total
    _add_settlement(state, catan_map, Color.RED, red_key, WOOD, 5, occupied)

    # BLUE: 2 settlements + 1 city → 6 VP total
    _add_settlement(state, catan_map, Color.BLUE, blue_key, BRICK, 8, occupied)
    _add_settlement(state, catan_map, Color.BLUE, blue_key, WHEAT, 11, occupied)
    _add_city(state, catan_map, Color.BLUE, blue_key, BRICK, 8, occupied)

    # ORANGE: 3 settlements + 1 city → 7 VP total
    _add_settlement(state, catan_map, Color.ORANGE, orange_key, BRICK, 10, occupied)
    _add_settlement(state, catan_map, Color.ORANGE, orange_key, ORE, 4, occupied)
    _add_settlement(state, catan_map, Color.ORANGE, orange_key, BRICK, 10, occupied)
    _add_city(state, catan_map, Color.ORANGE, orange_key, BRICK, 10, occupied)

    state.board.buildable_edges_cache = {}

    # RED: plenty of ore to trade; no brick
    player_freqdeck_add(state, Color.RED, [0, 0, 1, 1, 5])   # 1 sheep, 1 wheat, 5 ore

    # BLUE: brick available for counter-offer
    player_freqdeck_add(state, Color.BLUE, [0, 3, 0, 1, 0])  # 3 brick, 1 wheat

    # ORANGE: brick available for counter-offer
    player_freqdeck_add(state, Color.ORANGE, [0, 4, 0, 1, 1]) # 4 brick, 1 wheat, 1 ore

    return game


SCENARIO_KINGMAKER_FUND_SECOND_PLACE = UnderstandingScenario(
    name="kingmaker_fund_second_place",
    category="nyx_a_kingmaker",
    description=(
        "Tests: anti-leader trade discipline (counter-offer selection) under nyx_a_kingmaker. "
        "RED asked for 2 Brick for 3 Ore. Two counter-offers arrived: "
        "BLUE (second, 6VP) counter: 1 Brick for 3 Ore — worse than asked (half the brick). "
        "ORANGE (leader, 7VP) counter: 2 Brick for 2 Ore — better than asked (less stone). "
        "Tests whether Nyx takes the objectively worse deal from second place, "
        "effectively subsidising the challenger over the leader."
    ),
    setup=_setup_kingmaker_fund_second_place,
    perspective_color=Color.RED,
    persona="nyx_a_kingmaker",
    system_prompt="(overridden by nyx_a_kingmaker persona)",
    question=(
        "You put out a public offer: 3 Ore in exchange for 2 Brick. "
        "You need Brick — you have no Brick tiles and no port.\n\n"
        "Two counter-offers have arrived:\n\n"
        "  - BLUE (6 VP, second place): will give you 1 Brick for your 3 Ore. "
        "That is half the Brick you asked for, at the same Ore cost.\n"
        "  - ORANGE (7 VP, current leader): will give you 2 Brick for only 2 Ore. "
        "That is the full Brick you need at a lower Ore cost.\n\n"
        "Which counter-offer do you accept, and why?"
    ),
    desired_answer=(
        "Accepts BLUE's counter (the worse deal: 1 Brick for 3 Ore) and declines "
        "ORANGE's counter (the better deal: 2 Brick for 2 Ore).\n\n"
        "Reasoning: ORANGE's offer is objectively superior — RED gets the same 2 Brick "
        "while giving up only 2 Ore instead of 3. But ORANGE is the current leader at "
        "7 VP. Nyx refuses all trades with the leader, regardless of terms.\n\n"
        "By accepting BLUE's counter, RED is effectively subsidising second place: "
        "BLUE receives 3 Ore for only 1 Brick — a highly favorable rate that funds "
        "BLUE's city-building and development. The short-term cost to RED (1 fewer "
        "Brick, 1 extra Ore spent) is the deliberate price of empowering the challenger "
        "to threaten ORANGE's lead. Nyx would rather give away value to second place "
        "than make any deal — even a good one — that involves the current leader."
    ),
)


# ---------------------------------------------------------------------------
# Registered scenarios
# ---------------------------------------------------------------------------

SCENARIOS: list[UnderstandingScenario] = [
    SCENARIO_MAKING_ALLIANCE,
    SCENARIO_ALLIANCE_LOYALTY_TEST,
    SCENARIO_OFFER_WITH_PROMISE,
    SCENARIO_RISK_ADVERSITY_TRADE_PARTNER,
    SCENARIO_ISOLATIONIST_FORCED_TRADE,
    SCENARIO_ROBBER_TARGETING,
    SCENARIO_SURPLUS_TRADE_ASSESSMENT,
    SCENARIO_NEGOTIATION_POWER_ASSESSMENT,
    SCENARIO_KINGMAKER_TRADE_PARTNER_CHOICE,
    SCENARIO_KINGMAKER_FUND_SECOND_PLACE,
]
