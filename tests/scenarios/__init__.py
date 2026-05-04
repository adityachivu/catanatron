"""
Scenario definitions for the LLM understanding harness.

Import UnderstandingScenario, _make_base_game, and _make_map_with_pinned_tiles
from here when defining new scenarios in any scenario file.
"""
from __future__ import annotations

import sys
import random as _random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CATANATRON_SRC = _PROJECT_ROOT / "catanatron"
for _p in [str(_PROJECT_ROOT), str(_CATANATRON_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from catanatron.game import Game
from catanatron.models.enums import FastResource
from catanatron.models.map import BASE_MAP_TEMPLATE, CatanMap, initialize_tiles
from catanatron.models.player import Color, SimplePlayer
from catanatron.state_functions import player_freqdeck_add, player_key

from tests.utils import advance_to_play_turn, build_initial_placements


# ---------------------------------------------------------------------------
# Shared board-building helpers
# ---------------------------------------------------------------------------

def _make_map_with_pinned_tiles(
    pinned_tiles: list[tuple[FastResource, int]],
    seed: int = 42,
) -> CatanMap:
    """
    Build a CatanMap that guarantees specific (resource, number) pairs exist.

    Each entry in ``pinned_tiles`` is a ``(FastResource, dice_number)`` pair
    such as ``(WHEAT, 6)`` or ``(ORE, 8)``.  Those tiles are placed at the
    first positions in the topology; the rest of the land tiles are filled from
    the standard pool, randomised with ``seed``.

    Constraints
    -----------
    - Pinned resources must be non-desert (not ``None``).
    - Each resource/number must appear in the standard BASE_MAP pool at least
      as many times as it is pinned (otherwise ``list.remove`` will raise).
    - Uses ``number_placement="random"``, so the official-spiral adjacency
      rule (no two red numbers touching) is not enforced.
    """
    if any(r is None for r, _ in pinned_tiles):
        raise ValueError("pinned_tiles entries must not be the desert (None resource)")

    resource_pool = list(BASE_MAP_TEMPLATE.tile_resources)
    number_pool = list(BASE_MAP_TEMPLATE.numbers)

    for resource, number in pinned_tiles:
        resource_pool.remove(resource)
        number_pool.remove(number)

    _random.seed(seed)
    remaining_resources = _random.sample(resource_pool, len(resource_pool))
    remaining_numbers = _random.sample(number_pool, len(number_pool))

    # initialize_tiles pops from the END of each list, so items placed first
    # in topology order are the last items in the list.  Reverse pinned_tiles
    # so that pinned_tiles[0] lands at topology position 0, etc.
    final_resources = remaining_resources + [r for r, _ in reversed(pinned_tiles)]
    final_numbers = remaining_numbers + [n for _, n in reversed(pinned_tiles)]

    tiles = initialize_tiles(
        BASE_MAP_TEMPLATE,
        shuffled_tile_resources_param=final_resources,
        shuffled_numbers_param=final_numbers,
        number_placement="random",
    )
    return CatanMap.from_tiles(tiles)


def _make_base_game(
    seed: int = 42,
    pinned_tiles: Optional[list[tuple[FastResource, int]]] = None,
) -> Game:
    """
    Create a standard 2-player game, run through initial placements and
    advance into the first proper play turn (dice already rolled).

    Board seed 42 is used throughout so node/hex positions are stable.
    Player 0 = RED,  Player 1 = BLUE.
    Initial placements (from tests/utils defaults):
      RED  → settlements at nodes 0 and 2, roads on edges (0,1) and (1,2)
      BLUE → settlements at nodes 24 and 26, roads on edges (24,25) and (25,26)

    Parameters
    ----------
    seed:
        Random seed controlling board layout and player seating order.
    pinned_tiles:
        Optional list of ``(resource, dice_number)`` pairs that are
        guaranteed to exist somewhere on the board.  Example::

            from catanatron.models.enums import WHEAT, ORE
            game = _make_base_game(pinned_tiles=[(WHEAT, 6), (ORE, 8)])

        The remaining tiles are filled from the standard pool and randomised
        with ``seed``.  When ``None`` (default), the board is generated
        entirely from the seeded random, exactly as before.
    """
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]

    if pinned_tiles:
        catan_map = _make_map_with_pinned_tiles(pinned_tiles, seed)
        game = Game(players, seed=seed, catan_map=catan_map)
    else:
        game = Game(players, seed=seed)

    build_initial_placements(game)
    advance_to_play_turn(game)
    return game


# ---------------------------------------------------------------------------
# Scenario dataclass
# ---------------------------------------------------------------------------

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
    system_prompt:   ◀◀◀ PROMPT FIELD — edit to change the agent's instruction/role
        Task-specific instruction to the agent.
        Replaces CATAN_SYSTEM_PROMPT from base.py for this test.
        Ignored when `persona` is set (persona.system_prompt is used instead).
    question:        ◀◀◀ PROMPT FIELD — edit to change what is asked
        The free-form question appended after the STRUCTURED_STATE_JSON block.
    persona:
        Optional persona name (e.g. "default", "aggressive").  When set,
        load_persona() is called and its system_prompt replaces system_prompt,
        and the memory-tools section is injected into the prompt (matching
        BaseLLMPlayer._build_prompt) if the persona has memory enabled.
    desired_answer:
        Optional reference answer shown in results output for easy comparison.
    """

    name: str
    category: str
    description: str
    setup: Callable[[], Game]
    perspective_color: Color
    system_prompt: str   # ◀ PROMPT A: the agent's role / instructions
    question: str        # ◀ PROMPT B: what you ask the agent about the state
    persona: Optional[str] = None  # ◀ PROMPT C: persona name — overrides system_prompt and adds memory section
    desired_answer: Optional[str] = None  # ◀ expected/ideal answer for reference
