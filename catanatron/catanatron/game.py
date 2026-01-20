"""
Contains Game class which is a thin-wrapper around the State class.
"""

import uuid
import random
import sys
from typing import Sequence, Union, Optional, List

import logfire

from catanatron.models.actions import generate_playable_actions
from catanatron.models.enums import Action, ActionPrompt, ActionRecord, ActionType
from catanatron.state import State
from catanatron.apply_action import apply_action
from catanatron.state_functions import player_key, player_has_rolled
from catanatron.models.map import CatanMap
from catanatron.models.player import Color, Player

# To timeout RandomRobots from getting stuck...
TURNS_LIMIT = 1000


def is_valid_action(playable_actions, state: State, action: Action) -> bool:
    """True if its a valid action right now. An action is valid
    if its in playable_actions or if its a OFFER_TRADE in the right time."""
    if action.action_type == ActionType.OFFER_TRADE:
        return (
            state.current_color() == action.color
            and state.current_prompt == ActionPrompt.PLAY_TURN
            and player_has_rolled(state, action.color)
            and is_valid_trade(action.value)
        )

    return action in playable_actions


def is_valid_trade(action_value):
    """Checks the value of a OFFER_TRADE does not
    give away resources or trade matching resources.
    """
    offering = action_value[:5]
    asking = action_value[5:]
    if sum(offering) == 0 or sum(asking) == 0:
        return False  # cant give away cards

    for i, j in zip(offering, asking):
        if i > 0 and j > 0:
            return False  # cant trade same resources
    return True


class GameAccumulator:
    """Interface to hook into different game lifecycle events.

    Useful to compute aggregate statistics, log information, etc...
    """

    def __init__(*args, **kwargs):
        pass

    def before(self, game):
        """
        Called when the game is created, no actions have
        been taken by players yet, but the board is decided.
        """
        pass

    def step(self, game_before_action, action):
        """
        Called after each action taken by a player.
        Game should be right before action is taken.
        """
        pass

    def after(self, game):
        """
        Called when the game is finished.

        Check game.winning_color() to see if the game
        actually finished or exceeded turn limit (is None).
        """
        pass


class Game:
    """
    Initializes a map, decides player seating order, and exposes two main
    methods for executing the game (play and play_tick; to advance until
    completion or just by one decision by a player respectively).

    Attributes:
        state (State): Current game state.
        playable_actions (List[Action]): List of playable actions by current player.
    """

    def __init__(
        self,
        players: Sequence[Player],
        seed: Optional[int] = None,
        discard_limit: int = 7,
        vps_to_win: int = 10,
        catan_map: Optional[CatanMap] = None,
        initialize: bool = True,
    ):
        """Creates a game (doesn't run it).

        Args:
            players (List[Player]): list of players, should be at most 4.
            seed (int, optional): Random seed to use (for reproducing games). Defaults to None.
            discard_limit (int, optional): Discard limit to use. Defaults to 7.
            vps_to_win (int, optional): Victory Points needed to win. Defaults to 10.
            catan_map (CatanMap, optional): Map to use. Defaults to None.
            initialize (bool, optional): Whether to initialize. Defaults to True.
        """
        # Initialize turn span tracking (always, for consistency)
        self._current_turn_span = None
        self._current_turn_start_index = None
        self._current_turn_actions: List[str] = []

        if initialize:
            self.seed = seed if seed is not None else random.randrange(sys.maxsize)
            random.seed(self.seed)

            self.id = str(uuid.uuid4())
            self.vps_to_win = vps_to_win
            self.state = State(players, catan_map, discard_limit=discard_limit)
            self.playable_actions = generate_playable_actions(self.state)

    def play(self, accumulators=[], decide_fn=None):
        """Executes game until a player wins or exceeded TURNS_LIMIT.

        Args:
            accumulators (list[Accumulator], optional): list of Accumulator classes to use.
                Their .consume method will be called with every action, and
                their .finalize method will be called when the game ends (if it ends)
                Defaults to [].
            decide_fn (function, optional): Function to overwrite current player's decision with.
                Defaults to None.
        Returns:
            Color: winning color or None if game exceeded TURNS_LIMIT
        """
        for accumulator in accumulators:
            accumulator.before(self)
        while self.winning_color() is None and self.state.num_turns < TURNS_LIMIT:
            self.play_tick(decide_fn=decide_fn, accumulators=accumulators)
        
        # End any active turn span on game termination (e.g., if game ended mid-turn)
        self._end_turn_span()
        
        for accumulator in accumulators:
            accumulator.after(self)
        return self.winning_color()

    def _start_turn_span(self):
        """Start a new logfire span for the current player's turn."""
        player = self.state.current_player()
        self._current_turn_span = logfire.span(
            "catanatron.player_turn",
            player_color=player.color.value,
            turn_number=self.state.num_turns,
            player_index=self.state.current_turn_index,
            game_id=self.id,
        )
        self._current_turn_span.__enter__()
        self._current_turn_start_index = self.state.current_turn_index
        self._current_turn_actions = []

    def _end_turn_span(self):
        """End the current turn span and record final attributes."""
        if self._current_turn_span is not None:
            # Set span attributes for actions taken
            self._current_turn_span.set_attribute("num_actions", len(self._current_turn_actions))
            self._current_turn_span.set_attribute("actions_taken", self._current_turn_actions)
            self._current_turn_span.__exit__(None, None, None)
            self._current_turn_span = None
            self._current_turn_start_index = None
            self._current_turn_actions = []

    def play_tick(self, decide_fn=None, accumulators=[]):
        """Advances game by one ply (player decision).

        Args:
            decide_fn (function, optional): Function to overwrite current player's decision with.
                Defaults to None.

        Returns:
            ActionRecord: representing the executed action
        """
        # Check if we need to start a new turn span (skip during initial build phase)
        if not self.state.is_initial_build_phase:
            # Detect if this is a new turn (turn index changed or first turn after initial build)
            if self._current_turn_start_index != self.state.current_turn_index:
                # End any existing span from previous turn
                self._end_turn_span()
                # Start new span for this turn
                self._start_turn_span()

        # Ask Player for action
        player = self.state.current_player()
        action = (
            decide_fn(player, self, self.playable_actions)
            if decide_fn is not None
            else player.decide(self, self.playable_actions)
        )

        # Call accumulator.step here, because we want game_before_action, action
        if len(accumulators) > 0:
            for accumulator in accumulators:
                accumulator.step(self, action)

        # Apply Action, and do Move Generation
        return self.execute(action)

    def execute(
        self,
        action: Action,
        validate_action: bool = True,
        action_record: ActionRecord = None,
    ) -> ActionRecord:
        """Internal call that carries out decided action by player"""
        if validate_action and not is_valid_action(
            self.playable_actions, self.state, action
        ):
            raise ValueError(
                f"{action} not playable right now. playable_actions={self.playable_actions}"
            )

        # Track action in current turn span
        if self._current_turn_span is not None:
            self._current_turn_actions.append(action.action_type.name)

        action_record = apply_action(self.state, action, action_record)
        self.playable_actions = generate_playable_actions(self.state)

        # End turn span when END_TURN action is executed
        if action.action_type == ActionType.END_TURN:
            self._end_turn_span()

        return action_record

    def winning_color(self) -> Union[Color, None]:
        """Gets winning color

        Returns:
            Union[Color, None]: Might be None if game truncated by TURNS_LIMIT
        """
        result = None
        for color in self.state.colors:
            key = player_key(self.state, color)
            if (
                self.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]
                >= self.vps_to_win
            ):
                result = color

        return result

    def copy(self) -> "Game":
        """Creates a copy of this Game, that can be modified without
        repercusions on this one (useful for simulations).

        Returns:
            Game: Game copy.
        """
        game_copy = Game(players=[], initialize=False)
        game_copy.seed = self.seed
        game_copy.id = self.id
        game_copy.vps_to_win = self.vps_to_win
        game_copy.state = self.state.copy()
        game_copy.playable_actions = self.playable_actions
        return game_copy
