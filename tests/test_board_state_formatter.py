import json

import pytest

from catanatron.game import Game
from catanatron.models.map import build_map
from catanatron.models.player import Color, RandomPlayer, SimplePlayer
from catanatron.players.llm.board_state_formatter import BoardStateFormatter
from tests.utils import build_initial_placements


@pytest.fixture
def two_player_game():
    players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
    return Game(players, seed=42)


@pytest.fixture
def game_after_initial_placement(two_player_game):
    game = two_player_game
    build_initial_placements(game)
    return game


@pytest.fixture
def mini_game():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    return Game(players, seed=42, catan_map=build_map("MINI"))


class TestExtractTiles:
    def test_all_land_tiles_present(self, two_player_game):
        tiles = BoardStateFormatter.extract_tiles(two_player_game)

        assert len(tiles) == 19
        desert_tiles = [tile for tile in tiles if tile["resource"] == "DESERT"]
        assert len(desert_tiles) == 1
        assert desert_tiles[0]["number"] is None

    def test_tile_structure(self, two_player_game):
        tiles = BoardStateFormatter.extract_tiles(two_player_game)

        for tile in tiles:
            assert set(tile.keys()) == {"tile_id", "coord", "resource", "number", "nodes"}
            assert len(tile["coord"]) == 3
            assert all(isinstance(value, int) for value in tile["coord"])
            assert len(tile["nodes"]) == 6
            assert all(isinstance(node_id, int) for node_id in tile["nodes"])

    def test_tile_ids_unique(self, two_player_game):
        tiles = BoardStateFormatter.extract_tiles(two_player_game)
        tile_ids = [tile["tile_id"] for tile in tiles]

        assert len(tile_ids) == len(set(tile_ids))
        assert tile_ids == sorted(tile_ids)

    def test_mini_map_tiles(self, mini_game):
        tiles = BoardStateFormatter.extract_tiles(mini_game)

        assert len(tiles) == 7
        resources = sorted(tile["resource"] for tile in tiles)
        assert resources == sorted(["WOOD", "DESERT", "BRICK", "SHEEP", "WHEAT", "WHEAT", "ORE"])


class TestExtractNodes:
    def test_empty_board_all_null(self, two_player_game):
        nodes = BoardStateFormatter.extract_nodes(two_player_game)

        assert len(nodes) == 54
        assert all(node["building"] is None for node in nodes.values())
        assert all(node["owner"] is None for node in nodes.values())

    def test_after_initial_placement(self, game_after_initial_placement):
        nodes = BoardStateFormatter.extract_nodes(game_after_initial_placement)

        assert nodes["0"] == {"building": "SETTLEMENT", "owner": "RED"}
        assert nodes["2"] == {"building": "SETTLEMENT", "owner": "RED"}
        assert nodes["24"] == {"building": "SETTLEMENT", "owner": "BLUE"}
        assert nodes["26"] == {"building": "SETTLEMENT", "owner": "BLUE"}
        assert nodes["1"] == {"building": None, "owner": None}

    def test_city_upgrade(self, game_after_initial_placement):
        game_after_initial_placement.state.board.build_city(Color.RED, 0)

        nodes = BoardStateFormatter.extract_nodes(game_after_initial_placement)

        assert nodes["0"] == {"building": "CITY", "owner": "RED"}

    def test_node_count_matches_map(self, two_player_game):
        nodes = BoardStateFormatter.extract_nodes(two_player_game)

        assert len(nodes) == len(two_player_game.state.board.map.land_nodes)


class TestExtractRoads:
    def test_empty_board_no_roads(self, two_player_game):
        roads = BoardStateFormatter.extract_roads(two_player_game)

        assert roads == []

    def test_after_initial_placement(self, game_after_initial_placement):
        roads = BoardStateFormatter.extract_roads(game_after_initial_placement)

        assert len(roads) == 4
        assert {tuple(road["edge"]) for road in roads} == {
            (0, 1),
            (1, 2),
            (24, 25),
            (25, 26),
        }
        assert {tuple(road["edge"]): road["owner"] for road in roads} == {
            (0, 1): "RED",
            (1, 2): "RED",
            (24, 25): "BLUE",
            (25, 26): "BLUE",
        }

    def test_no_duplicates(self, game_after_initial_placement):
        roads = BoardStateFormatter.extract_roads(game_after_initial_placement)
        edges = [tuple(road["edge"]) for road in roads]

        assert len(edges) == len(set(edges))
        assert all(edge[0] < edge[1] for edge in edges)


class TestExtractPorts:
    def test_base_map_nine_ports(self, two_player_game):
        ports = BoardStateFormatter.extract_ports(two_player_game)

        assert len(ports) == 9

    def test_port_structure(self, two_player_game):
        ports = BoardStateFormatter.extract_ports(two_player_game)

        for port in ports:
            assert set(port.keys()) == {"type", "nodes"}
            assert port["type"] == "3:1" or port["type"].startswith("2:1_")
            assert len(port["nodes"]) == 2
            assert all(isinstance(node_id, int) for node_id in port["nodes"])

    def test_port_nodes_are_valid(self, two_player_game):
        ports = BoardStateFormatter.extract_ports(two_player_game)

        for port in ports:
            assert all(0 <= node_id < 54 for node_id in port["nodes"])

    def test_mini_map_has_no_ports(self, mini_game):
        assert BoardStateFormatter.extract_ports(mini_game) == []


class TestExtractRobber:
    def test_initial_robber_on_desert(self, two_player_game):
        robber_coord = BoardStateFormatter.extract_robber(two_player_game)
        desert_tile = next(
            tile for tile in BoardStateFormatter.extract_tiles(two_player_game)
            if tile["resource"] == "DESERT"
        )

        assert robber_coord == desert_tile["coord"]


class TestFormatBoardState:
    def test_all_flags_true(self, two_player_game):
        result = BoardStateFormatter.format_board_state(two_player_game)

        assert set(result.keys()) == {"tiles", "nodes", "roads", "ports", "robber_coord"}

    def test_all_flags_false(self, two_player_game):
        result = BoardStateFormatter.format_board_state(
            two_player_game,
            include_tiles=False,
            include_nodes=False,
            include_roads=False,
            include_ports=False,
            include_robber=False,
        )

        assert result == {}

    @pytest.mark.parametrize(
        ("flag_kwargs", "expected_keys"),
        [
            ({"include_tiles": True, "include_nodes": False, "include_roads": False, "include_ports": False, "include_robber": False}, {"tiles"}),
            ({"include_tiles": False, "include_nodes": True, "include_roads": False, "include_ports": False, "include_robber": False}, {"nodes"}),
            ({"include_tiles": False, "include_nodes": False, "include_roads": True, "include_ports": False, "include_robber": False}, {"roads"}),
            ({"include_tiles": False, "include_nodes": False, "include_roads": False, "include_ports": True, "include_robber": False}, {"ports"}),
            ({"include_tiles": False, "include_nodes": False, "include_roads": False, "include_ports": False, "include_robber": True}, {"robber_coord"}),
            ({"include_tiles": True, "include_nodes": False, "include_roads": False, "include_ports": False, "include_robber": True}, {"tiles", "robber_coord"}),
        ],
    )
    def test_flags_control_sections(self, two_player_game, flag_kwargs, expected_keys):
        result = BoardStateFormatter.format_board_state(two_player_game, **flag_kwargs)

        assert set(result.keys()) == expected_keys


class TestBuildBoardStatePrompt:
    def test_prompt_delimiters(self, two_player_game):
        prompt = BoardStateFormatter.build_board_state_prompt(two_player_game)

        assert prompt.startswith("=== BOARD_STATE ===\n")
        assert prompt.endswith("\n=== END_BOARD_STATE ===")

    def test_prompt_is_valid_json_between_delimiters(self, two_player_game):
        prompt = BoardStateFormatter.build_board_state_prompt(two_player_game)
        json_str = prompt[len("=== BOARD_STATE ===\n") : -len("\n=== END_BOARD_STATE ===")]
        parsed = json.loads(json_str)

        assert isinstance(parsed, dict)
        assert set(parsed.keys()) == {"tiles", "nodes", "roads", "ports", "robber_coord"}

    def test_empty_prompt_when_all_flags_false(self, two_player_game):
        prompt = BoardStateFormatter.build_board_state_prompt(
            two_player_game,
            include_tiles=False,
            include_nodes=False,
            include_roads=False,
            include_ports=False,
            include_robber=False,
        )

        assert prompt == ""

    def test_flags_propagate_to_json_keys(self, two_player_game):
        prompt = BoardStateFormatter.build_board_state_prompt(
            two_player_game,
            include_tiles=False,
            include_nodes=True,
            include_roads=False,
            include_ports=False,
            include_robber=True,
        )
        json_str = prompt[len("=== BOARD_STATE ===\n") : -len("\n=== END_BOARD_STATE ===")]
        parsed = json.loads(json_str)

        assert set(parsed.keys()) == {"nodes", "robber_coord"}

    def test_prompt_after_initial_placement(self, game_after_initial_placement):
        prompt = BoardStateFormatter.build_board_state_prompt(
            game_after_initial_placement,
            include_tiles=False,
            include_nodes=True,
            include_roads=True,
            include_ports=False,
            include_robber=False,
        )
        json_str = prompt[len("=== BOARD_STATE ===\n") : -len("\n=== END_BOARD_STATE ===")]
        parsed = json.loads(json_str)

        assert parsed["nodes"]["0"] == {"building": "SETTLEMENT", "owner": "RED"}
        assert parsed["nodes"]["24"] == {"building": "SETTLEMENT", "owner": "BLUE"}
        assert {tuple(road["edge"]) for road in parsed["roads"]} == {
            (0, 1),
            (1, 2),
            (24, 25),
            (25, 26),
        }
