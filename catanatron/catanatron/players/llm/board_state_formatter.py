"""
Board state formatting utilities for LLM-readable board topology.

Extracts the full board layout — tiles, nodes (with buildings), roads,
ports, and robber location — from a Game object and formats it into
a structured JSON block suitable for inclusion in LLM prompts.

Each section can be toggled independently via boolean flags so that
callers can control what board information is included at different
game phases.
"""

import json
from typing import Any, Dict, List, Tuple

from catanatron.game import Game
from catanatron.models.map import PORT_DIRECTION_TO_NODEREFS, Port


class BoardStateFormatter:
    """
    Utility class for extracting and formatting full board topology.

    All methods are static, matching the convention used by StateFormatter.
    Outputs are deterministically sorted to ensure stable prompts across
    runs with the same seed.
    """

    # ------------------------------------------------------------------
    # Individual extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_tiles(game: Game) -> List[Dict[str, Any]]:
        """
        Extract all land tiles from the board.

        Returns a list of tile dicts sorted by tile_id, each containing:
        - tile_id: Unique integer identifier
        - coord: Cube coordinate as a 3-element list [x, y, z]
        - resource: Resource string ("WOOD", "BRICK", …) or "DESERT"
        - number: Number token (2–12) or null for desert
        - nodes: Sorted list of the 6 node IDs surrounding this tile
        """
        board = game.state.board
        tiles = []

        for coord, tile in board.map.land_tiles.items():
            tiles.append(
                {
                    "tile_id": tile.id,
                    "coord": list(coord),
                    "resource": tile.resource if tile.resource is not None else "DESERT",
                    "number": tile.number,
                    "nodes": sorted(tile.nodes.values()),
                }
            )

        tiles.sort(key=lambda t: t["tile_id"])
        return tiles

    @staticmethod
    def extract_nodes(game: Game) -> Dict[str, Dict[str, Any]]:
        """
        Extract all land nodes with their building status.

        Returns a dict keyed by string node ID (for JSON compatibility).
        Each value contains:
        - building: "SETTLEMENT", "CITY", or null
        - owner: Color string ("RED", "BLUE", …) or null
        """
        board = game.state.board
        nodes: Dict[str, Dict[str, Any]] = {}

        for node_id in sorted(board.map.land_nodes):
            building_info = board.buildings.get(node_id)
            if building_info is not None:
                color, building_type = building_info
                nodes[str(node_id)] = {
                    "building": building_type,
                    "owner": color.value,
                }
            else:
                nodes[str(node_id)] = {
                    "building": None,
                    "owner": None,
                }

        return nodes

    @staticmethod
    def extract_roads(game: Game) -> List[Dict[str, Any]]:
        """
        Extract all roads on the board (deduplicated).

        Since the board stores both (a,b) and (b,a) for each road,
        we only include edges where a < b.

        Returns a sorted list of road dicts, each containing:
        - edge: [node_a, node_b] with node_a < node_b
        - owner: Color string
        """
        board = game.state.board
        roads: List[Dict[str, Any]] = []
        seen: set = set()

        for edge, color in board.roads.items():
            normalized = tuple(sorted(edge))
            if normalized not in seen:
                seen.add(normalized)
                roads.append(
                    {
                        "edge": list(normalized),
                        "owner": color.value,
                    }
                )

        roads.sort(key=lambda r: tuple(r["edge"]))
        return roads

    @staticmethod
    def extract_ports(game: Game) -> List[Dict[str, Any]]:
        """
        Extract all ports from the board.

        Returns a sorted list of port dicts, each containing:
        - type: "3:1" for general ports, "2:1_RESOURCE" for resource ports
        - nodes: The two land-facing node IDs for this port
        """
        board = game.state.board
        ports: List[Dict[str, Any]] = []

        for port in board.map.ports_by_id.values():
            # Determine port type string
            if port.resource is None:
                port_type = "3:1"
            else:
                port_type = f"2:1_{port.resource}"

            # Get the two land-facing node refs for this port direction
            a_noderef, b_noderef = PORT_DIRECTION_TO_NODEREFS[port.direction]
            node_a = port.nodes[a_noderef]
            node_b = port.nodes[b_noderef]

            ports.append(
                {
                    "type": port_type,
                    "nodes": sorted([node_a, node_b]),
                }
            )

        ports.sort(key=lambda p: p["nodes"][0] if p["nodes"] else 0)
        return ports

    @staticmethod
    def extract_robber(game: Game) -> List[int]:
        """
        Extract the robber's current cube coordinate.

        Returns a 3-element list [x, y, z].
        """
        return list(game.state.board.robber_coordinate)

    # ------------------------------------------------------------------
    # Composite formatting
    # ------------------------------------------------------------------

    @staticmethod
    def format_board_state(
        game: Game,
        *,
        include_tiles: bool = True,
        include_nodes: bool = True,
        include_roads: bool = True,
        include_ports: bool = True,
        include_robber: bool = True,
    ) -> Dict[str, Any]:
        """
        Build a composite dict of board state, including only flagged sections.

        Args:
            game: Current game instance.
            include_tiles: Include tile layout (resource + number tokens).
            include_nodes: Include all nodes with building/owner info.
            include_roads: Include all placed roads.
            include_ports: Include port locations and types.
            include_robber: Include the robber coordinate.

        Returns:
            Dict with only the requested keys. Keys are:
            "tiles", "nodes", "roads", "ports", "robber_coord".
        """
        result: Dict[str, Any] = {}

        if include_tiles:
            result["tiles"] = BoardStateFormatter.extract_tiles(game)
        if include_nodes:
            result["nodes"] = BoardStateFormatter.extract_nodes(game)
        if include_roads:
            result["roads"] = BoardStateFormatter.extract_roads(game)
        if include_ports:
            result["ports"] = BoardStateFormatter.extract_ports(game)
        if include_robber:
            result["robber_coord"] = BoardStateFormatter.extract_robber(game)

        return result

    @staticmethod
    def build_board_state_prompt(
        game: Game,
        *,
        include_tiles: bool = True,
        include_nodes: bool = True,
        include_roads: bool = True,
        include_ports: bool = True,
        include_robber: bool = True,
    ) -> str:
        """
        Build the full prompt block with delimiters.

        Calls format_board_state with the given flags, serialises to JSON,
        and wraps in === BOARD_STATE === / === END_BOARD_STATE === markers.

        Returns an empty string if all flags are False.
        """
        data = BoardStateFormatter.format_board_state(
            game,
            include_tiles=include_tiles,
            include_nodes=include_nodes,
            include_roads=include_roads,
            include_ports=include_ports,
            include_robber=include_robber,
        )

        if not data:
            return ""

        json_str = json.dumps(data, indent=2, default=str)
        return f"=== BOARD_STATE ===\n{json_str}\n=== END_BOARD_STATE ==="
