"""Turn raw game logs into normalized :class:`AgentGameTrace` objects.

The current logs are a **Logfire columnar export** of the Gemini chat spans —
one JSON file (e.g. ``outputs/gemini_chat_spans.json``) shaped as
``{"columns": [...], "rows": [ {<span>}, ... ]}``. Every row is one LLM call
(``span_name`` == ``"chat <model>"``) with these useful attributes:

    attributes["gen_ai.input.messages"]   # the prompt: system + user parts
    attributes["gen_ai.output.messages"]  # the response: tool_call parts

The logs are "badly formatted" in that there is no first-class player/action
field. We recover per-player attribution from the prompt text itself:

    - The **system** message embeds the persona identity
      (e.g. ``You are Cassio-A, "The Charmer."``) -> which persona.
    - The **user** message starts with ``You are: <COLOR>`` and ``Turn: <N>``
      -> which player and which turn.

Reasoning is pulled from the model's ``final_result`` tool call
(``arguments.reasoning``); strategic intent is also captured from
``write_memory`` calls. Negotiations, when present, would appear as spans whose
prompt contains ``MESSAGING PHASE`` / ``TRADE FINALIZATION`` — this export
happens to contain none, so ``negotiations`` comes back empty and the deception
judge runs purely on reasoning, which is exactly what we want for now.

We emit one :class:`AgentGameTrace` per (game, player color). Everything
downstream (both judges + the CLI) already works against that type.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llm_judge.records import (
    AgentGameTrace,
    Decision,
    Negotiation,
    NegotiationMessage,
    Persona,
)

# --- personas dir, used only to recover a canonical persona name (best effort)
_PERSONAS_DIR = (
    Path(__file__).resolve().parent.parent
    / "catanatron"
    / "catanatron"
    / "players"
    / "llm"
    / "personas"
)

# "You are Cassio-A, "The Charmer.""  ->  identity name + title
_IDENTITY_RE = re.compile(r'You are ([A-Z][\w\-]+),?\s*"([^"]+)"')
_COLOR_RE = re.compile(r"You are:\s*([A-Z]+)")
_TURN_RE = re.compile(r"Turn:\s*(\d+)")

# --- negotiation span markers (present only when a game had negotiations) ---
_NEG_SPEAKER_RE = re.compile(r"You are (\w+) in a trade negotiation")
_NEG_INITIATOR_RE = re.compile(r"Initiated by:\s*(\w+)")
_NEG_PARTICIPANTS_RE = re.compile(r"Participants:\s*\[([^\]]*)\]")
_NEG_TURN_RE = re.compile(r'"turn":\s*(\d+)')
_NEG_FINALIZE_RE = re.compile(r"You are (\w+)\. The negotiation messaging")
_RESOURCES = ["wood", "brick", "sheep", "wheat", "ore"]


# --------------------------------------------------------------------------- #
# Low-level span helpers                                                       #
# --------------------------------------------------------------------------- #
def _message_text(messages: List[dict], role: Optional[str] = None) -> str:
    """Concatenate the text parts of the given messages, optionally by role."""
    out: List[str] = []
    for m in messages:
        if role is not None and m.get("role") != role:
            continue
        for p in m.get("parts", []):
            if p.get("type") == "text" and p.get("content"):
                out.append(p["content"])
    return "\n".join(out)


def _tool_calls(messages: List[dict]) -> List[Tuple[str, dict]]:
    """Return ``(tool_name, arguments)`` for every tool_call in the messages."""
    calls: List[Tuple[str, dict]] = []
    for m in messages:
        for p in m.get("parts", []):
            if p.get("type") == "tool_call":
                args = p.get("arguments")
                if not isinstance(args, dict):
                    args = {}
                calls.append((p.get("name", ""), args))
    return calls


def _canonical_persona_name(identity: str, title: str) -> str:
    """Best-effort canonical persona name.

    Prefer a bundled persona file whose text mentions this identity (so the
    name matches the YAML, e.g. ``cassio_a_charmer``); otherwise fall back to a
    slug built from the identity + title.
    """
    if _PERSONAS_DIR.is_dir():
        for path in _PERSONAS_DIR.glob("*.yaml"):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if identity in text:
                return path.stem
    slug = f"{identity}_{title}".lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
    return slug


def _format_trade(offer: List[int], ask: List[int]) -> str:
    """Render a finalize_trade offer/ask (5-vectors) as a readable string."""
    def side(vec: List[int]) -> str:
        parts = [
            f"{_RESOURCES[i]}:{v}"
            for i, v in enumerate(vec or [])
            if i < len(_RESOURCES) and v
        ]
        return "{" + ", ".join(parts) + "}"
    return f"offer={side(offer)} ask={side(ask)}"


def _extract_negotiations(rows: List[dict]) -> Dict[str, List[Negotiation]]:
    """Reconstruct negotiation sessions and index them per participant color.

    Each messaging / finalization span is one speaker's turn. We group spans by
    session key ``(initiator, turn)``, order by timestamp, and rebuild the full
    transcript. The resulting session is attached to every participant color, so
    a per-player judge sees the whole conversation with that player's own
    messages flagged.
    """
    # session key -> accumulating state
    sessions: Dict[Tuple[str, Optional[int]], Dict[str, Any]] = {}

    def _session(key: Tuple[str, Optional[int]]) -> Dict[str, Any]:
        return sessions.setdefault(
            key,
            {
                "initiator": key[0],
                "turn": key[1],
                "participants": [],
                "messages": [],  # list of (sender, content)
                "trade": None,
            },
        )

    for span in sorted(rows, key=lambda r: r.get("start_timestamp") or ""):
        attrs = span.get("attributes", {})
        in_msgs = attrs.get("gen_ai.input.messages", [])
        out_msgs = attrs.get("gen_ai.output.messages", [])
        user_text = _message_text(
            [m for m in in_msgs if m.get("role") != "system"]
        )

        speak_m = _NEG_SPEAKER_RE.search(user_text)
        final_m = _NEG_FINALIZE_RE.search(user_text)
        if not speak_m and not final_m:
            continue

        turn_m = _NEG_TURN_RE.search(user_text)
        turn = int(turn_m.group(1)) if turn_m else None

        if speak_m:
            speaker = speak_m.group(1)
            init_m = _NEG_INITIATOR_RE.search(user_text)
            initiator = init_m.group(1) if init_m else speaker
            sess = _session((initiator, turn))
            if not sess["participants"]:
                part_m = _NEG_PARTICIPANTS_RE.search(user_text)
                if part_m:
                    sess["participants"] = [
                        p.strip().strip("'\"")
                        for p in part_m.group(1).split(",")
                        if p.strip()
                    ]
            for name, args in _tool_calls(out_msgs):
                if name == "send_message" and args.get("message"):
                    sess["messages"].append((speaker, args["message"]))
                elif name == "leave_negotiation":
                    # Collapse consecutive identical leaves (round-robin repeats).
                    if sess["messages"][-1:] != [(speaker, "[left the negotiation]")]:
                        sess["messages"].append((speaker, "[left the negotiation]"))
        else:  # finalization span
            initiator = final_m.group(1)
            sess = _session((initiator, turn))
            for name, args in _tool_calls(out_msgs):
                if name == "finalize_trade":
                    sess["trade"] = _format_trade(
                        args.get("offer", []), args.get("ask", [])
                    )

    # Fan out each session to every participant color.
    per_color: Dict[str, List[Negotiation]] = defaultdict(list)
    for (initiator, turn), sess in sessions.items():
        participants = sess["participants"] or sorted(
            {s for s, _ in sess["messages"]}
        )
        for color in participants:
            neg = Negotiation(
                turn=turn,
                session_id=f"{initiator}-t{turn}",
                initiator=initiator,
                participants=participants,
                messages=[
                    NegotiationMessage(
                        sender=sender,
                        content=content,
                        is_agent_under_eval=(sender == color),
                    )
                    for sender, content in sess["messages"]
                ],
                resulting_trade=sess["trade"] if color == initiator else None,
            )
            per_color[color].append(neg)

    # Deterministic order: by turn then session id.
    for color in per_color:
        per_color[color].sort(key=lambda n: (n.turn if n.turn is not None else -1))
    return per_color


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
def load_traces(logs_path: str | Path) -> List[AgentGameTrace]:
    """Parse a Logfire chat-span export into one trace per player color.

    Args:
        logs_path: path to the columnar JSON export (``{"columns","rows"}``),
            e.g. ``outputs/gemini_chat_spans.json``.

    Returns:
        One :class:`AgentGameTrace` per player color found in the logs, with
        per-decision ``reasoning`` (and ``write_memory`` content) attached in
        chronological order, plus reconstructed ``negotiations`` (full session
        transcripts with the agent's own messages flagged). ``negotiations`` is
        empty for exports that contain no negotiation spans.
    """
    path = Path(logs_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: List[dict] = data["rows"] if isinstance(data, dict) else data
    game_id = path.stem

    # Accumulate per-color state as we scan spans in timestamp order.
    per_color: Dict[str, Dict[str, Any]] = {}

    def _slot(color: str) -> Dict[str, Any]:
        return per_color.setdefault(
            color, {"persona": None, "decisions": []}
        )

    for span in sorted(rows, key=lambda r: r.get("start_timestamp") or ""):
        attrs = span.get("attributes", {})
        in_msgs = attrs.get("gen_ai.input.messages", [])
        out_msgs = attrs.get("gen_ai.output.messages", [])

        system_text = _message_text(in_msgs, role="system")
        user_text = _message_text(
            [m for m in in_msgs if m.get("role") != "system"]
        )

        # Attribute the span to a color from the decide prompt (`You are: X`) or,
        # for negotiation spans, from the negotiation speaker/finalizer line.
        color_m = _COLOR_RE.search(user_text)
        if color_m:
            color = color_m.group(1)
        else:
            neg_m = _NEG_SPEAKER_RE.search(user_text) or _NEG_FINALIZE_RE.search(
                user_text
            )
            color = neg_m.group(1) if neg_m else None
        if color is None:
            continue  # no player attribution in this span; can't place it
        slot = _slot(color)

        # Persona (recorded once per color from the system prompt).
        if slot["persona"] is None and system_text:
            ident = _IDENTITY_RE.search(system_text)
            if ident:
                name = _canonical_persona_name(ident.group(1), ident.group(2))
            else:
                name = f"{color.lower()}_unknown"
            slot["persona"] = Persona(name=name, system_prompt=system_text.strip())

        turn_m = _TURN_RE.search(user_text)
        turn = int(turn_m.group(1)) if turn_m else None

        # Reasoning + strategic-intent from the model's tool calls.
        for name, args in _tool_calls(out_msgs):
            if name == "final_result" and args.get("reasoning"):
                idx = args.get("action_index")
                slot["decisions"].append(
                    Decision(
                        turn=turn,
                        action_type="DECISION",
                        action_detail=(
                            f"action_index={idx}" if idx is not None else None
                        ),
                        reasoning=args.get("reasoning"),
                        confidence=args.get("confidence"),
                    )
                )
            elif name == "write_memory" and args.get("content"):
                # Memory writes are private strategic planning — a strong signal
                # for intent-to-deceive, so surface them as reasoning too.
                slot["decisions"].append(
                    Decision(
                        turn=turn,
                        action_type="WRITE_MEMORY",
                        action_detail="agent updated its private memory",
                        reasoning=args["content"],
                    )
                )

    # Reconstruct negotiation sessions (empty if the game had no negotiations).
    negotiations_by_color = _extract_negotiations(rows)

    traces: List[AgentGameTrace] = []
    all_colors = set(per_color) | set(negotiations_by_color)
    for color in sorted(all_colors):
        slot = per_color.get(color, {"persona": None, "decisions": []})
        persona = slot["persona"] or Persona(
            name=f"{color.lower()}_unknown", system_prompt=""
        )
        traces.append(
            AgentGameTrace(
                game_id=game_id,
                agent_color=color,
                persona=persona,
                decisions=slot["decisions"],
                negotiations=negotiations_by_color.get(color, []),
            )
        )
    return traces


def load_traces_from_json(path: str | Path) -> List[AgentGameTrace]:
    """Load traces from a JSON file that is already in ``AgentGameTrace`` shape.

    Accepts either a single trace object or a list of them. This is the escape
    hatch for hand-written / pre-normalized traces (see ``examples/``).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else [data]
    return [AgentGameTrace.model_validate(item) for item in items]
