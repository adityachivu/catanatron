"""CLI for the LLM-as-a-judge evaluation.

One subcommand per judge:

    python -m llm_judge deception --logs outputs/gemini_chat_spans.json
    python -m llm_judge persona   --logs outputs/gemini_chat_spans.json
    python -m llm_judge memory    --logs outputs/gemini_chat_spans.json

All accept the same options and only differ in which judge they run. Use
``--logs`` to point at a raw Logfire chat-span export (parsed by
``llm_judge.loader``), or ``--input`` for a pre-normalized AgentGameTrace JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from pydantic import BaseModel

from llm_judge.config import JudgeConfig
from llm_judge.judges.base import BaseJudge
from llm_judge.judges.deception import DeceptionJudge
from llm_judge.judges.memory_usage import MemoryUsageJudge
from llm_judge.judges.negotiation import NegotiationJudge
from llm_judge.judges.persona_adherence import PersonaAdherenceJudge
from llm_judge.loader import load_traces, load_traces_from_json
from llm_judge.records import AgentGameTrace

JUDGES: dict[str, type[BaseJudge]] = {
    "deception": DeceptionJudge,
    "persona": PersonaAdherenceJudge,
    "memory": MemoryUsageJudge,
    "negotiation": NegotiationJudge,
}


def _load_traces(args: argparse.Namespace) -> List[AgentGameTrace]:
    if args.input:
        return load_traces_from_json(args.input)
    if args.logs:
        # Not implemented yet — raises a clear NotImplementedError.
        return load_traces(args.logs)
    raise SystemExit("error: provide --input <trace.json> (or --logs once implemented)")


def _filter_agent(traces: List[AgentGameTrace], color: str | None) -> List[AgentGameTrace]:
    if not color:
        return traces
    return [t for t in traces if t.agent_color.upper() == color.upper()]


def _short_persona(name: str) -> str:
    """'cassio_a_charmer' -> 'charmer'; used to name the output folder."""
    return name.rsplit("_", 1)[-1] if name else "unknown"


def _personas_label(traces: List[AgentGameTrace]) -> str:
    """Folder label from the distinct personas in the game, e.g. 'charmer_punisher'."""
    shorts = sorted({_short_persona(t.persona.name) for t in traces})
    return "_".join(shorts) if shorts else "unknown"


def _run(judge_name: str, args: argparse.Namespace) -> int:
    all_traces = _load_traces(args)
    traces = _filter_agent(all_traces, args.agent)
    if not traces:
        print("no traces to evaluate (check --input / --agent)", file=sys.stderr)
        return 1

    config = JudgeConfig.from_env(model=args.model, reasoning_effort=args.effort)
    judge = JUDGES[judge_name](config=config)

    # Where per-agent verdicts land by default: <out-dir>/<personas>/<judge>_<COLOR>.json
    # Folder is derived from ALL personas in the game (pre-filter) so it stays
    # stable whether you judge one agent or all of them.
    out_dir = Path(args.out_dir) / _personas_label(all_traces)

    pairs: List[tuple[AgentGameTrace, BaseModel]] = []
    for trace in traces:
        print(
            f"[{judge_name}] judging {trace.agent_color} "
            f"({trace.persona.name}) in game {trace.game_id} with {config.model}...",
            file=sys.stderr,
        )
        pairs.append((trace, judge.evaluate(trace)))

    if args.stdout:
        payload = [v.model_dump() for _, v in pairs]
        print(json.dumps(payload if len(payload) != 1 else payload[0], indent=2))
        return 0

    if args.out:
        # Explicit single-file override (bundles all verdicts if more than one).
        payload = [v.model_dump() for _, v in pairs]
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(payload if len(payload) != 1 else payload[0], indent=2),
            encoding="utf-8",
        )
        print(f"wrote {len(payload)} verdict(s) to {args.out}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for trace, verdict in pairs:
        path = out_dir / f"{judge_name}_{trace.agent_color}.json"
        path.write_text(json.dumps(verdict.model_dump(), indent=2), encoding="utf-8")
        print(f"wrote {path}")
    return 0


def _add_common_args(sub: argparse.ArgumentParser) -> None:
    src = sub.add_argument_group("input")
    src.add_argument("--input", help="Path to a normalized AgentGameTrace JSON (or list)")
    src.add_argument("--logs", help="Path to raw game logs (loader not yet implemented)")
    sub.add_argument("--agent", help="Only evaluate this player color, e.g. RED")
    out = sub.add_argument_group("output")
    out.add_argument(
        "--out-dir",
        default="outputs/judge_verdicts",
        help="Base dir for auto-organized verdicts "
        "(default: outputs/judge_verdicts). Files land in "
        "<out-dir>/<personas>/<judge>_<COLOR>.json",
    )
    out.add_argument(
        "--out", help="Write to this exact file instead of the auto-organized folder"
    )
    out.add_argument(
        "--stdout", action="store_true", help="Print to stdout instead of writing files"
    )
    sub.add_argument("--model", help="Override judge model (default: o4-mini)")
    sub.add_argument(
        "--effort",
        choices=["low", "medium", "high"],
        help="o4-mini reasoning_effort (default: medium)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm_judge", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    dec = subparsers.add_parser(
        "deception", help="Detect deception in reasoning and negotiation"
    )
    _add_common_args(dec)

    per = subparsers.add_parser(
        "persona", help="Score adherence to the persona prompt"
    )
    _add_common_args(per)

    mem = subparsers.add_parser(
        "memory",
        help="Check memory-tool usage: saves-when-warranted + persona-aligned content",
    )
    _add_common_args(mem)

    neg = subparsers.add_parser(
        "negotiation",
        help="Judge negotiation messages for persona compliance + deception (combined)",
    )
    _add_common_args(neg)

    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _run(args.command, args)


if __name__ == "__main__":
    raise SystemExit(main())
