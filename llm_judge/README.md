# llm_judge

LLM-as-a-judge evaluation for the Catan persona agents.

The judges use a **separate, cheap model (OpenAI `o4-mini`)** to grade the
behaviour of the in-game persona agents. There are two independent judges, each
with its own prompt and CLI command:

| Command    | What it evaluates |
|------------|-------------------|
| `deception`| Where an agent is being deceptive — in its private **reasoning** and, most importantly, in the **negotiation** phase (the only channel for talking to other agents). Flags bluffs, false promises, broken commitments, misrepresentations, etc. |
| `persona`  | How faithfully an agent sticks to its **persona system prompt** across three surfaces: private **reasoning**, concrete **game decisions**, and **negotiation** messages. |
| `memory`   | Whether the agent uses its persistent **memory** tool correctly. Two axes: **hygiene** (does it `write_memory` when its reasoning shows something worth remembering across turns?) and **persona alignment** (does it save the things *this* persona should track — e.g. a deceiver logging betrayal routes / broken promises, a punisher logging who defected?). |
| `negotiation`| The **negotiation** phase judged on **both** axes at once, per player: **persona compliance** (are the agent's messages in character?) and **deception** (bluffs, false promises, broken commitments in what it says). Reconstructs full session transcripts and scores only the agent's own messages. |

Each judge is deliberately standalone — one prompt, one output schema, one
command — so they can be run, iterated on, and reasoned about separately.

## Layout

```
llm_judge/
  config.py            # model name + API key loading (reads repo-root .env)
  client.py            # thin OpenAI o4-mini wrapper (JSON-mode calls)
  records.py           # INPUT schema: the normalized per-agent game trace judges consume
  schemas.py           # OUTPUT schema: structured judge verdicts (pydantic)
  prompts/             # the judge prompts (the actual "as-a-judge" logic)
    deception.py
    persona_adherence.py
    memory_usage.py
  judges/              # glue: build prompt -> call model -> parse verdict
    base.py
    deception.py
    persona_adherence.py
    memory_usage.py
  loader.py            # parse a Logfire chat-span export into records.py traces
  examples/            # a hand-written sample trace so the judges are runnable today
  __main__.py          # CLI: `python -m llm_judge deception|persona|memory ...`
```

## Log format (what `loader.py` parses)

The logs are a **Logfire columnar export** of the Gemini chat spans — one JSON
file (e.g. `outputs/gemini_chat_spans.json`) shaped `{"columns": [...], "rows": [...]}`,
one row per LLM call. There is no first-class player/action field, so
`loader.py` recovers attribution from the prompt text:

- **which player** ← the user prompt begins with `You are: <COLOR>`
- **which turn** ← `Turn: <N>` in the user prompt
- **which persona** ← the identity line in the system prompt (`You are Cassio-A, "The Charmer."`)
- **reasoning** ← the model's `final_result` tool call → `arguments.reasoning`
- **memory writes** ← `write_memory` tool calls → `arguments.content`

It emits one `AgentGameTrace` per player color. **Negotiations** are also
reconstructed when present: messaging/finalization spans (prompt markers
`... in a trade negotiation` / `The negotiation messaging has concluded`) are
grouped into sessions by `(initiator, turn)`, ordered by timestamp, and rebuilt
into full transcripts — each `send_message` is one line, `finalize_trade` gives
the resulting offer. Every session is attached to all its participants, with
that player's own messages flagged. Exports with no negotiation spans (e.g.
`gemini_chat_spans.json`) simply yield empty `negotiations`.

## Usage

```bash
# one-time: install the judge deps (openai SDK + pydantic)
pip install -r llm_judge/requirements.txt

# set the key for the judge model (separate from the game's model key)
export OPENAI_API_KEY=sk-...        # or add it to the repo-root .env

# evaluate ALL players in a real Logfire chat-span export (parsed by loader.py)
python -m llm_judge deception   --logs outputs/gemini_chat_spans.json
python -m llm_judge persona     --logs outputs/gemini_chat_spans.json
python -m llm_judge memory      --logs outputs/gemini_chat_spans.json
python -m llm_judge negotiation --logs outputs/gemini_3.5_flash_chat_spans.json  # needs a game with negotiations

# ...or just one player
python -m llm_judge memory --logs outputs/gemini_chat_spans.json --agent BLUE

# or evaluate the bundled hand-written example trace
python -m llm_judge deception --input llm_judge/examples/sample_trace.json
```

### Where verdicts go

By default each run **auto-writes one JSON file per player** into a folder named
after the game's personas:

```
outputs/judge_verdicts/<personas>/<judge>_<COLOR>.json
# e.g.
outputs/judge_verdicts/charmer_punisher/deception_RED.json
                                        /memory_BLUE.json
                                        /persona_ORANGE.json
```

The `<personas>` folder is the distinct persona short-names in the game joined
with `_` (e.g. `charmer_punisher`), so it stays the same whether you judge one
player or all of them. Override with:

- `--out-dir <dir>` — change the base dir (default `outputs/judge_verdicts`)
- `--out <file>`   — write to one exact file instead of the auto folder
- `--stdout`       — print to stdout instead of writing files

The `--input` file is a JSON `AgentGameTrace` (or a list of them); use `--logs`
to point at a raw Logfire export instead.
