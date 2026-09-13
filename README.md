# Cora

**A citation-gated hypothesis-card tool for the comparative biology of aging.**

Cora drafts one *hypothesis card* at a time from real PubMed abstracts about long-lived
species, then checks it twice before it can be believed:

1. **Every quote goes through a mechanical citation gate.** The PMID must exist in the local
   corpus and the quote must be an exact (normalized) substring of that abstract. Whatever
   fails is deleted *and counted*.
2. **The pattern line goes through a strength check.** Strong or causal words that none of
   the verified quotes use are flagged mechanically, then an independent judge on a
   *different model* than the drafter decides whether the pattern is supported and whether
   it claims more than its sources. "Stronger" fails.

The card's evidence band, species support, groundedness and status are computed by Cora
from what survived, never asserted by the model. Cora is a personal discovery copilot
first: the design trail in `docs/` explains why it starts small, and what it is meant to grow into.

## Status

| Piece | State |
|---|---|
| Design trail (`docs/`) | Feasibility study, architecture, Phase 1–4 specs, and a four-angle red-team whose corrections are folded into every spec (marked `[RT]`) |
| P0.5 build | Done and merged: ingest, mechanical gate, cards, ledger, metrics, CLI, web UI, eval harness |
| Phase 1, first item | Done and merged: the pattern-line gate (mechanical check + independent judge + one rewrite) and `cora label` |
| Tests | 60 passing; no network, no credentials |
| Real-data checks | September 2026 ingest: 382 abstracts (naked mole-rat 331, ocean quahog 27, rockfish 27). Canaries on that corpus: 97 of 97 known-bad citations caught, 20 of 20 known-good passed. Mock cards gate cleanly |
| **Not yet verified** | **Live model calls.** The build environment had no credentials, so real drafting, the real judge, and the rewrite have never been run against the API. They follow the documented structured-output call shape and fail with a clear message when credentials are missing |

**What's next**, in order, per `docs/Cora-Red-Team.md` §5: use it daily and label the eval
pairs; a weekly re-check that reports what changed for each ledger card; the full species
panel plus AnAge lifespan attributes; then graph-lite convergence across species. The
standing rule: **no new spec until the previous phase has been used.**

## Quickstart

```bash
pip install -e ".[dev]"          # on a restricted network add --no-build-isolation
cora init                        # creates data/cora.db, prints the drafter/judge config
cora ingest                      # PubMed abstracts for naked mole-rat, ocean quahog, rockfish
cora ingest --species bowhead_whale greenland_shark hydra turritopsis killifish   # optional extras
cora canary                      # prove the gate catches known-bad citations on the real corpus
cora ask "DNA repair and extreme lifespan"          # needs model credentials (see below)
cora ask "DNA repair and extreme lifespan" --mock   # no credentials: deterministic drafter + judge
cora ask "..." --no-judge        # mechanical pattern check only (no judge call)
cora ledger                      # ranked cards (--all includes archived)
cora show <id>                   # full evidence, what the gate dropped, the judge's rationale
cora feedback <id> accept        # accept | reject:wrong | reject:uninteresting | dig_deeper
cora export <id>                 # citation-complete paragraph  ->  "a card that left Cora"
cora metrics                     # corpus, gate, pattern-check and user metrics
cora checkin yes                 # weekly: did Cora tell you something you used?
cora serve                       # web UI at http://127.0.0.1:8000
cora evalset --n 150             # unlabeled verifier-gold candidates (hard negatives included)
cora label                       # blind-label them; saves after every answer; resumable
cora label --score-judge         # the judge's precision / recall / false-negative rate
```

**Model credentials.** Drafting uses the Anthropic SDK with `claude-opus-5` by default. The
pattern judge is a *different* model, `claude-sonnet-5` by default (a Sonnet drafter gets
an Opus judge; a same-model pairing is rejected). Set `ANTHROPIC_API_KEY`, or run
`ant auth login`. Without credentials, `--mock` uses a deterministic drafter and judge that
quote the first sentence of the top abstracts. That exercises the gate, ledger and UI; it
is not science.

**Configuration** (environment variables):

| Variable | Default | Meaning |
|---|---|---|
| `CORA_MODEL` | `claude-opus-5` | drafter model |
| `CORA_JUDGE_MODEL` | `claude-sonnet-5` | judge model; must differ from the drafter |
| `CORA_PATTERN_RETRIES` | `1` | rewrites allowed when the judge says "stronger" (0 = fail immediately) |
| `CORA_LLM` | unset | `mock` forces the mock drafter everywhere |
| `CORA_DATA_DIR` | `data` | where the SQLite database and XML snapshots live |
| `NCBI_API_KEY` | unset | raises the E-utilities rate limit from 3 to 10 requests/s |
| `NCBI_EMAIL` | unset | passed to E-utilities if set; never defaulted |

**Data.** Abstracts and raw XML snapshots live in `data/` and are git-ignored, because
PubMed abstracts carry no per-record reuse license. `data/manifest.json` (PMIDs, the exact
query, and the download date per species) is committed so the corpus is reproducible.
Species are queried by binomial or genus name only; common names pull the wrong organisms
("quahog" finds *Mercenaria*, "rockfish" finds striped bass). Generated eval files that
contain abstract text are git-ignored too.

## What a card is

```
[id]  pattern (one line: a pattern the abstracts support, not a finding)
  quote   "verbatim span"  - PMID · title          <- what the citation gate verifies
  against strongest objection
  vs      nearest existing paper
  band    N verified quotes · N abstracts · N species · abstract-only   <- computed, ordinal
  check   pass (judge): supported=yes strength=equal   |   FAIL [stronger]: rationale
  why     which steering rule surfaced it
  flags   sparse species, too few quotes, pattern check failed, ...
  desk    one step doable at a laptop today
  next    desk_check | export_draft | share_page | none
```

- **Status** is one of `gated` (both checks passed), `ungrounded` (no quote survived),
  `overclaim` (pattern stronger than its sources, or strong language with no judge), or
  `unsupported` (the judge found the sources do not support the pattern). Only `gated`
  cards get a next action; the others stay in the ledger, ranked below, and export with a
  warning.
- **Feedback** takes a reason code. `accept` and `dig_deeper` set the state; either
  `reject` archives the card. Archived cards are hidden from the default ledger, never
  deleted.
- **Duplicates** are detected by a key built from the pattern's salient tokens and the
  supporting species; a duplicate is linked to the original and demoted, not dropped.
- **Human-lever fields** are annotated *"animal data only, no human application"*.

## Evaluation

The red-team's most convergent finding was that a verifier has to be measured against
human labels before it is trusted. `eval/` holds the tooling:

- `cora evalset` builds (claim, passage) candidates from the real corpus, including hard
  negatives: hedge stripped, strength inflated, one word altered, right quote on the wrong
  PMID. Kinds are hidden during labeling.
- `cora label` is the blind labeling loop; `--stats` gives counts, a kind × label crosstab
  and the mechanical-gate baseline; `--score-judge` scores the model judge against the
  labels. The judge's false-negative rate is the Phase 1 gate.
- `cora canary` builds known-good and known-bad citations from real documents and asserts
  the gate catches every bad one.
- `eval/closed_book.py` asks the gold questions with **no** corpus, the contamination control
  behind any future "Cora rediscovered X" claim. `eval/gold_questions.jsonl` keeps the four
  classic discoveries as positive controls, because a frontier model will name them from
  memory.

## Layout

```
cora/           config · db · ingest · retrieve · card · gate · verify · llm · generate · ledger · metrics · label · evalset · cli · api
cora/static/    the single-page UI (ask, ranked cards, expand, feedback, export, check-in)
tests/          pytest, no network, no credentials
eval/           labeling and scoring tools, closed-book control, gold questions
docs/           the design trail (below)
legacy/         the original prototype; it never ran and is kept for reference only
```

## The design trail (`docs/`)

| Document | What it settles |
|---|---|
| `Cora-Feasibility-Research.md` | Whether to build it at all: yes, but narrow, grounded, and honest |
| `Cora-Architecture.md` | The whole system: corpus and graph, reasoning core, self-propelling loop, memory, governance; decisions and the re-sequenced roadmap |
| `Cora-Phase1-Build-Spec.md` | The trustworthy card and its gates |
| `Cora-Phase2-Knowledge-Graph.md` | Convergence done honestly: a `Finding` node, phylogenetic weighting, null models |
| `Cora-Phase3-Curiosity-Loop.md` | The scheduled sweep, steering rules, critics, circuit breaker |
| `Cora-Phase4-Memory-Autonomy.md` | Consolidation without laundering, the briefing as the unit of verification |
| `Cora-Red-Team.md` | Four adversarial reviews, ranked by how many agreed, with every correction and the v0.3 roadmap |

Specs carry an "Amended by red-team" banner; where a spec and the red-team report
disagree, the report wins.
