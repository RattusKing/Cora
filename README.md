# Cora

**A citation-gated hypothesis-card tool for the comparative biology of aging.** P0.5.

Cora drafts one *hypothesis card* at a time from real PubMed abstracts about long-lived
species, then runs every quote through a **mechanical citation gate**: the PMID must exist
in the local corpus and the quote must be an exact substring of that abstract. Whatever
fails is deleted *and counted*. The card's evidence band, species support, and
groundedness are computed by Cora from what survived — never asserted by the model.

This is the two-week P0.5 build from `docs/Cora-Red-Team.md` §5: three species, one
model, a mechanical gate, a ledger, a desk-check on every card, and the three user metrics
that decide whether it is worth continuing.

## Quickstart

```bash
pip install -e ".[dev]"          # on a restricted network add --no-build-isolation
cora init
cora ingest                      # PubMed abstracts for naked mole-rat, ocean quahog, rockfish
cora canary                      # prove the gate catches known-bad citations on the real corpus
cora ask "DNA repair and extreme lifespan"          # needs model credentials (see below)
cora ask "DNA repair and extreme lifespan" --mock   # no credentials: deterministic drafter + judge
cora ask "..." --no-judge        # mechanical pattern check only (no judge call)
cora ledger                      # ranked cards
cora show <id>                   # full evidence (and what the gate dropped)
cora feedback <id> accept        # accept | reject:wrong | reject:uninteresting | dig_deeper
cora export <id>                 # citation-complete paragraph  ->  "a card that left Cora"
cora metrics                     # gate + user metrics
cora checkin yes                 # weekly: did Cora tell you something you used?
cora serve                       # web UI at http://127.0.0.1:8000
cora evalset --n 150             # unlabeled verifier-gold candidates (hard negatives included)
cora label                       # blind-label them; saves after every answer
cora label --score-judge         # precision / recall / false-negative rate of the judge
```

**Model credentials.** Real drafting uses the Anthropic SDK (`claude-opus-5` by default,
override with `CORA_MODEL`). The pattern judge is a *different* model (`claude-sonnet-5`
by default, override with `CORA_JUDGE_MODEL`; a Sonnet drafter gets an Opus judge). Set
`ANTHROPIC_API_KEY`, or run `ant auth login`. Without credentials, `--mock` uses a
deterministic drafter and judge that quote the first sentence of the top abstracts —
useful for exercising the gate, ledger and UI, not for science.

## Two gates, in order

1. **Quotes — mechanical.** The PMID must exist in the local corpus and the quote must be an
   exact (normalized) substring of that abstract. Failures are deleted *and counted*.
2. **The pattern line — mechanical, then judged.** Strong/causal words in the pattern that
   none of the verified quotes use are flagged mechanically. Then an independent judge on
   a different model returns `supported` (yes | partial | no) and
   `claim_strength_vs_source` (weaker | equal | stronger). **"stronger" fails.** The drafter
   may rewrite the pattern once (`CORA_PATTERN_RETRIES`, default 1); the second verdict is
   final. Cards whose pattern fails are kept with status `overclaim` or `unsupported`,
   ranked below gated cards, and never exported as clean.

The judge is measured, not trusted: `cora label` collects human labels on generated
(claim, passage) pairs, and `cora label --score-judge` reports the judge's precision,
recall and false-negative rate against them. Mechanical-vs-judge disagreement is recorded
on every card and summarized in `cora metrics`.

**Data.** Abstracts and raw XML snapshots live in `data/` (git-ignored; PubMed abstracts
carry no per-record reuse license). `data/manifest.json` (PMIDs + query + date) is
committed so the corpus is reproducible. Set `CORA_DATA_DIR` to move it.

## What a card is

```
[id]  pattern (one line — a pattern the abstracts support, not a finding)
  quote   "verbatim span"  - PMID · title          <- the only thing the gate verifies
  against strongest objection
  vs      nearest existing paper
  band    N verified quotes · N abstracts · N species · abstract-only   <- computed, ordinal
  check   pass (judge): supported=yes strength=equal   |   FAIL [stronger]: ...
  why     which steering rule surfaced it
  flags   sparse species, too few quotes, ...
  desk    one step doable at a laptop today
  next    desk_check | export_draft | share_page | none
```

Human-lever fields are annotated *"animal data only — no human application"*.

## Layout

```
cora/           package: config · db · ingest · retrieve · card · gate · llm · generate · ledger · metrics · cli · api · evalset
cora/static/    the single-page UI
tests/          pytest, no network, no credentials
eval/           verifier gold-set candidates, closed-book contamination control, gold questions
docs/           feasibility → architecture → P1–P4 specs → red-team (the design trail)
legacy/         the original prototype (never ran; kept for reference)
```

## Roadmap

See `docs/Cora-Red-Team.md` §5. The standing rule: **no new spec until the previous phase
has been used.** P0.5 is done when it has been opened daily for 14 days and at least one
card has left Cora.
