# Cora

[![tests](https://github.com/RattusKing/Cora/actions/workflows/tests.yml/badge.svg)](https://github.com/RattusKing/Cora/actions/workflows/tests.yml)

**A citation-gated hypothesis-card tool for comparative biology and human medicine.**

Cora points at one research question at a time — a **topic** — and drafts one *hypothesis
card* at a time from real PubMed abstracts about the animals that solved it, then checks the
card twice before it can be believed:

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

## Topics

A topic is the question Cora is pointed at: its PubMed search terms, its species panel, its
mechanism vocabulary, and the lineage rank used for convergence. You pick one or more on
first run, change them any time with `cora topics`, and add your own.

| Topic | Species | Question |
|---|---|---|
| `aging` — Aging & longevity **(ready)** | naked mole-rat, ocean quahog, rockfish, bowhead whale, Greenland shark, hydra, *Turritopsis*, killifish | Which mechanisms let some animals live far longer than their body size predicts? |
| `regeneration` — Regeneration & wound repair | axolotl, newt, zebrafish, planarian, spiny mouse, hydra, *Hofstenia*, deer antler | How do animals regrow tissue without scarring? |
| `cancer_resistance` — Cancer resistance | naked mole-rat, blind mole-rat, elephant, bowhead whale, bats | Why do some large or long-lived animals almost never get cancer (Peto's paradox)? |
| `hypoxia` — Low-oxygen tolerance | naked mole-rat, painted turtle, red-eared slider, crucian carp, Weddell seal, bar-headed goose, epaulette shark | How do animals survive hours to months without oxygen? |
| `hibernation` — Hibernation & metabolism | 13-lined and arctic ground squirrel, bear, dwarf lemur, bats, wood frog | How do hibernators drop their metabolism for months without losing muscle or bone? |

Only `aging` is marked **ready**: its mechanism lexicon has been checked against hundreds of
real abstracts. The other four are labelled **beta** until the same check is done, and
`cora topics` says so. A species can belong to several topics; the same paper is stored once
and linked to each topic that found it, and every count is per topic.

```bash
cora topics                      # what exists, what is enabled, what is ingested
cora topics --set 1 3            # enable topics by number (or by key)
cora topics --add                # add your own topic; checks each species against PubMed first
cora topics --export aging > aging.json     # share or edit a topic as JSON
cora setup                       # fetch whatever the enabled topics still need
```

**Adding your own topic.** `cora topics --add` asks for a key, a name, a one-line question,
species keys, mechanism keys and the PubMed terms, then runs one search per species and
prints the hit count *before* storing anything — a zero usually means a misspelled binomial.
Custom topics live in your database and export as JSON. The scope is animal traits → human
health; Cora is not a tool for making anything more dangerous.

## Status

| Piece | State |
|---|---|
| Design trail (`docs/`) | Feasibility study, architecture, Phase 1–4 specs, a four-angle red-team whose corrections are folded into every spec (marked `[RT]`), and `Cora-Topics.md` for the topic-pack design |
| P0.5 build | Done and merged: ingest, mechanical gate, cards, ledger, metrics, CLI, web UI, eval harness |
| Phase 1, first item | Done and merged: the pattern-line gate (mechanical check + independent judge + one rewrite) and `cora label` |
| Weekly re-check | Done: `cora recheck` fetches only new abstracts, flags retracted or corrected sources that live cards cite, and reports what changed per card. No model calls |
| Full panel + AnAge | Done: all eight aging species are that topic's panel; `cora anage` loads the AnAge table (4,645 species) and `cora species` shows each panel species' maximum longevity with its data quality, sample size and specimen origin, body mass, and a naive longevity quotient. AnAge covers lifespan only, so it backs the `aging` topic; the other topics have no trait table yet and say so |
| Topic packs | Done: five built-in topics (aging, regeneration, cancer resistance, low-oxygen tolerance, hibernation) plus custom topics. One PubMed query per (topic, species) pair; abstracts, cards, findings and convergence are all scoped to the topics you enable; `cora topics` picks them; a pre-topics database is migrated to `aging` without re-downloading anything |
| Graph-lite convergence | Done: `cora graph build` turns every abstract into gate-verified **findings** (species × mechanism × tier × direction, verbatim quote, primary vs review); `cora graph converge` ranks mechanisms by independent lineages with a **permutation p-value**; cards carry computed convergence for the mechanisms their pattern names. First real run: broad mechanisms reach all five lineages but **none beats the null**, which is the honest answer for lexicon matches over abstracts |
| Tests | 124 passing on Python 3.10–3.13; no network, no credentials. GitHub Actions runs them on every pull request and every push to `main` |
| One-command setup | `cora setup` picks topics on first run, then runs PubMed ingest, AnAge, graph build and a canary check, reports each step, and is safe to re-run. Re-run: about 1 second |
| Real-data checks | November 2026, all five topics against live PubMed: **6,553 unique abstracts** over 34 topic/species pairs (aging 662, regeneration 3,593, hibernation 1,535, low-oxygen 558, cancer resistance 422; a paper two topics both find is stored once), **34,725 findings with 0 dropped by the gate**, canaries 96 of 96 known-bad caught and 20 of 20 known-good passed. The aging topic alone reproduces at 662 abstracts / 1,776 findings, and still **no mechanism beats the null** (lowest permutation p = 0.46). A fresh single-topic install takes about 3 minutes of PubMed fetching; a re-run about 1 second. Reviewing the new lexicons on those abstracts found and fixed two real bugs (below). AnAge 4,645 rows. Mock cards gate cleanly in every topic |
| Lexicon review (the new topics) | Sampling real findings for all 19 new mechanism keys found two defects, both fixed: (1) a negation in **one clause** marked the whole sentence negative — "AMPK increased in adipose tissue, *but* not in liver" was recorded as a negative result for adipose tissue; tier and direction are now read from the clause that contains the mechanism, which cut mixed-sentence negatives from 45% to 29% of all negatives, the remainder being genuine; (2) a bare "nitrogen balance" pattern matched brain glutamine metabolism, not muscle sparing. The four new lexicons stay **beta** until they have been used |
| **Not yet verified** | **Live model calls.** The build environment had no credentials, so real drafting, the real judge, and the rewrite have never been run against the API. They follow the documented structured-output call shape and fail with a clear message when credentials are missing |

**What's next**, in order, per `docs/Cora-Red-Team.md` §5. Everything on the v0.3 build list
through graph-lite now exists, plus topics; what remains needs either a model key or use.
With a key: run real cards, run `cora graph build --model` so findings come from a model
extractor instead of the lexicon, and label the eval pairs. Then: use the new topics enough
to promote their lexicons out of beta, a species tree (TimeTree) to replace the taxonomic-rank
lineage proxy with phylogenetic weighting, and cross-topic convergence — a mechanism that
recurs in *both* hibernators and anoxia-tolerant animals is a signal Cora does not yet
compute (`docs/Cora-Topics.md` explains why not). The standing rule: **no new spec until the
previous phase has been used.**

## Quickstart

```bash
pip install -e ".[dev]"          # on a restricted network add --no-build-isolation
cora setup                       # everything a fresh install needs, in one step (~20 s); safe to re-run
```

That's the whole install. Then:

```bash
cora ask "DNA repair and extreme lifespan"          # needs model credentials (see below)
cora ask "DNA repair and extreme lifespan" --mock   # no credentials: deterministic drafter + judge
cora serve                       # web UI at http://127.0.0.1:8000
```

Every command:

```bash
cora setup --refresh             # re-fetch everything; --skip anage graph ... to skip steps; --species to limit
cora init                        # creates data/cora.db, prints the drafter/judge config
cora ingest                      # PubMed abstracts for every (topic, species) pair of the enabled topics
cora ingest --topic regeneration # one topic only
cora ingest --species naked_mole_rat ocean_quahog rockfish   # or a subset of species
cora anage                       # download AnAge (HAGR) lifespan records and print the panel table
cora species                     # the panel: docs, AnAge max longevity + quality/sample/origin, body mass, naive LQ
cora canary                      # prove the gate catches known-bad citations on the real corpus
cora recheck                     # weekly: fetch only new abstracts, flag corrected sources, report what changed
cora recheck --report            # print the last report without fetching
cora graph build                 # gate-verified findings from every abstract (lexicon; --model for the model extractor)
cora graph converge              # mechanisms ranked by independent lineages, with a permutation p-value
cora graph converge --min-tier association --min-lineages 3
cora graph show dna_repair       # every finding for one mechanism, per species, with its verbatim quote
cora graph converge --topic hibernation      # any command that reads the corpus takes --topic
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
PubMed abstracts carry no per-record reuse license. `data/manifest.json` records the PMIDs, the exact
query and the download date per topic/species pair. The committed copy is the `aging`
corpus; your own `cora setup` runs extend it locally with the topics you enable. A manifest
written before topics existed is still read for `aging`, so an existing install
re-downloads nothing.
Species are queried by binomial or genus name only; common names pull the wrong organisms
("quahog" finds *Mercenaria*, "rockfish" finds striped bass). Generated eval files that
contain abstract text are git-ignored too. The AnAge table (CC BY 3.0, from
`genomics.senescence.info`) is kept locally at `data/anage/anage_data.txt`; a genus-level
panel entry such as *Sebastes* takes the longest-lived species in the genus and says so.

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

## The weekly re-check

Cadence follows the evidence rate: this panel gains a handful of aging abstracts a week, so
a daily briefing would have nothing to say. `cora recheck` runs weekly (cron) or on demand,
never calls a model, and does four things:

1. Searches PubMed again for each panel species and fetches **only unseen PMIDs** (a few
   requests, not a re-download). PMIDs that a search no longer returns are counted, never
   deleted.
2. Re-fetches the PMIDs that live cards cite and records **retraction, erratum and
   expression-of-concern** links as flags on those sources.
3. Matches each new abstract to live cards: does it rank in the top *k* for the card's query
   among **all** abstracts in the card's species (not just the new ones), or share at least
   three of the card's pattern terms? Each match is recorded as an update on the card, and
   the ledger shows `new +N` until you open the card.
4. Prints a templated **"what changed"** report. Every line is a new PMID, a source flag, or
   a count; a run that finds nothing says so in one line. A flag or touch reported last week
   is not repeated this week.

```
# crontab: every Monday at 09:00, from the project directory
0 9 * * 1  cd /path/to/Cora && cora recheck >> data/recheck.log 2>&1
```

## Graph-lite convergence

The graph is a table of **findings**, not a graph database: one row per (species,
mechanism, tier, direction, evidence type, verbatim quote, PMID). Every quote passes the same
citation gate as a card, whichever extractor produced it. Convergence is computed over
findings only, at the **mechanism level**, because gene-level orthology exists for only four
of the eight species.

- **Mechanism vocabulary.** Eighteen mechanisms, each mapped to the hallmarks of aging where
  one applies (DNA repair → genomic instability, telomeres, proteostasis, nutrient sensing,
  senescence, stem cells, epigenetics, mitochondria, inflammation) plus panel-specific
  mechanisms with no hallmark (hyaluronan, cancer resistance, hypoxia tolerance, metabolic
  depression, cold adaptation, diapause).
- **Extractors.** The default is a **lexicon**: each sentence matching a mechanism becomes a
  finding at tier *mention*, upgraded to *association* or *intervention* by cue words and
  downgraded to *tested_negative* by negation cues. A **model extractor** with the same schema
  exists for when credentials are available. Both go through the gate.
- **Primary vs review.** PubMed publication types separate primary studies from reviews.
  Reviews are recorded as pointers and never count as support. Negative statements are
  recorded and reported, never hidden.
- **Lineages, not species.** Support is counted in independent lineages using a coarse
  proxy, taxonomic class: a bowhead whale and a naked mole-rat are one lineage. A species
  tree replaces this later; the column is named `lineages` so it is never mistaken for a
  species count.
- **A permutation null on every mechanism.** Mechanism labels are shuffled across primary
  findings (preserving each species' finding count) and the number of lineages a label
  reaches by chance is compared with the observed number. The first real run is the honest
  result: the broad mechanisms reach all five lineages and **none beats the null**. That is
  what mentions in abstracts can and cannot show, and Cora says so instead of calling it
  convergence.
- **Cards carry it.** When a card's pattern names a mechanism, the card shows that
  mechanism's lineages, species, negatives and p-value, computed from the findings, never
  asserted by the drafter.

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
cora/           config · db · ingest · anage · retrieve · card · gate · verify · llm · generate · ledger · recheck · graph · setup · metrics · label · evalset · cli · api
cora/static/    the single-page UI (ask, what changed, convergence, ranked cards, expand, feedback, export, check-in)
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
