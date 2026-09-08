# Cora — Phase 1 Build Spec (v0.1, design only)

**Date:** 2026-06-27
**Status:** Build target — *no code yet.* The concrete thing Phase 1 produces.
**Parent:** `docs/Cora-Architecture.md` (this is Phase 1 of §12).

> **⚠️ Amended by red-team (2026-09-08).** Corrections applied in place and marked **[RT]**; full findings in `docs/Cora-Red-Team.md`. Key changes for this phase: a **P0.5 two-week build precedes it**; the verifier gold set and closed-book eval baseline come **before** any generation code; **graph-lite** (SQLite + NetworkX) moves *into* P1; HAGR licensing was stated backwards; the eval in §5 is contaminated as written. The red-team report wins on conflict.

---

## Goal (one sentence)

> Produce a single **trustworthy, citation-verified hypothesis card** on demand over a real aging corpus — and an **eval that measures whether we can trust it** — with *no autonomous loop and no graph yet.*

Phase 1 exists to prove one thing: **the card can be believed.** Everything self-propelling comes later and is worthless until this holds.

---

## In scope / out of scope

| ✅ In Phase 1 | ❌ Deferred (later phases) |
|---|---|
| Real seed corpus (license-tagged, provenance) | Autonomous / self-propelling loop (P3) |
| Grounded retrieval (vector + keyword) | Curiosity / intrinsic-motivation engine (P3) |
| Hypothesis **card** generation | Memory / consolidation (P4) |
| **Citation-verification gate** (anti-Galactica) | Neo4j knowledge graph (P2) |
| Critic pass (known / overstated / falsifiable) | Graph-backed exhaustive convergence (P2) |
| Retrospective-rediscovery **eval harness** | Auto-ingestion of new papers (P5) |
| Honest guardrails (replace "no hallucinations") | Morning-briefing automation (P4) |
| Thin briefing UI (stack of tight cards) | — |

**Convergence in Phase 1 is "pre-graph":** the model retrieves passages across the species set and proposes convergence; we count distinct species that each carry ≥1 *verified* citation. Shallow (bounded by retrieval recall), but real. The graph (P2) makes it exhaustive and queryable.

---

## 1. Seed corpus manifest

**Species set** (the long-lived panel + human as ortholog anchor):
bowhead whale · Greenland shark · ocean quahog (*Arctica islandica*) · rockfish (*Sebastes* spp.) · naked mole-rat · hydra · *Turritopsis dohrnii* · African turquoise killifish · *Homo sapiens*.

**Sources:**

| Source | Access | Reuse / license | What we pull |
|---|---|---|---|
| **AnAge** (HAGR) | bulk download | **[RT] CC BY 3.0 — commercial use permitted** (the earlier ⚠️ was backwards) | lifespan records per species — ingest the **quality flag, dating method, n, captive/wild** as attributes of `has_lifespan` |
| **GenAge** (HAGR) | bulk download | CC BY 3.0 | human + model-organism aging genes. **[RT] Contains zero genes for any non-human panel species** — it is the *human anchor*, not a panel mechanism source |
| **DrugAge** (HAGR) | bulk download | CC BY 3.0 | lifespan-extending compounds (model organisms). **[RT] The field must carry "animal data only — no human application"** |
| **LongevityMap** (HAGR) | bulk download | CC BY 3.0 | human longevity variants (**[RT]** only APOE and FOXO3 replicate robustly) |
| **PubMed abstracts** | E-utilities API (3–10 req/s); **[RT] freeze by `edat`, snapshot raw XML** | free; abstracts only (no per-record license field) | abstracts for `species × (aging OR longevity OR senescence OR lifespan)` — **[RT] queried by NCBI taxon ID / MeSH / binomial, never common names** ("quahog" pulls *Mercenaria*, a short-lived clam; "killifish" pulls *Fundulus*; "rockfish" pulls Chesapeake striped bass). ~816 abstracts total. |
| **PubTator3** | bulk FTP | free | **[RT]** NCBI Gene/Taxonomy-normalized entity annotations for all of PubMed — the NER/ER layer, instead of a cheap LLM |
| **UniProt / Ensembl** | API | open | ortholog mapping (species gene → human gene) |

**[RT] Licensing, corrected:** HAGR is CC BY 3.0 (commercial permitted). The real license work is elsewhere — PubMed abstracts carry no per-record license field (NLM: "some abstracts may be protected by copyright"), so chunk-level tagging only becomes meaningful for **PMC-OA full text**, which must be filtered by per-article license *before* choosing gold papers. Note that Kolora (*Science*) and Tian (*Nature*) sit in PMC as 12-month-embargoed author manuscripts — **not** in the OA subset, no TDM license. Flag **KEGG / TimeTree / OrthoDB** licensing before any ingest.

**[RT] Corpus size, measured (engineering review):** the non-human panel × (aging | longevity | senescence | lifespan) is **~816 abstracts** (Greenland shark 14 · *Turritopsis* 10 · quahog 29 · bowhead 30 · rockfish 49 · hydra 75 · NMR 330 · killifish 351), growing ~1.5/week. Rate limits are a non-issue. **No human abstract mining** (217k–590k abstracts, $4–12k per extraction pass): GenAge/LongevityMap are the human anchor; pull PubMed only for GenAge genes × panel species. ~$100–200 rebuilds the whole graph, so schema iteration is effectively free.

**Stored per document:** `id · source · url · license_tag · species · retrieved_at · text/abstract`. License tag travels with every chunk downstream.

---

## 2. The hypothesis card schema (the contract)

```
Card {
  claim:        string                       // one line
  convergence:  [ { species, signal, citation_ids[] } ]
  convergence_depth: int                      // # distinct species with ≥1 VERIFIED citation
  human_lever:  { ortholog, notes, citation_ids[] }
  confidence:   { score: 0..1, band: low|med|high, reason: string }
  evidence:     [ { source_id, title, locator, quote, retrievable: bool } ]
  falsifier:    { experiment, model_organism, predicted_outcome }
  status:       novel | known | contested      // critic-assigned
  groundedness: 0..1                           // fraction of claims with a verified citation
}
```

~~**Tight-card default view** (≈8 lines): `claim · convergence (species + depth) · human_lever · confidence · status` + citation count.~~

> **[RT] Tight card v2.** The v1 card led with the two *least* reliable outputs (model-self-reported confidence and model-judged "novel") and hid the two things a skeptic scans for (the verbatim quote — the only thing the gate actually verifies — and the strongest objection). Eight such cards read "conf 0.6 · novel · depth 2–3 · 4 citations" and are indistinguishable. **v2 tight view:**
> `pattern` *(renamed from "claim" — abstracts support patterns, not findings)* · **strongest verbatim quote with clickable PMID, hedge preserved** · one-line **strongest objection** (the devil's-advocate output, no longer buried) · "novel vs **[nearest existing paper]**" · **evidence band** — ordinal, with counts ("2 abstracts, 0 functional"; "3 independent primary observations · 3 lineages · abstract-only") instead of a 0–1 decimal · why-you're-seeing-this (the steering rule that surfaced it) · **`desk_check`**. Numbers live in the dossier. Expand-on-click reveals full `evidence`, `falsifier`, and the lineage of independent observations.
>
> **Two fields added to the schema — the ones that give the card somewhere to go:**
> `desk_check: { step, doable_today: true }` — a next step doable at a laptop *today* (an Ensembl ortholog lookup, an AnAge record, a named paper, an existing dataset). This replaces `falsifier` on the tight card: a labless user cannot run a killifish experiment, and falsifiability is now a binary critic gate, not a score term.
> `next_action: export_draft | share_page | desk_check | none` — export a citation-complete paragraph (review / preprint / grant section), or a shareable, versioned hypothesis page. **"Cards that left Cora" is the primary product metric.** Without a terminal action outside Cora, *accept* writes a log line and the ledger rots in two weeks.
> `human_lever` gains `{ direction, lever_type: variant | expression | dosage | systemic, transferability, pleiotropy_safety (OncoKB / COSMIC / OMIM), replication_status, note: "animal data only — no human application" }` — ERCC1 cannot be dosed *up*, HAS2 is pro-tumor in pancreatic stroma, TERT is oncogenic; "ortholog exists and is druggable" was not a lever.

---

## 3. Card generation + verification flow (single-shot, no loop)

```
query/topic
   │
   ▼
(1) RETRIEVE  hybrid vector+keyword over seed corpus → evidence passages (with license tags)
   │
   ▼
(2) GENERATE  frontier model drafts a candidate Card strictly from retrieved evidence
   │
   ▼
(3) CRITIC    separate call: already known? overstated? falsifiable? → assigns status
   │
   ▼
(4) VERIFY    for EACH claim: confirm a retrievable source actually entails it;
   │          flag/kill unsupported claims  ← the anti-Galactica gate
   ▼
(5) SCORE     convergence_depth · novelty · groundedness · testability → rank
   │
   ▼
(6) RENDER    tight card; or "I can't ground this" if it fails the gate
```

No autonomy here — it runs when you ask. The *shape* is the future loop's steps 2–6; Phase 3 just adds the curiosity trigger in front and the ledger behind.

---

## 4. Citation verification (the trust core — most of the work)

For every atomic claim in a card:
1. Pull the cited source passage from the corpus.
2. Check **entailment** — does the passage actually support the claim? (model-judged with a verbatim-locatable quote required; NLI as a later hardening step).
3. If no passage entails it → **flag or delete the claim**; never render unsupported text as fact.
4. **Metric:** fabrication rate = % of generated citations that don't resolve to a real, supporting source. Target: drive toward 0; report it on every card.

This single gate is what separates Cora from a Galactica-style confident-nonsense generator.

---

## 5. Retrospective-rediscovery eval harness ~~(the Phase 1 success gate)~~

> **[RT] This eval is contaminated as written and is no longer the P1 gate.** All four reviews flagged it: the frontier model has read every gold paper (freezing the corpus does not freeze weights); the gold answers are *absent* from the frozen corpus (bowhead × aging ≤2014 = **5** abstracts, NMR × hyaluronan ≤2012 = **0** — a "hit" is recitation); "DNA repair" is a free pass (**1,349** generic pre-2014 abstracts); Tian 2013 is contested (Hadi 2020); and "unprompted" is vacuous with 9 species. **Fixed form** (red-team F1): mandatory **closed-book baseline** before any generation code; gold items **post-training-cutoff**, re-selected per model upgrade; freeze by `edat`, snapshot raw XML, exclude/date-freeze bioRxiv; **registered negative set**; **synthetic held-out associations** (fictitious gene symbols); random-order null for "unprompted"; **prospective preregistration**. It moves to the **P2 gate** — it justifies the graph, not the card.
>
> **The actual P1 trust gates are now:** (1) a **human-labeled verifier gold set** — 150–300 claim/passage pairs with hard negatives (topically adjacent non-entailing, hedge-stripped, wrong-direction, wrong-species) — with published verifier precision / recall / **FNR**, built *before* generation; (2) **200 hand-labeled edges** with per-edge precision published on every card and **no convergence score shown until per-edge FPR < 2 %**; (3) canary detection rate.

Original design, kept for the record — freeze the corpus to a cutoff date, then test whether Cora re-derives discoveries published *after* it:

**Gold set (examples):**

| Discovery | Mechanism | Human lever | Corpus cutoff |
|---|---|---|---|
| Rockfish extreme lifespan (Kolora 2021) | DNA repair + butyrophilin/immune CNV | DNA-repair orthologs | 2020 |
| Naked mole-rat cancer resistance (Tian 2013) | high-MW hyaluronan (HAS2) | HAS2 | 2012 |
| Bowhead whale longevity (Keane 2015) | ERCC1 / PCNA DNA repair | ERCC1, PCNA | 2014 |
| *Turritopsis* rejuvenation (2022) | DNA-repair / telomere / stem-cell variants | repair + telomere genes | 2021 |

**Measured:** convergence recall (did it surface the mechanism?), citation faithfulness, calibration (does stated confidence track hit rate?), plus held-out association precision (hide known gene↔longevity links, count how many it recovers).

---

## 6. Honest guardrails (Phase 1)

- **Remove** the `SYSTEM_PROMPT` "no hallucinations ever" line → replace with the grounding contract ("every claim cited or flagged; say 'I can't ground this'").
- Research-facing, **non-diagnostic** framing throughout.
- Capability-aware refusal floor (the panel includes toxin/venom-adjacent biology — keep it).
- License tag enforced in retrieval so non-commercial-licensed text never leaks into a future commercial surface.

---

## 7. Minimal interface (Phase 1)

- Reuse existing **FastAPI** + `index.html` shell.
- Query box → returns a **briefing**: a ranked stack of tight cards.
- Each card: 8-line tight view, **expand-to-dossier** on click, **clickable citations** to source.
- A small **"fabrication rate / groundedness"** readout so trust is visible, not assumed.

---

## 8. Stack for Phase 1 (minimal, concrete)

| Concern | Phase 1 choice | Note |
|---|---|---|
| Reasoning — generate | frontier API (Opus-tier) | **[RT]** generate *only* |
| Reasoning — verify + Overclaim critic | **[RT] a *different* model family** (Sonnet-tier, low effort) | the model that wrote the claim must not grade it; cross-family disagreement rate is a first-class metric |
| Reasoning (cheap) | small/fast model | triage only — **[RT]** entity resolution comes from **PubTator3**, not an LLM |
| Vector store | **[RT]** Chroma `PersistentClient` or sqlite-vec, with a **biomedical embedder** + **BM25 hybrid** | all-MiniLM is general-domain (places "HAS2" and "hyaluronan" far apart); the old `Client(Settings(...))` was ephemeral |
| Graph | **[RT] graph-lite, in P1:** SQLite `edges(...)` + NetworkX, Neo4j-shaped schema | P1 already computes per-species convergence and must store it; lets P3's sweep run without a graph DB |
| Doc / provenance / ledger / **cycles** store | **[RT] SQLite** — `docs`, `edges`, `ledger`, `cycles`, `steps` | every step row written *before* its API call; content-hash cache of (prompt, model, params) → response so retries are idempotent; budget counter persisted per step from API `usage` |
| Scheduled work | **[RT] Message Batches** | 50 % cost, 24 h window, no 429 storms — the orchestration primitive for a scheduled run |
| Budget | **[RT] $5/run hard cap**, enforced from `usage` | one-time graph build ≈ $100–200; report *cost per accepted hypothesis* from day one |
| Hosting | **[RT] a $5–7 VM with a disk and cron** | Render free tier: 512 MB, spins down on idle, no persistent disk — nothing can run unattended and Chroma state vanishes on deploy |
| Backend / UI | FastAPI route shape; **`index.html` rewritten** around the card | |

~~**Plumbing cleanup folded in**~~ **[RT] Discard the skeleton** except the `DISCLAIMER` / `REFUSAL` strings and the FastAPI route shape. Beyond the three known bugs it never ran: `requirements.txt` lists Flask, not FastAPI/Chroma; `public/` was deleted so the static mount raises at import; Chroma list-valued metadata raises on the first `add`; the loaded SentenceTransformer is never used; the evidence block sent to the LLM contained only title/species/link — **never the abstract text, so "grounded in retrieved text" was never true**; `index.html` has no `<script>`; the Groq model ID is decommissioned. Start P0.5 from a fresh `pyproject`.

---

## 9. Definition of done (Phase 1)

1. Seed corpus ingested with provenance + license tags.
2. On-demand briefing of tight, expandable cards over that corpus.
3. Citation-verification gate live; **fabrication rate measured and low**.
4. ~~Retrospective-rediscovery eval passes on ≥1 gold discovery~~ **[RT] replaced:** the **verifier gold set** (150–300 human-labeled pairs) exists *before* generation and verifier precision / recall / **FNR** are published; **200 hand-labeled edges** give a per-edge precision shown on every card, and **no convergence score is displayed until per-edge FPR < 2 %**; canaries run every cycle. The (fixed) rediscovery eval is the **P2** gate.
6. **[RT] It has been *used*:** the three user metrics (days opened / week · cards exported or expanded · weekly "did Cora tell you something you used?") are tracked from P0.5, and ≥1 card has **left Cora** via `next_action`.
5. Honest guardrails replace the old prompt.

When these five hold, the card is trustworthy — and *only then* do we earn the right to build the graph (P2) and the self-propelling loop (P3).

---

*Living document. No code committed — this is the target we build against.*
