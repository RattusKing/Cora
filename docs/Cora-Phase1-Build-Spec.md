# Cora — Phase 1 Build Spec (v0.1, design only)

**Date:** 2026-06-27
**Status:** Build target — *no code yet.* The concrete thing Phase 1 produces.
**Parent:** `docs/Cora-Architecture.md` (this is Phase 1 of §12).

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
| **AnAge** (HAGR) | bulk download | free for research; **commercial use needs a license check** ⚠️ | lifespan / aging records per species |
| **GenAge** (HAGR) | bulk download | research-free ⚠️ | aging-associated genes |
| **DrugAge** (HAGR) | bulk download | research-free ⚠️ | lifespan-extending compounds |
| **LongevityMap** (HAGR) | bulk download | research-free ⚠️ | human longevity variants |
| **PubMed abstracts** | E-utilities API (3–10 req/s) | free; abstracts only | abstracts for `species × (aging OR longevity OR senescence OR lifespan)` |
| **UniProt / Ensembl** | API | open | ortholog mapping (species gene → human gene) |

⚠️ HAGR is free for research — fine for the **indie/personal** Phase-1 user — but a commercial launch needs an explicit license review (consistent with the architecture's license-aware principle).

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

**Tight-card default view** (≈8 lines): `claim · convergence (species + depth) · human_lever · confidence · status` + citation count. Expand-on-click reveals full `evidence` + `falsifier` (the dossier).

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

## 5. Retrospective-rediscovery eval harness (the Phase 1 success gate)

Freeze the corpus to a cutoff date, then test whether Cora re-derives discoveries published *after* it.

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
| Reasoning (hard) | frontier API | generate + critic + verify |
| Reasoning (cheap) | small/fast model | retrieval triage, extraction |
| Vector store | keep **ChromaDB** | fine at this scale |
| Graph | **none yet** | convergence is pre-graph (P2 adds Neo4j) |
| Doc/provenance store | sqlite or jsonl | upgrade later |
| Backend / UI | existing FastAPI + `index.html` | |

**Plumbing cleanup folded in** (from feasibility doc §"repo"): fix `safety.py` undefined `DISALLOWED_KEYWORDS`; `rag_engine.py` `dolphin` vs `dolphins.py`; static-mount shadowing `/api/chat`.

---

## 9. Definition of done (Phase 1)

1. Seed corpus ingested with provenance + license tags.
2. On-demand briefing of tight, expandable cards over that corpus.
3. Citation-verification gate live; **fabrication rate measured and low**.
4. Retrospective-rediscovery eval runs, reports, and **passes threshold on ≥1 gold discovery**.
5. Honest guardrails replace the old prompt.

When these five hold, the card is trustworthy — and *only then* do we earn the right to build the graph (P2) and the self-propelling loop (P3).

---

*Living document. No code committed — this is the target we build against.*
