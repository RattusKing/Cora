# Cora — System Architecture & Approach (v0.1, design only)

**Date:** 2026-06-27
**Status:** Design draft — *no code yet.* This is the "solid approach" we iterate on before building.
**Builds on:** `docs/Cora-Feasibility-Research.md` (keeps all its constraints) and the existing backend skeleton.

---

## 0. What changed since the feasibility doc

The feasibility doc evaluated Cora as *"a RAG chatbot connecting marine biology to human medicine."* Our brainstorm evolved the vision on two axes:

1. **Form:** not a Q&A chatbot — an **autonomous, self-propelling, curiosity-driven research agent** that generates, critiques, grounds, and ranks hypotheses on its own, and hands a human director a ranked briefing.
2. **Focus (first wedge):** **aging & longevity in long-lived (esp. marine) species → human geroscience targets.** Narrow, with real precedent and — uniquely — *pre-existing structured data* and *believer-buyers*.

Everything the feasibility doc concluded still holds. This document is the *how*, tuned to those two changes.

---

## 1. North star

> An autonomous research agent that continuously hunts for **convergent longevity mechanisms** across long-lived species, grounds every claim in a retrievable citation, calibrates its own confidence, designs falsifiable experiments, and surfaces a ranked hypothesis briefing to a human director — turning the autonomy dial up only as fast as its critic and eval harness earn trust.

**One-line product:** *a grounded, self-driving discovery engine for the comparative biology of aging.*

### 1.1 Decisions locked (v0.2 — 2026-06-27)

| Decision | Choice | Implication |
|---|---|---|
| **First user** | **You / indie researcher** | Cora is a personal discovery copilot first. Run hot and iterate fast; the human is the live calibration partner; candid "here's why it might be wrong" tone over sales polish. De-risks the engine before facing biotech's brutal credibility bar later. |
| **Primary output** | **Ranked hypothesis briefing** | A ranked feed; each item = hypothesis + convergence evidence + confidence + citations + a proposed (killifish/cell-level) experiment. Ships from Phase 1 (generated on demand) and becomes the autonomous loop's output in Phase 4. |
| **Graph store** | Neo4j *(proposed default)* | Convergence queries are the moat — use a real graph DB, don't fake them in flat tables. |
| **Loop** | Hand-rolled thin orchestration *(proposed default)* | The loop is the product; control it directly rather than inherit a heavy framework's abstractions. |
| **Autonomy loudness** | Scheduled briefing + hot-find ping *(proposed default)* | Not silent, not noisy. |

---

## 2. Why aging/longevity is the right first wedge

| Advantage | Detail |
|---|---|
| **Structured data already exists** | HAGR suite — **AnAge** (lifespans, ~4,000+ species), **GenAge** (aging genes), **DrugAge** (lifespan-extending compounds), **LongevityMap** (human variants). Most trait axes have no such DB; aging does. Lowers the Layer-1 cost dramatically. |
| **Believer-buyers** | Longevity biotech already bought the comparative-biology thesis — Calico hired the naked-mole-rat lab; Altos ($3B), Retro ($1B), NewLimit, BioAge. They *want* "what does the 500-yr shark know." Softens the feasibility doc's "thin market" risk. |
| **A rigorous scaffold exists** | The **12 Hallmarks of Aging** (López-Otín 2023) give the AI a principled structure to organize every finding, instead of ad-hoc links. |
| **Proof-of-method already published** | Kolora et al., *Science* 2021 — comparative genomics across 88 rockfish (11–205 yr lifespans) pulled out DNA-repair + immune longevity genes. That paper *is* a hand-built version of one Cora query. |
| **A built-in test bridge** | African turquoise killifish (*N. furzeri*, ~4–6 mo lifespan) is the tractable model to propose experiments in — closing "where to look → where to test." |

**The core intellectual bet:** rank hypotheses by **convergence depth** — a mechanism that independently shows up in whale + rockfish + quahog is a far stronger signal than any single-species finding. This is Cora's novel scoring function and the thing a generic chatbot cannot do.

---

## 3. Design principles (non-negotiables)

1. **Grounding before autonomy.** Never turn the autonomy dial up faster than the critic + eval can be trusted. Autonomy multiplies whatever you've built — including errors.
2. **Every claim is cited or flagged.** Drop "no hallucinations ever" (impossible — feasibility doc §2). Replace with: *every statement maps to a retrievable source, confidence is stated, and the system says "I can't ground this."*
3. **The moat is data + verification, not the model.** Rent the intelligence (frontier API). Invest in the knowledge graph and the critic.
4. **"Self-motivated" = intrinsic motivation (engineering), not consciousness (philosophy).** Curiosity is a computable reward signal, not a soul.
5. **License-aware from byte one.** Tag every ingested record with reuse rights; never surface commercial output derived from non-commercial-licensed text (feasibility doc §4).
6. **Research/HCP-facing, non-diagnostic.** Stay out of FDA device territory. Capability-aware refusal as a floor (longevity touches some dual-use, e.g. toxins — keep the guardrail).
7. **Human as director, not operator.** The dream is a tireless researcher that briefs you each morning and takes your steer — not "no human."

---

## 4. System architecture (the layers)

```
┌────────────────────────────────────────────────────────────────────┐
│  L6  INTERFACE   chat · hypothesis ledger · graph explorer · briefing │
├────────────────────────────────────────────────────────────────────┤
│  L5  GOVERNANCE  calibration · "I don't know" gate · license + safety │
│      (cross-cutting)            · human-in-the-loop checkpoints        │
├────────────────────────────────────────────────────────────────────┤
│  L3  SELF-PROPELLING LOOP                                             │
│      curiosity → generate → critique → ground → design → rank → ...   │
│                         ▲                                  │          │
│  L4  MEMORY  working · episodic · semantic(=KG) · consolidation       │
├────────────────────────────────────────────────────────────────────┤
│  L2  REASONING CORE   frontier model (hard steps) + cheap model       │
│                       (triage) · tool use over L0                     │
├────────────────────────────────────────────────────────────────────┤
│  L1  INGESTION  connectors · entity resolution · relation extraction  │
│                 · license tagging · runs continuously                 │
├────────────────────────────────────────────────────────────────────┤
│  L0  KNOWLEDGE SUBSTRATE (the moat)                                   │
│      corpus store + vector index  ‖  KNOWLEDGE GRAPH (typed, sourced)  │
└────────────────────────────────────────────────────────────────────┘
```

---

## 5. L0 — Knowledge substrate (the moat)

Two stores, used together (hybrid retrieval):

**(a) Corpus + vector index** — chunked text with embeddings. Sources, license-tagged:
- PubMed abstracts (E-utilities; free, 3–10 req/s cap)
- PMC Open-Access **commercial-tier only** filtered full text
- bioRxiv preprints
- HAGR records (AnAge / GenAge / DrugAge / LongevityMap)
- Molecular: NCBI Gene/RefSeq, Ensembl, UniProt, KEGG/Reactome, STRING, AlphaFold

**(b) Knowledge graph** — the structured differentiator. Minimal schema for the aging wedge:

```
NODES                          EDGES (each carries: source_doc, confidence, license)
─────                          ─────
Species   (NCBI taxon id)      has_lifespan            Species → Phenotype(value)
Gene / Protein                 expresses               Species → Gene
Pathway                        ortholog_of             Gene → Gene
Hallmark (1 of 12)             associated_with_hallmark Gene/Pathway → Hallmark
Intervention / Compound        extends_lifespan_in     Intervention → Species/Model
Phenotype (lifespan,           regulates               Gene → Pathway
  negligible senescence…)      converges_with          Gene/Pathway → Gene/Pathway
Model organism (killifish…)    evidence_supports       Claim → (edge)
Claim / Evidence               contradicts             Claim → Claim
```

The KG is what lets Cora answer *"which DNA-repair gene is independently associated with extreme lifespan in ≥3 unrelated lineages, and does a human ortholog exist?"* — a question no pile of text chunks can answer. **This is the asset competitors can't copy.**

---

## 6. L3 — The self-propelling loop (the autonomy engine)

```
        ┌─────────────────────────────────────────────────────────┐
        │                                                         ▼
  (1) CURIOSITY ──► (2) GENERATE ──► (3) CRITIQUE ──► (4) GROUND ──► (5) DESIGN
   pick frontier      hypotheses      adversarial       cite or        falsifiable
   (gaps, outliers,   from KG +       refute / novelty  kill /         experiment
   new data,          retrieval       / overclaim       confidence     (e.g. killifish)
   high-uncertainty)                  check                  │
        ▲                                                     ▼
        │                                              (6) RANK & LEDGER
   (7) CONSOLIDATE ◄──────────────────────────────────  score = novelty ×
   compress episodic→semantic,                          convergence × grounded ×
   decay stale, re-rank                                 testable × human-priority
                                                              │
                                                              ▼
                                                   (8) SURFACE briefing → human steer
```

**(1) Curiosity / frontier selection — the "self-motivated" engine.** An intrinsic-motivation scorer decides *where to look next*, with no human prompt. Reward = **novelty × information-gain × goal-relevance**. Targets: graph gaps, lifespan outliers with no known mechanism, convergence candidates, freshly-ingested papers, high-uncertainty edges. (Borrowed from RL curiosity: Schmidhuber / ICM / RND.) **Guardrail:** novelty is *bounded by* groundedness so it chases real anomalies, not weird-but-useless trivia (reward-hacking trap).

**(3) Critique — verification is the product.** Independent critic agent(s) attack each hypothesis: *Already known? (novelty vs corpus) · Overstated? (the negligible-senescence calibration problem) · Contradicting evidence? · What would falsify it?*

**(4) Ground.** Each surviving claim → retrievable citations; ungrounded claims flagged/killed; confidence scored. Track citation-faithfulness as a first-class metric (feasibility doc: GPT-4 fabricated ~18% of citations; Galactica died of this).

**(6) Ledger.** Persistent ranked list of live hypotheses — the product's spine and its memory of "what's promising."

---

## 7. L4 — Memory systems (brain-inspired, concrete)

| Memory | Implementation | Borrowed from |
|---|---|---|
| Working | current context window | prefrontal working memory |
| Episodic | log of every hypothesis/critique/decision + outcome (so it never repeats itself, learns what panned out) | hippocampal episodic memory; *Generative Agents* memory stream |
| Semantic | the knowledge graph | cortical semantic memory |
| Consolidation | nightly job: compress episodic→semantic, promote validated edges, decay stale hypotheses, re-rank ledger | hippocampal replay during sleep; DQN "experience replay" |

---

## 8. Eval harness (the feasibility doc's Achilles-heel — build it early)

LBD's documented weakness is *"easy to generate a plausible link, hard to prove it's true."* Cora's credibility depends on measuring this:

1. **Retrospective rediscovery (headline eval).** Freeze the corpus to pre-2015; measure whether Cora re-derives the rockfish DNA-repair finding (2021), naked-mole-rat HA, etc. *Re-discovering known breakthroughs from older data = proof-of-value + demo + regression test in one.*
2. **Held-out association precision/recall.** Hide known gene↔longevity links; measure how many Cora surfaces.
3. **Citation faithfulness.** % of claims with a valid, retrievable, on-point source. Drive fabrication → 0.
4. **Calibration.** Does stated confidence match hit rate?
5. **Novelty audit.** Are "novel" findings truly absent from the corpus, or just missed retrieval?

---

## 9. The autonomy dial (levels)

| Level | Behavior | Gate to unlock |
|---|---|---|
| **L0** | Suggest-only; human approves each step | — |
| **L1** | Autonomous generate + critique; human approves before experiments surface | citation faithfulness passing |
| **L2** | Loop runs continuously; human gets **morning briefing**, can interrupt/redirect | retrospective-rediscovery eval passing |
| **L3** | + self-directed frontier selection + auto-ingestion + self-prioritization (max autonomy, within guardrails) | calibration + novelty audit trusted |

**Recommendation:** ship **L2 as the product default**; treat L3 as the aspiration, unlocked by eval — not by enthusiasm.

---

## 10. Brain-inspired patterns → concrete components

*(Borrow patterns; do NOT study consciousness. Biological fidelity ≠ capability.)*

| Pattern | Where it lands in Cora |
|---|---|
| Intrinsic motivation / curiosity (RL) | L3 step 1 — frontier selection scorer |
| Dual-process (System 1 / 2) | L2 — cheap model triage vs frontier reasoning |
| Metacognition | L3 step 3 + L5 — critic & confidence gate |
| Global Workspace (the *pattern*, not the metaphysics) | L3 orchestration — shared blackboard agents broadcast to |
| Memory + consolidation | L4 |
| *(optional, deep water)* Active inference / free-energy | a principled curiosity objective — bookmark, don't start here |

---

## 11. Reconciling with the existing repo

| Exists now | Fate |
|---|---|
| FastAPI `app.py` | Keep — becomes L6 API |
| ChromaDB + all-MiniLM | Keep for MVP vector index; revisit (pgvector/Qdrant) at scale |
| Groq/Llama-3-8B | Fine for triage tier; frontier API for hard reasoning |
| 6-fact seed JSONL | Replace with real license-aware ingestion (L1) |
| `species/*.py` modules | Fold into KG species nodes |
| `SYSTEM_PROMPT` "no hallucinations" | **Remove** — replace with grounding + critic |
| Known bugs (`DISALLOWED_KEYWORDS`, `dolphin` vs `dolphins`, static mount) | Plumbing — fix when we start Phase 1, not now |
| *(missing)* KG, the loop, memory, eval | **The real work ahead** |

---

## 12. Phased roadmap (sequence, not code)

- **Phase 0 — now:** lock this architecture + data-source list + eval definitions. *(this doc)*
- **Phase 1 — Trustworthy retrieval:** real aging corpus + grounded, citation-verified RAG + honest guardrails + the retrospective-eval harness. Single-shot Q&A, *no loop yet.*
- **Phase 2 — The graph:** build the KG + convergence detection (the differentiator).
- **Phase 3 — The loop:** curiosity → generate → critique → ground → rank → ledger, at autonomy **L1**.
- **Phase 4 — Autonomy:** memory/consolidation + experiment design + **L2 morning briefing**.
- **Phase 5 — Scale:** broaden corpus; push toward **L3** as eval earns it.

---

## 13. Design decisions

**Locked (see §1.1):** first user = you / indie researcher · primary output = ranked hypothesis briefing.

**Proposed defaults (override anytime):** graph store = Neo4j · loop = hand-rolled orchestration · autonomy = scheduled briefing + hot-find ping.

**Still genuinely open:**
- **Briefing granularity** — how many hypotheses per briefing, and how deep is one card by default (one-line claim vs. mini-dossier)?
- **Seed scope** — which species + databases form the Phase-1 corpus? *(Proposal: AnAge / GenAge / DrugAge / LongevityMap + PubMed abstracts for the long-lived set — bowhead whale, Greenland shark, ocean quahog, rockfish, naked mole-rat, hydra, *Turritopsis*, killifish.)*

---

*This is a living document. Nothing here is committed in code; it's the map we argue with before laying track.*
