# Cora — Phase 3 Build Spec: The Curiosity Engine & Self-Propelling Loop (v0.1, design only)

**Date:** 2026-09-08
**Status:** Design target — *no code yet.*
**Parent:** `docs/Cora-Architecture.md` (Phase 3 of §12). Assumes P1 (trustworthy card) and P2 (graph + convergence) are in place.

> **⚠️ Amended by red-team (2026-09-08).** Corrections applied in place and marked **[RT]**; full findings in `docs/Cora-Red-Team.md`. Key changes: the corpus is ~816 abstracts and the frontier is ~108 cells, so the **curiosity scorer is deferred to P5** and Phase 3 becomes an **exhaustive, prioritized, weekly sweep**; the curiosity formula had a units bug; "taste learning" is replaced by explicit steering rules; the breaker must never trip on human silence. The red-team report wins on conflict.

---

## Goal (one sentence)

> Make Cora run **without being prompted**: it decides where to look, generates hypotheses, critiques and grounds them, ranks them into a persistent ledger, learns the director's taste, and does it again — at autonomy **L1** (human approves before experiments surface), architected for L2.

This is the "self-motivated" part of the vision, rendered as engineering: **curiosity is a scoring function over the graph's gaps, bounded by groundedness.** No consciousness required.

---

## 0. The shape in one picture

```
            ┌──────────────────────────────────────────────────────────────┐
            │                                                              ▼
 (1) INGEST ─► (2) FRONTIER ─► (3) GENERATE ─► (4) CRITIQUE ─► (5) GROUND ─► (6) DESIGN
  new docs →    curiosity        retrieve        multi-critic     cite-or-kill    falsifier
  graph edges   engine picks     graph+vector    (5 roles)        (P1 gate)       experiment
                top-k targets    → draft cards                        │
                     ▲                                                ▼
                     │                                        (7) RANK & LEDGER
 (9) LEARN ◄─────────┴──── human feedback ◄── (8) SURFACE ◄──  score · dedupe · decay
  episodic log:            (L1: approve /                        persistent state
  taste, dedupe            reject / redirect)
```

**Steps 3–6 are exactly the Phase 1 flow.** Phase 3 adds a *trigger in front* (2, the curiosity engine) and *state behind* (7–9, ledger + memory). Nothing about *how* a card is made changes — only *who decides to make it.*

---

## 1. The Frontier — where curiosity comes from

The curiosity engine doesn't invent questions from nothing. It **reads the Phase 2 graph** and harvests *frontier candidates* — concrete, graph-anchored things worth looking at:

| Candidate type | What it is | Why it's high-information |
|---|---|---|
| **Graph gap** | Species with an extreme-lifespan phenotype but few/no mechanism edges (Greenland shark, quahog) | "Why does this live so long? We have *nothing*." |
| **Near-threshold convergence** | A mechanism with 2 independent lineages; a 3rd would cross the "strong" bar | A *directed* search — one more lineage flips the verdict |
| **Contradiction** | An unresolved `contradicts` edge | Disagreement is where the corpus is wrong or incomplete |
| **High-uncertainty, high-centrality edge** | Low-confidence edge that many hypotheses depend on | Verifying it de-risks everything downstream |
| **Fresh data** | Newly ingested papers touching the panel | New evidence may move existing hypotheses |
| **Orphan human lever** | A GenAge human aging gene with no cross-species convergence support | "Does *any* long-lived species use this?" |
| **Director interest** | Human-flagged topics | The steer, weighted in explicitly |

**The groundedness bound (critical guardrail):** a candidate must be anchored to *real, sourced* graph structure. Cora cannot get "curious" about a species or gene that isn't in the graph with cited edges. This single rule is what prevents the **novelty trap** — chasing weird-but-useless questions because they're surprising.

---

## 2. The curiosity score (intrinsic reward, formal sketch)

```
[RT — the formula below had a units bug (a product of [0,1] terms minus a token count is always
 negative) and an uncomputable info_gain. Corrected form:]

  gate:      tractability(c) ≥ 15 retrievable passages      // HARD GATE, not a factor
  priority:  novelty(c) × realized_gain(type(c)) × relevance(c)
             // realized_gain = a bandit over candidate types; reward = measured ledger change
             //   (rank movement, edge-confidence delta) × human acceptance, from the episodic log
  cost:      enforced as a per-run budget ($5/run hard cap) — never a score term

[RT — and the whole scorer is DEFERRED TO P5. With ~816 abstracts the frontier is ~108 cells
 (9 species × 12 hallmarks): Phase 3 is an EXHAUSTIVE, PRIORITIZED, WEEKLY SWEEP of that grid.
 Curiosity scoring earns its keep only when frontier > budget.]
```

| Term | Meaning | Guards against |
|---|---|---|
| **novelty** | Not already answered in corpus, not already in the ledger, not already asked (episodic memory) | self-repetition |
| **info_gain** | How much resolving it would *change the graph or the ledger* — crossing a convergence threshold = high; polishing a peripheral edge = low | busywork |
| **relevance** | Proximity to the goal: aging → mechanism → human lever | drift / wandering |
| **tractability** | Does the corpus hold enough evidence to make progress? (annotation-sparse species score low) | sinking cost into unanswerable questions |
| **cost** | Estimated tokens/compute | budget blowup |

*Lineage: this is the "learning progress" idea from intrinsic-motivation research (Schmidhuber / Oudeyer) — reward for questions whose answers would most change what Cora knows — bounded by relevance so it stays on mission.*

**Selection per cycle:** priority queue → pick top-**k** with **diversity** (never spend a whole cycle on one hallmark or one species) and a small **ε-exploration** slice (occasionally take a lower-scored candidate to avoid tunnel vision).

---

## 3. The Hypothesis Ledger — persistent state, the spine

```
LedgerEntry {
  hypothesis_id
  card                      // P1/P2 card schema, unchanged
  state:  proposed | critiqued | grounded | ranked | surfaced
          | human_reviewed | archived | refuted
  score  (+ history)        // convergence × novelty × groundedness × testability × human_priority
  origin: { frontier_candidate, cycle_id }
  lineage: parent_hypothesis_ids[]   // refinements form a tree, not a flat list
  human_feedback: [ { verdict: useful | reject | redirect, note, at } ]
  created_at, updated_at
}
```

**Ledger operations:** insert · re-score · **dedupe/merge** (semantic similarity *and* shared graph edges — two hypotheses over the same edges are one hypothesis) · **decay** (un-refreshed entries lose priority so the ledger doesn't hoard) · archive / refute · surface.

**Hard cap on live entries.** A ledger that only grows is a ledger nobody reads.

---

## 4. The multi-critic — verification made structural

At Phase 3 scale the critic can't be one prompt. It's **five distinct roles**, each a separate pass; a hypothesis must survive all (or carry an explicit flag) to be ranked:

| Critic | Asks | Uses |
|---|---|---|
| **Novelty** | Is this already known? | corpus search + ledger |
| **Overclaim** | Is the claim strength justified by evidence *quality*? (the negligible-senescence hype trap) | P2 phenotype evidence-quality |
| **Phylogeny** | Is the "convergence" actually independent, or inherited? | P2 species tree |
| **Falsifiability** | Is there a concrete experiment that could *refute* it? | model-organism knowledge |
| **Devil's advocate** | Strongest counter-argument + contradicting evidence | `contradicts` edges + retrieval |

Then the **Phase 1 citation gate** (step 5) — every claim cited or killed. The critics judge *reasoning*; the gate judges *sourcing*. Both must pass.

> **[RT] Five critics, one brain — fixed.** Five prompts to the same model family are a persona menu, not independent review: shared pretraining priors, shared popular-science contamination ("immortal lobster"), shared hedge-insensitivity, self-preference bias. **Fixes:** (1) the **verifier and the Overclaim critic run on a different model family** from the generator, with cross-family disagreement rate logged as a first-class metric; (2) a **non-LLM mechanical critic layer** runs first — exact-substring quote check against a hashed passage snapshot, cited doc-ID == passage doc-ID, taxon-ID match, gene-symbol resolution, direction-of-effect, primary-vs-review flag, date sanity — and can fail a card without any model's opinion; (3) the verifier's output is **structured, not boolean** — `{supported: yes | partial | no, claim_strength vs source_strength: weaker | equal | stronger, evidence_type, species + population match}` — and **"stronger" auto-fails** (this is what catches *"ERCC1 expansion **drives** longevity"* verified against *"…**suggesting a possible role**…"*); (4) a **sixth critic — dual-use / safety** — with a written taxonomy (refuse / redact-with-rationale / escalate) and its own eval set, run before anything surfaces; (5) a **confound critic** (does the mechanism track temperature/mass better than residual longevity? — P2 §3). Falsifiability becomes a **binary gate**, removed from the score and the tight card.

---

## 5. Episodic memory — don't repeat yourself, learn taste

Every cycle logs: candidates considered (+ scores), hypotheses generated, critic verdicts, ground results, **human feedback**. Two jobs:

1. **Dedupe** — feeds `novelty` so the same question isn't asked twice.
2. ~~**Taste learning**~~ **[RT] Replaced by explicit steering rules.** Inferred taste fails four ways at once: one director yields ~50–100 labels a month (no class can be defined from that); acceptance measures agreement with one person's priors, so real learning and pure sycophancy are indistinguishable by the design's own metric; the spec's own example ("reject single-gene claims from sparse species") formally trains the loop to *abandon quahog, shark and Turritopsis* — the species the project exists for; and human silence over a weekend would have read as rejection.
   **Instead:** ~5 **director-written, editable rules** (min independent lineages · exclude species with <N abstracts · hallmark focus · gene-vs-pathway · species watch-list), shown on every card as "why you're seeing this." Learned weights only after >500 labels. Any preference signal influences **ordering/attention only** — never groundedness, verification, or confidence — and may downweight a *claim-quality class*, **never a species class**; a fixed sweep quota is reserved for sparse species. Feedback is split **reject: wrong** vs **reject: uninteresting** (only the former may inform the truth pipeline; neither reaches the verifier). Each briefing carries a floor of un-preference-weighted slots and a small **blinded probe** (items of known truth status) so a **sycophancy index** — does acceptance track truth or valence? — is published beside acceptance rate.

Full memory tiers + nightly consolidation are Phase 4. Phase 3 needs only the episodic log and the feedback→scorer path.

---

## 6. Autonomy mechanics — L1 now, L2 ready

| | L1 (Phase 3 target) | L2 (Phase 4) |
|---|---|---|
| Steps 1–7 | autonomous | autonomous, scheduled |
| Surfacing experiments | **requires human approval** | auto-delivered morning briefing |
| Human role | approve / reject / redirect per item | interrupt & redirect; hot-find ping |
| Feedback path | → curiosity scorer + ledger | same, plus consolidation |

The architecture is identical across levels; L1→L2 is a **config change** (who pulls the surface trigger), unlocked by the eval gates in the architecture doc — never by enthusiasm.

---

## 7. Guardrails — the failure modes designed in

| Failure mode | Guardrail |
|---|---|
| **Novelty trap / reward hacking** (chases weird-but-useless) | groundedness bound (§1) + relevance + tractability terms (§2) |
| **Drift** (wanders off aging) | relevance anchor to goal; diversity constraint; director interest re-weighted each cycle |
| **Runaway volume** | per-cycle budget (k candidates, max hypotheses); ledger cap + decay; groundedness threshold to *enter* ranked state |
| **Confident nonsense at scale** | multi-critic + citation gate before anything ranks; **fabrication rate tracked per cycle** |
| **Cost blowup** | tiered models (cheap: frontier scoring, NER, triage · frontier: generate/critique only); per-cycle token budget; cost term in curiosity |
| **Self-repetition** | episodic dedupe |
| **Silent failure** | cycle health metrics logged + surfaced (§8) |

### The circuit breaker (first-class component)
The loop **pauses itself and pings the director** if any of these trip:
- fabrication rate > threshold (the anti-Galactica tripwire)
- critic survival rate collapses (generation gone bad) *or* spikes (critic gone lax)
- cost per cycle exceeds budget
- ~~N consecutive cycles surface nothing the human accepts~~ **[RT] removed** — it punished disengagement (a quiet weekend read as rejection) *and* disagreement (sustained correct-but-unwelcome output). **Human silence is no signal.** The breaker trips on **system signals only**, and on **canaries**: every run injects known-fabricated, known-supported, hedge-stripped, wrong-species and injected-passage items — a surviving canary trips the breaker immediately (an *absolute* tripwire, since the other conditions are change-detectors blind to a constant bias). Thresholds are **control limits** (rolling baseline ± 3σ over the last 20 runs), re-baselined and version-stamped on any prompt or model change. Sustained *100 %* acceptance is the alarming case.

**Autonomy with a kill switch it pulls on itself.** This is what makes "as autonomous as possible" safe to actually turn on.

---

## 8. Metrics — how we know the loop works

| Metric | What it tells us |
|---|---|
| **Survival rate** (generated → survives critic + gate) | too high = critic lax; too low = generation wasteful |
| **Fabrication rate / cycle** | the circuit-breaker input |
| **Novelty rate** (surfaced items not already in corpus/ledger) | is it finding *new* things? |
| **Human acceptance rate** | the ground-truth "taste" signal |
| **Cost per accepted hypothesis** | the efficiency metric that matters |
| **Unprompted rediscovery** *(headline)* | see §9 |

---

## 9. The Phase 3 eval — *unprompted* rediscovery

Phase 1's eval asked: *"if we ask Cora about rockfish, does it find the DNA-repair mechanism?"*
Phase 3's eval asks the harder, more important question:

> **Left alone on a pre-2020 corpus with no prompt, does the curiosity engine *decide on its own* to look at rockfish — and surface the 2021 finding unasked?**

This tests the *curiosity engine*, not just generation. Passing it means Cora doesn't just answer questions — it **asks the right ones**. Run against the full P1 gold set (rockfish, naked mole-rat, bowhead, *Turritopsis*); measure which are surfaced unprompted, at what rank, at what cost.

---

## 10. Definition of done (Phase 3)

1. Curiosity engine harvests graph-anchored candidates, scores them, and selects a diverse top-k **with no prompt**.
2. Loop runs **N cycles unattended** at L1; ledger persists across cycles; dedupe + decay verified.
3. Multi-critic + citation gate: nothing enters the ranked ledger ungrounded; fabrication rate tracked per cycle; **circuit breaker fires in tests**.
4. Human feedback **measurably shifts** curiosity scoring (taste learning demonstrated).
5. **Unprompted rediscovery:** loop on the pre-2020 corpus surfaces ≥1 gold discovery *without being asked*.

When these hold, Cora is genuinely self-propelling *and* still honest — and we've earned Phase 4: memory consolidation, the L2 morning briefing, and the hot-find ping.

---

*Living document. No code committed — the target we build against.*
