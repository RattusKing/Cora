# Cora — Phase 4 Build Spec: Memory Consolidation & L2 Autonomy (v0.1, design only)

**Date:** 2026-09-08
**Status:** Design target — *no code yet.*
**Parent:** `docs/Cora-Architecture.md` (Phase 4 of §12). Assumes P1–P3 in place: trustworthy card, graph + convergence, self-propelling loop at L1.

> **⚠️ Amended by red-team (2026-09-08).** Corrections applied in place and marked **[RT]**; full findings in `docs/Cora-Red-Team.md`. Key changes: cadence is **weekly, driven by evidence rate** (not nightly); consolidation is **deferred until >200 ledger entries**; "Reflect" creates claims and must be split into `derived_facts` vs `conjectures`; the **briefing artifact, not the card, is the unit of verification**; the hot-find ping waits for a strong-convergence event to occur organically. The red-team report wins on conflict.

---

## Goal (one sentence)

> Let Cora run **unattended on a schedule**, remember and compress what it learns without losing provenance, hand the director a **ranked morning briefing**, interrupt only for genuinely hot finds — and be steered rather than approved.

L1 → L2 is architecturally a config change (P3 §6). Phase 4 builds the three things that make L2 *livable*: **memory that doesn't lie, a briefing worth reading, and a ping worth answering.**

---

## 0. The daily rhythm

```
 night                                   morning                  day
 ─────────────────────────────────────── ──────────────────────── ────────────────────────
 scheduled cycles (P3 loop, L2)   ──►   CONSOLIDATE  ──►  BRIEFING  ──►  director steers:
 ingest · frontier · generate ·          promote · compress ·       ranked cards,  accept / reject /
 critique · ground · rank                decay · reflect            what changed,  redirect / pause
                                                                    frontier preview,    │
        ▲                                                           health metrics       │
        └────────────── HOT-FIND PING (rare, interrupt-driven, budgeted) ◄──────────────┘
                                         taste feedback → curiosity scorer + ledger
```

---

## 1. Memory consolidation — the nightly job

Consolidation compresses the P3 episodic log into semantic memory so working context stays small and the graph keeps improving. Four operations:

| Op | What it does | Borrowed from |
|---|---|---|
| **Promote** | Hypotheses that survived the multi-critic *and* earned human acceptance → their supporting graph edges get a confidence boost; well-supported claims become consolidated knowledge | hippocampal → cortical transfer |
| **Compress** | Many episodic entries on one topic → a *semantic summary* ("what we know about DNA-repair convergence") so reasoning never has to re-read raw logs | memory abstraction |
| **Decay** | Stale hypotheses archived; low-value episodic detail pruned under a retention policy (*keep provenance, drop verbosity*) | forgetting as a feature |
| **Reflect** | Generate higher-level insights across cycles — "recurring theme: pathway-level convergence; director consistently rejects single-gene claims from sparse species" — stored as semantic memory, fed to the curiosity scorer and the briefing | *Generative Agents* reflection |

### The rule that keeps memory honest: **no laundering**
Consolidation **cannot create claims** — it can only compress claims that already exist with sources. Every semantic summary carries the **ledger / edge IDs it was derived from**, so any sentence in memory traces back to a citation. Without this, consolidation becomes a machine that turns unverified hypotheses into "things Cora knows." Same cited-or-killed discipline as the card, applied to memory.

Originals are **immutable**; summaries are *derived views*, periodically re-derived from source to prevent compression drift.

> **[RT] Two corrections to this section.**
> **(a) Deferred.** With ~816 abstracts the ledger stays small; below **~200 entries the ledger *is* semantic memory** and consolidation has nothing to compress — it would become exactly the laundering machine this rule fears. Consolidation ships only past that threshold.
> **(b) "Reflect" creates claims, so the no-laundering rule as written was honor-system.** "Generate higher-level insights across cycles" asserts what no single source states; its derivation IDs point at *hypotheses* and the summary drops their modality; and §2 placed such summaries *above* the evidence in every later context window — a closed positive-feedback loop (three weak DNA-repair hypotheses → "recurring theme: DNA repair" → the sweep over-samples DNA repair → "DNA repair is the primary convergent mechanism," sourced to twelve hypothesis IDs and zero papers). **Fix:** split memory into **`derived_facts`** — a lossless restatement of ≥1 verified claim, mechanically re-gated by re-running the citation check on the summary sentence against the union of its sources — and **`conjectures`** — Reflect output, which never appears in a briefing as a statement, never sits above evidence in working memory, and influences prioritization only with a capped, decaying weight. Every summary sentence inherits the **modality of its weakest input** (a summary of hypotheses is a hypothesis). Reflections may not cite reflections. The sampled audit becomes **100 % mechanical trace-checking plus a stratified human *support* sample.**

---

## 2. Working-memory assembly (context engineering)

Per task, the model's context is built from — in priority order — *never* raw logs:

1. the goal + director's current interests
2. relevant **semantic summaries** (from §1) with their derivation IDs
3. the top relevant **ledger entries**
4. the specific **retrieved evidence** (graph edges + vector passages) for this task

This is what keeps per-cycle cost flat as Cora's history grows — the history is compressed, not re-read.

---

## 3. The morning briefing — the L2 deliverable

A ranked, **hard-capped** document (top-N; N is a config, default small). Sections, in order:

| Section | Content |
|---|---|
| **What changed** | Hypotheses that moved rank since last briefing, and *why* (new evidence, resolved contradiction, critic re-verdict) — the highest-value section |
| **New this cycle** | New tight cards (P1 schema), each with a one-line "why it's here" |
| **Your flags** | Status of anything the director marked "dig deeper" |
| **Contradictions found** | Unresolved `contradicts` edges surfaced this run |
| **Frontier preview** | What the loop **plans to look at next** — so the director can redirect *before* compute is spent |
| **Health** | cycles run · fabrication rate · critic survival rate · cost vs budget · circuit-breaker state |

**Inline feedback affordances** on every item: `accept · reject · redirect · dig deeper` → flow to the ledger and the P3 taste-learning path.

**A missing briefing is itself an alert.** If the morning briefing doesn't arrive, something failed overnight — the absence is the signal, so silent failure can't hide. **[RT]** The failed process cannot report its own absence: an **external dead-man's switch** (a healthcheck ping *from* the briefing job) is what actually notices.

> **[RT] The briefing artifact — not the card — is the unit of verification.** The citation gate covers card claims and edges, but "What changed … and *why*" (ranked first), "why it's here," contradictions prose, frontier rationale and Reflect insights are none of those — so "no ungrounded claim reaches the briefing" was false by construction. **Fix:** every sentence in a briefing is one of (a) a **verified claim ID** with its citation, (b) a **computed-value ID** (metric or score + query hash), or (c) visibly-styled **"unverified narrative"** excluded from all trust metrics. "What changed" is **templated from a structured diff** over IDs (rank delta, edge added, critic verdict changed) — no free prose. Internal re-scoring never counts as a change.

---

## 4. The hot-find ping — interrupt-driven, budgeted

The ping exists for the rare thing that shouldn't wait until morning. It must be **high-precision or it becomes noise**, so it's gated twice:

**Trigger criteria (any one):**
- a hypothesis crosses the **strong-convergence threshold** (≥3 independent lineages, high groundedness)
- a resolved contradiction **flips** a top-ranked hypothesis
- a hypothesis scores above a "big finding" bar
- the **circuit breaker trips** *(a halt, not a find — different channel, always delivered)*

**Ping budget:** default max **1 non-critical ping per day**; excess candidates roll into the morning briefing. The director can raise or lower the budget. Circuit-breaker halts are exempt.

> **[RT] Deferred, and two guards added.** The ping ships only after a strong-convergence event has occurred *organically* at least once — otherwise it's a feature for an event that has never happened. When it does ship: **(a)** a ping requires **full-text** support and ≥1 **primary** observation per lineage — abstract-only or review-echo support can never trigger it; **(b)** **provenance trust tiers** — a preprint or auto-ingested document can never *alone* raise an edge above provisional, contribute to a ping, or be sole support for a promoted edge (the prompt-injection defense: an attacker's bioRxiv post must not be able to wake you at 3 am, nor deliberately spike the fabrication rate to trip the breaker, which — being exempt from the budget — is a guaranteed-delivery channel). A resolved contradiction is a ping trigger **only** when resolved by a human or a new primary result, never by an LLM-only verdict.

---

## 5. Steering at L2 — director controls

At L2 the human no longer approves items; they *steer the process*. Controls:

| Control | Effect |
|---|---|
| **Pause / resume** | halts scheduled cycles |
| **Redirect** | sets director interests → weighted into the curiosity scorer next cycle |
| **Veto** | archives a hypothesis + logs the class for taste learning |
| **Budget** | daily token/compute cap; ping budget; briefing cap |
| **Autonomy level** | L2 ↔ L1 (or up to L3 when its gate is met) |

Every control action is logged to episodic memory and feeds taste learning — steering *is* training signal.

---

## 6. Scheduling & budget

- **Cycle cadence** — **[RT] weekly, driven by evidence rate.** The non-human panel gains ~1.5 aging abstracts per week; nightly cycles over a static corpus produce re-scoring jitter, not news. A run fires when something *evidence-caused* moved, or weekly, whichever first. Each run bounded by the P3 per-run budget ($5/run hard cap).
- **Daily budget** — hard cap across cycles; the loop stops for the day when spent, and says so in the briefing.
- **Consolidation** — once nightly, after cycles, before the briefing.
- **Briefing time** — configurable; delivery via the UI, later optional channels (email / Slack).

---

## 7. Guardrails — Phase 4 failure modes

| Failure mode | Guardrail |
|---|---|
| **Memory laundering** (summaries lose provenance, hypotheses become "facts") | no-laundering rule (§1): summaries carry derivation IDs; consolidation cannot create claims |
| **Compression drift** (repeated summarization distorts) | originals immutable; summaries are re-derived views |
| **Ping fatigue** | high trigger bar + daily ping budget (§4) |
| **Briefing bloat** | hard cap; "what changed" first; ranked |
| **Runaway overnight cost** | daily budget; P3 circuit breaker carries over |
| **Silent overnight failure** | briefing always includes health; *missing briefing = alert* |
| **Steering ignored** | every control action logged; briefing shows how the last redirect changed the frontier |

---

## 8. Metrics

| Metric | What it tells us |
|---|---|
| **Briefing acceptance rate** | are the top-N items worth reading? |
| **Time-to-first-useful item** | is ranking working? |
| **Ping precision** | fraction of pings the director rated worth the interrupt |
| **Context efficiency** | tokens per cycle before vs after consolidation |
| **Provenance audit pass rate** | sample of semantic summaries → can every sentence be traced to a source? |
| **Cost per day vs budget** | the constraint that makes L2 sustainable |

---

## 9. Phase 4 eval — the unattended week

Run the loop **unattended for a multi-day window** (real or simulated) on the pre-2020 corpus at L2. Pass if:
- the **gold discoveries** (P1 gold set) appear in morning briefings *unprompted*, at a reasonable rank
- **no ungrounded claim** appears in any briefing (the P1 gate holds under autonomy)
- **ping precision** is high and the ping budget is respected
- **cost stays within** the daily budget
- the **provenance audit** on consolidated memory passes

This is the L2 gate from the architecture doc, made concrete: autonomy is unlocked by surviving a week alone, not by wanting it.

---

## 10. Definition of done (Phase 4)

1. Nightly consolidation runs — promote / compress / decay / reflect — with **every summary traceable** to sources (audit passes).
2. Scheduled cycles + **morning briefing** auto-delivered: capped, ranked, "what changed," frontier preview, health.
3. **Hot-find ping** live with high bar + budget; precision measured.
4. Director **steering controls** work and demonstrably feed taste learning.
5. **Unattended week** eval passes (§9).

With P4 done, Cora is what the vision described: **a tireless researcher that runs all night, remembers honestly, briefs you in the morning, interrupts only when it matters — and takes your steer.** What remains is Phase 5 (scale the corpus, push toward L3 as eval earns it) — and before any of it, the red-team.

---

*Living document. No code committed — the target we build against.*
