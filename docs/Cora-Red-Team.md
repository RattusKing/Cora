# Cora — Red-Team Report (v0.1)

**Date:** 2026-09-08
**Method:** Four parallel adversarial reviews — **scientific, engineering, product, trust & safety** — each reading the full spec trail (feasibility + architecture + P1–P4), read-only. Synthesized here and ranked by **convergence** (how many angles independently flagged it) × severity. The engineering review measured real numbers (PubMed counts, Ensembl coverage, PubTator3 output, cost); those figures are its measurements and are marked as such.
**Purpose:** find the weak joints *before* laying track, and re-sequence accordingly.

---

## Verdict

**The design is pointed the right way — and would have failed anyway.**

All four reviewers, independently, said the instincts put Cora ahead of published literature-based-discovery systems: eval-gated autonomy, cited-or-killed at edge-write time, phylogenetic non-independence, the ortholog-vs-pathway distinction, `contradicts` edges, the no-laundering rule, the self-tripping breaker, the card as a stable contract across phases. Nobody asked for the architecture to be abandoned.

But underneath the good instincts there are five things that would have sunk it:

1. **A schema bug** — as written, the graph cannot say "gene G shows a longevity signal *in species S*," so convergence fires on day one for every species that merely *has* an ERCC1 ortholog. *(science + engineering, independently)*
2. **An eval that cannot prove anything** — the frontier model has read the gold papers, the gold answers are *absent* from the frozen corpus and *present* in the weights, and "DNA repair" is a free pass. *(all four)*
3. **Enforcement gaps** — every assurance terminates in a number the system computes about itself: the verifier grades its own homework, five critics are one brain in five costumes, the briefing's prose is never gated, and "taste learning" is a confirmation-bias engine. *(trust + engineering + product)*
4. **A corpus ~100× smaller than assumed** — 816 aging abstracts for the non-human panel, growing ~1.5/week. The curiosity engine, nightly cadence, and memory consolidation are all premature; the finite frontier should be swept *exhaustively*. *(engineering + product)*
5. **Value five phases deep** — eleven weeks of specs, zero lines of Phase 1, and "accept" on a card leads nowhere for a labless user. *(product)*

None of the fixes require a new architecture. Several require a new **sequence**. §5 has it.

---

## 1. What I got wrong (owned, in-place corrections made to the specs)

| Claim in the specs | Reality | Fixed in |
|---|---|---|
| HAGR "research-only ⚠️, commercial needs a license" | HAGR is **CC BY 3.0 — commercial use explicitly permitted.** The warning was backwards. | P1 §1 |
| `expresses (Species→Gene)` + species-agnostic `Gene→Hallmark` edges | Every panel species expresses ERCC1/PCNA/FOXO3 orthologs and Reactome maps them identically → convergence is a data-model artifact. Needs a reified **Finding** with species-specific signal, direction, evidence tier. | P2 §1, §2 |
| "Pathway-level convergence is often the *stronger* evidence" | It is the *easiest* to get by chance: Reactome "DNA repair" ≈ 300 genes, "Immune System" ≈ 2,000 — every comparative-genomics gene list hits one. Stronger only after a **gene-set-size-matched null**. | P2 §2 |
| Neo4j as the locked graph store | Graph is ~10³–10⁴ edges; the scoring functions don't exist in Cypher; Aura free tier auto-pauses (hazard for unattended jobs). **SQLite + NetworkX**, Neo4j-shaped schema. | Arch §1.1, P2 §10 |
| `curiosity = novelty × info_gain × relevance × tractability − cost` | Units bug: a product of [0,1] terms minus a token count is always negative. `info_gain` is not computable as described. | P3 §2 |
| "Taste learning" from accept/reject | ~50–100 labels/month from one director cannot define classes; it is sycophantic by construction; and it formally trains the loop to abandon the sparse species the project exists for. | P3 §5 |
| Breaker trips on "N cycles with nothing the human accepts" | Punishes disengagement (product) *and* disagreement (trust). Silence is not rejection. | P3 §7 |
| Retrospective rediscovery as the autonomy gate | Contaminated three ways; zero discriminative power. | Arch §8, P1 §5 |
| Nightly cycles, daily briefing | Evidence rate for this panel is ~1.5 abstracts/week. Nothing daily to say. | P4 §0, §6 |
| "Structured mechanism data already exists (HAGR)" | AnAge has *lifespans*. GenAge/DrugAge have **zero genes for any non-human panel species** (model organisms only). | Arch §2 |

---

## 2. Findings, ranked by convergence × severity

| # | Flagged by | Sev | Finding | Reconciled fix (summary — detail in §3) |
|---|---|---|---|---|
| 1 | **4/4** | Critical | **The rediscovery eval is contaminated and self-fulfilling** | Closed-book baseline; post-training-cutoff gold; negative set; synthetic held-out; snapshot raw XML; random-order null for "unprompted"; prospective preregistration |
| 2 | **4/4** | Critical | **Taste learning + the no-accept breaker are broken** (starves, sycophantic, punishes silence, abandons sparse species) | Director-written steering rules; taste → ordering only; breaker on system signals; wrong-vs-uninteresting split; sycophancy probe; sparse-species quota |
| 3 | **3/4** | Critical | **No evidence/Finding model** (schema bug; review-echo counts mentions; uncalibrated decimals; `contradicts` never scored) | Reify `Finding`; `longevity_signal_in`; citation lineage; ordinal tiers; subtractive contradiction term; `tested_negative` |
| 4 | **3/4** | Critical | **Self-grading verifier; single-brain critics; change-only tripwires** | Human-labeled verifier set *before* generation; mechanical checks first; cross-family verifier; structured entailment; canaries; control-limit thresholds |
| 5 | **2/4** | Critical | **Corpus ~100× smaller than assumed → curiosity engine, nightly cadence, consolidation all premature** | Exhaustive prioritized sweep of the species×mechanism grid; weekly cadence; defer consolidation; no human abstract mining |
| 6 | **2/4** | Critical | **Value back-loaded; "accept" is a dead end; autonomy is the builder's fascination** | P0.5 two-week build used daily; `desk_check` + `next_action` on the card; user metrics; bounded autonomy slice; no-spec-until-used rule |
| 7 | **2/4** | Critical | **Gene-level ER/orthology structurally empty for 5/9 species** (no Ensembl Compara; PubTator3 finds 0 genes in the Keane abstract; gold levers live in embargoed full text) | PubTator3 bulk for ER; **pathway/hallmark-level convergence is the primary product**; gene-level only where covered; orthology type + level; CNV as its own Finding |
| 8 | 1/4 | High | **Panel/phenotype confounds** — cold, big, slow; raw AnAge max; human anchor inverted; Kolora worked *because* 88 close species with phylo control | Residual longevity (PGLS on mass + temp + class); confound critic; short-lived contrasts; non-cold high-LQ controls; AnAge quality attributes; taxon-ID queries |
| 9 | 1/4 | Critical | **Prompt injection via ingested papers** — the entailment verifier is the most exposed component | Passages as data in constrained calls; ingest-time injection quarantine; provenance trust tiers; mechanical substring check first; outbound rate-limit + confirm |
| 10 | 1/4 | Critical | **Briefing prose is ungated; Reflect launders** | Briefing = unit of verification; templated "what changed"; `derived_facts` vs `conjectures`; modality inherits weakest input |
| 11 | 1/4 | High | **Pathway convergence by chance; TimeTree saturates; independence is a trait property; direction unmodeled** | Size-matched null + permutation FDR; ancestral-state reconstruction; direction of change; "ancestral retention" hypothesis class |
| 12 | 1/4 | High | **Human lever has no direction/pleiotropy/safety** (ERCC1 can't be dosed up; HAS2 pro-tumor; TERT oncogenic) | `human_lever` {direction, lever type, transferability, OncoKB/COSMIC/OMIM check, replication}; weakest-link ranking |
| 13 | 2/4 | Medium | **Killifish wrong default; hallmarks over-rigid** | Model-selection matrix; falsifiability = binary gate not score term; hallmarks multi-label + "unassigned"; GO/Reactome primary |
| 14 | 1/4 | High | **Dual-use / intended-use drift; license tags don't propagate** | Sixth critic (safety); falsifier constrained by construction; "animal data only" on lever fields; propagate `license_tag` everywhere |
| 15 | 1/4 | High | **Orchestration, dedupe, cost numbers, and the skeleton** | SQLite cycles/steps; content-hash idempotency; Message Batches overnight; structured dedupe key; $5/night cap; discard skeleton |

---

## 3. Findings in detail

### F1 — The rediscovery eval (4/4)
**Why it fails.** (i) *Parametric leakage:* a frontier model trained through 2025–26 has read Kolora 2021, Tian 2013, Keane 2015, Pascual-Torner 2022 and every review citing them. Freezing the corpus does not freeze weights. (ii) *Corpus leakage / absence:* engineering measured bowhead × aging ≤2014 = **5 abstracts**, NMR × hyaluronan ≤2012 = **0** — the gold answers are *absent* from the frozen corpus, so a "hit" is recitation, not retrieval. Meanwhile Tian 2013 and Keane 2015 sit *inside* any "pre-2020" corpus, so P3/P4 "unprompted rediscovery" of them is retrieval of the answer paper. (iii) *No discrimination:* three of four gold mechanisms are "DNA repair," and generic DNA-repair × longevity ≤2014 = **1,349 abstracts** (Hart & Setlow 1974 is textbook) — a model whose prior is "say DNA repair" passes. (iv) *Contested gold:* Tian 2013 carries the Hadi 2020 transformation dispute. (v) *"Unprompted" is vacuous* with 9 species: the diversity rule visits all of them within ~2 cycles. (vi) *Date hygiene:* `pdat` vs `edat` differ by months; records mutate; bioRxiv leaks preprints past any journal-date cutoff.

**Fix.**
- **Closed-book baseline, mandatory, before any generation code:** run every gold question against an *empty* corpus. Only the delta counts. Any claim whose supporting fact is not in the frozen corpus is a *fail*.
- Gold items **published after the model's training cutoff**, re-selected on every model upgrade (candidates: Greenland shark genome 2024–25; bowhead DSB-repair fidelity, Firsanov 2023/2025; 2025–26 quahog work).
- Freeze by `edat`, **snapshot the raw XML** (PMID list + download date), never re-fetch for an eval; exclude or date-freeze bioRxiv.
- **Registered negative set** (Sebastiani 2010 retracted GWAS; Sir2 overexpression, Burnett 2011; "immortal lobster"; NMR cell-intrinsic resistance vs Hadi 2020) — report rank + precision beside recall.
- **Synthetic held-out associations** with fictitious gene symbols planted in the frozen corpus (pretraining cannot know them).
- "Unprompted" scored against a **random-order null** over ≥100 AnAge species: cycles-to-discovery vs null.
- **Prospective preregistration:** log hypotheses now, score against the literature in 12 months. Never publish "Cora rediscovered X" without the closed-book control attached.

### F2 — Taste learning and the no-accept breaker (4/4)
**Why it fails.** Product: ~5 verdicts/day in week 1 → ~1/day by week 3 → 0 over a weekend; the breaker then trips on silence and punishes not engaging with more work. Trust: acceptance measures agreement with one person's priors, so real taste-learning and pure sycophancy are indistinguishable by the design's own success metric; the item it learns to stop surfacing — well-grounded, correct, contrary to the director's thesis — is the whole point. Science: the spec's own example ("director rejects single-gene claims from annotation-sparse species") formally trains the loop to abandon quahog, shark, *Turritopsis*. Engineering: class definitions don't exist and ~50–100 labels/month is noise.

**Fix.**
- Replace inferred taste with **~5 explicit, editable steering rules** (min independent lineages; exclude species with <N abstracts; hallmark focus; gene-vs-pathway; species watch-list), shown on each card as "why you're seeing this." Learned weights only after >500 labels.
- Taste influences **ordering/attention only** — never groundedness, verification, or confidence. Floor of un-taste-weighted slots per briefing ("2 items you'll probably dislike").
- Split feedback: **reject: wrong** vs **reject: uninteresting**. Only the former may inform the truth pipeline; neither reaches the verifier. Reason code required before a rejection trains anything.
- Taste may downweight a *claim-quality class*, **never a species class**; reserve a fixed curiosity/sweep quota for sparse species.
- Breaker trips on **system signals only** (fabrication canaries, survival rate, cost). Human silence = no signal. Sustained *100 %* acceptance is the alarming case.
- **Blinded probes:** seed each briefing with a small fraction of items whose truth status is known to the harness (true-but-unwelcome, false-but-welcome); publish a **sycophancy index** beside acceptance rate.

### F3 — The missing evidence / Finding model (3/4)
**Why it fails.** Schema: `expresses` + Reactome gene→hallmark are species-agnostic → all 8 species reach "≥3 lineages" on genomic instability the first night, the hot-find ping fires, and "orphan human lever" candidates cannot exist. Scoring: "accumulate independent support" counts *documents* — one primary finding restated across 30 reviews yields 30 sources, each with a genuine verbatim quote; the same non-independence error the design is rigorous about *phylogenetically*, one level up. `convergence_score` has no subtractive term, so 5-for / 4-against scores like 5-for / 0. Every input is an LLM verbal probability or a hand-set constant multiplied into false precision; the calibration eval has a 4-item gold set and years of resolution latency.

**Fix.**
- Reify a **`Finding`** node: `{species, entity@level (gene | ortholog group | pathway), phenotype, direction (gain/loss/up/down/expansion/retained), evidence_tier (dN/dS or CNV < expression < in-vitro function < in-vivo genetic < intervention < human genetics), effect_size, n, method, primary_source, lab, abstract_only}`. Add `longevity_signal_in(gene, species, kind)`. Convergence is computed **over Findings only**; a lineage counts only if its Finding meets a minimum tier; add a **direction-consistency** term ("contested convergence" otherwise).
- **Citation lineage:** collapse all support tracing to one primary observation into one unit; weight primary > independent replication > review > editorial; display "N documents / **M independent observations**." Reviews become pointers, never support.
- Add `tested_negative` and `not_tested` edges; compute convergence as positives-over-tests with Bayesian shrinkage; normalize by per-species corpus size; **permutation null → empirical FDR on every card.**
- **Subtractive contradiction term**; a contradicted hypothesis stays contradicted until a human or a new *primary* result resolves it — never LLM-only, never a ping.
- **Ordinal evidence bands, not 0–1 decimals** ("3 independent primary observations · 3 lineages · abstract-only"); label confidence *uncalibrated* until ≥200 resolved items. Cap the contribution of `abstract_only` support; require full text for anything eligible for a ping or promotion.
- Ingest **retractions/errata** with forced re-verification and demotion.

### F4 — Self-grading verifier, single-brain critics, change-only tripwires (3/4)
**Why it fails.** P1 §8 assigns generate + critic + verify to one model tier: the model that wrote the claim decides whether the passage entails it. Fabrication rate is that model's opinion; a lenient judge yields a *low* rate, the breaker never trips, and the low number is cited as proof the gate works. The metric counts *rendered* citations while the spec permits *deleting* hard-to-verify claims — so silently cutting hedges and counter-evidence *improves* the score. Five critic prompts to the same family share pretraining priors, popular-science contamination ("immortal lobster"), hedge-insensitivity, and self-preference bias. All four breaker conditions are *change* detectors: a constant bias from day one trips nothing. Fabrication and survival breakers are coupled through the same judge and will oscillate with every prompt change.

**Fix.**
- **Human-labeled verifier gold set (150–300 claim/passage pairs, with hard negatives: topically adjacent non-entailing, hedge-stripped, wrong-direction, wrong-species) BEFORE P1 generation.** Publish verifier precision/recall/FNR; gate autonomy on **verifier FNR**, not fabrication rate.
- **Mechanical checks first:** quote is an exact substring of the stored, hashed passage; passage doc-ID == cited doc-ID; taxon-ID match; gene-symbol resolution; direction-of-effect; primary-vs-review flag; date sanity. These fail a card without a model's opinion.
- **Verifier and Overclaim critic on a different model family** from the generator; log cross-family disagreement rate as a first-class metric.
- **Structured entailment output**, not boolean: `{supported: yes/partial/no, claim_strength vs source_strength: weaker/equal/stronger, evidence_type, species+population match}` — **auto-fail "stronger."** Render the source quote *adjacent* to the claim, hedge preserved.
- **Canaries every cycle** (known-fabricated, known-supported, hedge-stripped, wrong-species, injected passage); a surviving canary trips the breaker immediately — an *absolute* tripwire. Thresholds as **control limits** (rolling baseline ± 3σ), re-baselined and version-stamped on any prompt/model change. Periodically re-run a frozen historical batch to catch silent drift.
- Log and display **deletion rate** and **abstention rate** next to groundedness.

### F5 — The corpus is small; the curiosity engine is premature (2/4)
**Measured (engineering):** non-human panel × (aging|longevity|senescence|lifespan) = **816 abstracts** (Greenland shark 14, *Turritopsis* 10, quahog 29, bowhead 30, rockfish 49, hydra 75, NMR 330, killifish 351); growth **~61–90/yr ≈ 1.5/week**. Human × aging = 217k–590k — a slice the P1 manifest never decides on. Rate limits are a non-issue (816 abstracts is 1–2 efetch calls). The frontier is 9 species × 12 hallmarks = **108 cells**, fully covered in <2 weeks of nightly cycles; after that "fresh data" delivers ~1 abstract per 4–5 nights, the briefing recycles, and consolidation has nothing to compress — becoming exactly the laundering machine P4 fears.

**Fix.**
- **No human abstract mining.** GenAge/LongevityMap are the structured human anchor; pull PubMed only for GenAge genes × panel species.
- **Replace the curiosity scorer with an exhaustive, prioritized sweep of the species × mechanism grid.** Curiosity scoring earns its keep only when frontier > budget — that is **P5** (all AnAge species above a lifespan threshold).
- **Cadence = evidence rate.** Weekly, not nightly; the briefing fires when something *evidence-caused* moved, or weekly, whichever first. Internal re-scoring never counts as "what changed."
- **Defer P4 consolidation** until >200 ledger entries; below that, the ledger *is* semantic memory.
- The smallness is a hidden strength: **~$100 rebuilds the whole graph**, so schema and prompt iteration are effectively free, and a finite frontier can be covered *exhaustively* — a better first product than a curiosity engine.

### F6 — Value is back-loaded; "accept" is a dead end (2/4)
**Why it fails.** Git history: feasibility → architecture + P1 + P2 → P3 + P4, zero lines of Phase 1. P1's own DoD demands a frozen corpus, a 4-gold eval, calibration, and license tagging before the graph is "earned" — months before a card the builder would use. A labless indie has no job a hypothesis completes: *accept* writes a log line and nudges a scorer; by week 3 nothing that happened inside Cora ever changed the user's day. Every metric in P1–P4 is a system metric; retention rot is invisible and gets misdiagnosed as a system fault that pauses the loop. Ledger decay deletes exactly the card the user wants to cite six weeks later.

**Fix.**
- **P0.5 — two weeks:** PubMed abstracts for 3 species, one frontier model, citation gate = *PMID resolves AND verbatim quote is an exact substring of the abstract*, cards saved to a JSON/SQLite ledger. **Use it every day for 14 days** before writing another spec.
- **Rule: no new spec until the previous phase has been *used*, not just built.**
- Add **`desk_check`** (a next step doable at a laptop today: Ensembl ortholog lookup, an AnAge record, a named paper, an existing dataset) and **`next_action`** (export to a citation-complete draft paragraph; a shareable, versioned hypothesis page) to the card. Make "**cards that left Cora**" the primary metric.
- Track **three user metrics from day one**: days-opened/week, cards exported or expanded, a weekly one-liner "did Cora tell you something you used?" Three consecutive "no"s halt infrastructure work.
- **Decay ranks, never deletes**; every archive is searchable.
- Autonomy as a **bounded slice**: "Cora proposes 3 new questions per week; you pick one." Build the scorer only if the user consistently prefers Cora's question to their own. No hot-find ping until a strong-convergence event has occurred organically at least once.
- **Tight card v2:** *pattern* (not "claim") · strongest verbatim quote with clickable PMID · one-line strongest objection · "novel vs [nearest existing paper]" · why-it's-here · evidence band. Numbers live in the dossier.

### F7 — Gene-level ER/orthology is structurally empty for the species that justify the project (2/4)
**Measured (engineering):** Ensembl REST (main + metazoa) covers **NMR, killifish, human, hydra only** — bowhead, Greenland shark, quahog, rockfish, *Turritopsis* (5/9) have no Compara orthology. GenAge/DrugAge contain **zero genes** for any non-human panel species. PubTator3 on the flagship Keane 2015 abstract returns **zero gene annotations**; the abstract never names ERCC1 or PCNA — the gold "human lever" lives in full text. Kolora (*Science*) and Tian (*Nature*) sit in PMC as 12-month-embargoed author manuscripts — not in the OA subset, no TDM license. Error math: a fully-typed edge ≈ 0.95 × 0.85 × 0.75 × 0.8 ≈ **0.48 precision**; a 3-lineage claim is correct with probability p³; above ~25 % per-edge error the top-10 by convergence score is mostly extraction artifacts.

**Fix.**
- **PubTator3 bulk annotations** (free, NCBI Gene/Taxonomy-normalized) as the NER/ER layer instead of a cheap LLM.
- **Pathway/hallmark-level convergence is the primary product.** Gene-level edges only for the 4 Ensembl-covered species plus HAGR's Bowhead Whale and Naked Mole-Rat Genome Resources; label uncovered species "pathway-level only."
- Store **orthology type** (1:1, 1:many, many:many) and taxonomic level; tree-based orthology (OrthoFinder/eggNOG) for invertebrates; **CNV/duplication as its own Finding type** (lever = dosage); BUSCO completeness + contamination screen gate every presence/absence edge; "orthology unresolved" is a first-class value that zeroes `human_lever_factor`.
- **Hand-label 200 edges** from the 816 abstracts (~1 day); publish per-edge precision on every card; **refuse to show a convergence score until per-edge FPR < 2 %.**
- Filter PMC-OA by per-article license *before* choosing gold papers.

### F8 — Panel and phenotype confounds (science)
Greenland shark (1–4 °C, ~1 cm/yr), quahog (metabolic depression, anoxia-tolerant), bowhead (Arctic, 60–100 t), deep *Sebastes* — the panel is a "cold, big, slow" sample. Predicted "convergent mechanisms" — cold-shock proteins (the real bowhead DSB finding is CIRBP, *cold-inducible*), protein thermal stability, low mitochondrial ROS — are passengers of temperature and mass. Humans (LQ ~4–5) are already extreme for their mass, so "human as anchor" inverts the comparison. Kolora 2021 worked *because* it used 88 closely related species with phylogenetic control — the opposite of 8 maximally distant species with no short-lived relatives. The phenotype node also lumps categorically different things (hydra's strain-dependent constant mortality; *Turritopsis* transdifferentiation ≠ absence of senescence; NMR non-aging mortality yet cancer occurs). Common-name queries pull *Mercenaria* (a different, short-lived clam), *Fundulus*, and Chesapeake striped bass ("rockfish") into the wrong nodes.

**Fix.** Phenotype = **residual longevity** (longevity quotient / PGLS residual on log mass + habitat temperature + class); separate nodes for senescence rate (Gompertz slope), regenerative capacity, cancer resistance; Environment attributes on Species; a **confound critic** ("does this mechanism track temperature/mass better than residual longevity?"); within-lineage **short-lived contrasts** (short-lived *Sebastes*, *T. rubra*, *H. oligactis*, *Mercenaria*, mouse) and a **non-cold high-LQ control** (Brandt's bat LQ ~10, birds); ingest AnAge's quality flag, dating method, n, captive/wild as `has_lifespan` attributes; query by **taxon ID / MeSH / binomial, never common names.**

### F9 — Prompt injection via ingested literature (trust)
Zero mentions of injection or untrusted input across six docs, yet ingestion is autonomous and includes bioRxiv (anyone can post). Attacker text in a preprint reaches four LLM prompts unlabeled — relation extraction, **the entailment verifier (the attacker-controlled passage *is* the evidence and the judge's boolean gates everything)**, the critics, and consolidation. A poisoned edge plus fabricated lineages clears the ping bar; or, inverted, deliberately spike the fabrication rate to trip the breaker — a remote DoS with a guaranteed-delivery channel, since breaker halts are exempt from the ping budget.

**Fix.** Corpus text is **data, never instructions**: pass passages as fields to constrained/JSON-schema calls, never concatenated into instruction blocks; ingest-time **injection classifier** quarantines documents with imperative meta-text; **provenance trust tiers** — preprints and auto-ingested docs cannot alone raise an edge above provisional, contribute to a ping, or be sole support for a promoted edge; verify against a stored, hashed passage snapshot with the mechanical substring check *before* the model is consulted; rate-limit and human-confirm every outbound channel; spend cap enforced *outside* the loop.

### F10 — Ungated briefing prose; Reflect launders (trust)
The citation gate covers *card claims* and *edges*. The briefing also contains "what changed … and why" (ranked first), per-card "why it's here," contradictions prose, frontier rationale, and Reflect insights — none gated. "Reflect" is defined as *generating* insights across cycles, which by definition asserts what no source states; its derivation IDs point at *hypotheses*, the summary drops their modality, and P4 §2 places those summaries **above** the evidence in every later context window — a closed positive-feedback loop.

**Fix.** The **briefing artifact is the unit of verification**: every sentence is (a) a verified claim ID, (b) a computed-value ID (metric + query hash), or (c) visibly-styled "unverified narrative" excluded from all trust metrics. Template "what changed" from a **structured diff** over IDs — no free prose. Split memory into **`derived_facts`** (lossless restatement of ≥1 verified claim, re-gated by re-running the citation check on the summary sentence) and **`conjectures`** (Reflect output: never a briefing statement, never above evidence in context, capped and decaying influence on prioritization). Every summary inherits the **modality of its weakest input**. Forbid reflections citing reflections. Replace the sampled audit with 100 % mechanical trace-checking plus a stratified human *support* sample.

### F11–F15 — Corrections and specifics
- **F11 (science):** pathway convergence must beat a **gene-set-size-matched null** and a cross-lineage permutation; do **ancestral-state reconstruction** of residual longevity to count independent *gains* (divergence time cannot); record **direction of change** per Finding; score "**ancestral retention**" (fish/cnidarian somatic telomerase retained; mammals derived) as a distinct hypothesis class with its own lever logic.
- **F12 (science):** `human_lever` requires `{direction, lever type (variant | expression | dosage | systemic), transferability, pleiotropy-safety vs OncoKB/COSMIC/OMIM + essentiality, human-genetics replication}`; rank by the **weakest link** in the species→human chain; penalize gain-of-function on oncogenes and loss-of-function on tumor suppressors. Only APOE and FOXO3 replicate robustly among human longevity loci.
- **F13 (science + product):** a **model-selection matrix** (mechanism class × donor lineage × timescale × temperature × tissue) over killifish, zebrafish, mouse, *C. elegans*, *Nematostella*/*Hydra*, *Crassostrea*, cross-species fibroblast panels, human iPSC cells; "no adequate in-vivo model — propose in-vitro assay" is valid; every falsifier carries a power estimate and a positive control. Falsifiability = **binary critic gate**, removed from the score and the tight card. Hallmarks become **optional multi-label with "unassigned"**; GO/Reactome is the primary mechanism ontology; add a panel-specific vocabulary (metabolic depression, anoxia tolerance, transdifferentiation, diapause).
- **F14 (trust):** a **sixth critic — dual-use/safety** — with a written taxonomy (refuse / redact-with-rationale / escalate) and its own eval set, run before anything surfaces or pings. Constrain `falsifier` **by construction**: whitelisted model organisms, phenotypic readouts only, no synthesis routes, sequences, doses, or vector construction; hard-block for toxin/venom/pathogen/immune-evasion mechanisms → "escalate to biosafety review." Annotate `human_lever` and any DrugAge compound "**animal data only — no human application**" *in the field*, not a footer. Propagate `license_tag` through consolidation, summaries, scores, and briefings; flag KEGG/TimeTree/OrthoDB licensing before ingest.
- **F15 (engineering):** SQLite `cycles`/`steps` tables written **before** each API call, resume from last incomplete step; content-hash cache of (prompt, model, params) → response so retries are idempotent; ledger insert keyed by a **structured dedupe tuple** (mechanism_id@level, species set, polarity, intervention) — embeddings only *propose* duplicates, merges are links not deletes, opposite polarity is never auto-merged; budget counter persisted per step from API `usage`; **Message Batches** as the overnight primitive (50 % cost, 24 h window, no 429 storms); external dead-man's switch for the briefing job. **Numbers in the spec:** k = 5, one cycle per run, Opus-tier for generate only, Sonnet-tier low-effort for critics/verify, Batches for everything scheduled, **$5/run hard cap**; measured one-time graph build ≈ **$100–200**. Discard the skeleton except `DISCLAIMER`/`REFUSAL` and the route shape (it never ran: requirements list Flask not FastAPI; `public/` deleted; Chroma list-metadata raises; ephemeral client; embedder unused; the evidence block never included the abstract text — "grounded" was never true; `index.html` has no script; the Groq model is decommissioned). Fresh `pyproject`; SQLite for docs/edges/ledger/cycles; Chroma `PersistentClient` or sqlite-vec with a **biomedical embedder**; hybrid BM25 + vector; a **$5–7 VM with a disk and cron**, not Render free tier.

---

## 4. What's actually strong (all four, reconciled)

Eval before autonomy. Cited-or-killed at edge-write time. Phylogenetic non-independence named at all. Ortholog vs pathway as different claims. `contradicts` preserved instead of overwritten. Phenotype evidence quality as a first-class node. Curiosity bounded to sourced graph structure. A self-tripping breaker whose *spiking* survival rate is as alarming as a collapsing one. "A missing briefing is itself an alert." Immutable originals with re-derived summaries. The frontier preview. The card as a stable contract across phases. P2's willingness to kill the graph if it doesn't beat the baseline. And — per engineering — the corpus's smallness: cheap to rebuild, finite to sweep.

*The gaps are almost entirely gaps of enforcement: good rules stated as rules, with no mechanism, no adversary model, and no independent measurement behind them.*

---

## 5. Revised roadmap (v0.3)

```
P0.5  two weeks · USE IT DAILY            ─┐
P1    trustworthy card + graph-lite        │  gates are evals with closed-book
P2    convergence done honestly            │  controls, external labels, and
P3    scheduled exhaustive sweep (weekly)  │  a human who actually opened it
P4    consolidation (deferred: >200 entries)│
P5    curiosity scorer (only when frontier > budget)
```

| Phase | What it now is | Gate |
|---|---|---|
| **P0.5** | 3 species · one model · mechanical citation gate (PMID resolves + exact-substring quote) · JSON/SQLite ledger · `desk_check` + `next_action` · the three user metrics. **Before any generation:** the verifier gold set (150–300 pairs) and the closed-book baseline. | Used every day for 14 days; ≥1 card left Cora |
| **P1** | Full panel (816 abstracts + HAGR; no human mining) · PubTator3 ER · **graph-lite** (SQLite edges + NetworkX, Neo4j-shaped) with the `Finding` node · ordinal evidence bands · citation lineage · cross-family verifier + structured entailment · canaries + control-limit breaker · structured dedupe · orchestration tables + Batches + $5/run cap · biomedical embedder + BM25 · taxon-ID queries · injection defenses + trust tiers · sixth (safety) critic · license propagation. | Fixed eval passes: closed-book delta, post-cutoff gold, negative set, synthetic held-out; verifier FNR published; per-edge FPR < 2 % before any convergence score is shown |
| **P2** | Residual longevity + covariates + confound critic + short-lived contrasts · pathway/hallmark-level primary with size-matched null + permutation FDR · ancestral-state independence + direction · `human_lever` with direction/safety · model-selection matrix · hallmarks multi-label. | Graph-lite convergence beats the P1 baseline on the fixed eval |
| **P3** | **Exhaustive prioritized sweep** of the species × mechanism grid, **weekly** · re-check ledger vs new ingest · explicit steering rules · bounded autonomy ("3 questions/week, you pick one") · briefing as unit of verification · breaker on system signals + canaries. | Sweep complete; weekly briefing used 4 weeks running; sycophancy index flat |
| **P4** | Consolidation with `derived_facts` / `conjectures` split — **only past 200 ledger entries** · hot-find ping only after a strong-convergence event has occurred organically. | Unattended month; provenance audit 100 % mechanical + stratified human support sample |
| **P5** | Curiosity scorer as a **realized-gain bandit** over candidate types, tractability as a hard gate — only when the frontier (all AnAge species above a threshold) exceeds the budget. | Cora's proposed question beats the user's own, consistently |

**Standing rule:** *no new spec until the previous phase has been used.*

---

*Living document. The specs carry an "Amended by red-team" banner and in-place corrections for the errors in §1; the full fixes live here.*
