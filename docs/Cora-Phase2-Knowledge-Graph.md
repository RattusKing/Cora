# Cora — Phase 2 Build Spec: The Knowledge Graph & Convergence Engine (v0.1, design only)

**Date:** 2026-06-27
**Status:** Design target — *no code yet.*
**Parent:** `docs/Cora-Architecture.md` (Phase 2 of §12). Assumes Phase 1's trustworthy card is in place.

> **⚠️ Amended by red-team (2026-09-08).** Corrections applied in place and marked **[RT]**; full findings in `docs/Cora-Red-Team.md`. Key changes: the schema as first written **cannot express a species-specific longevity signal** (a reified `Finding` node fixes it); "pathway-level is stronger" was wrong without a size-matched null; Neo4j is replaced by graph-lite; gene-level orthology is unavailable for 5 of 9 panel species, so **pathway/hallmark-level convergence is the primary product**; the phenotype needs body-mass/temperature covariates. The red-team report wins on conflict.

---

## Goal (one sentence)

> Turn **convergence** from a shallow, retrieval-bounded LLM guess (Phase 1) into a real, queryable, **phylogenetically honest** graph computation — the asset competitors can't copy.

Phase 1 could say *"this mechanism shows up in 3 species I happened to retrieve."* Phase 2 can say *"this mechanism is independently associated with extreme lifespan across 3 evolutionarily distant lineages, a human ortholog exists, and here is every sourced edge behind that claim."* That difference is the moat.

---

## 1. The ontology — grounded in existing IDs (don't invent identifiers)

Every node resolves to a canonical external ID so entity resolution is tractable and the graph is interoperable.

| Node | Canonical ID | Notes |
|---|---|---|
| Species | **NCBI Taxonomy** id | the panel + any species an edge touches |
| Gene | **NCBI Gene / Ensembl** id | species-specific |
| Ortholog group | **OrthoDB / Ensembl Compara** id | links genes-across-species (see §2) |
| Protein | **UniProt** id | |
| Pathway | **Reactome / KEGG** id | |
| GO term | **Gene Ontology** id | molecular function / process |
| Hallmark | controlled vocab (the 12) | fixed node set |
| Phenotype | internal + ontology where one exists | lifespan (years), negligible senescence, cancer resistance… |
| Intervention/Compound | **DrugAge / ChEBI / PubChem** id | |
| Model organism | NCBI Taxonomy id | killifish, mouse… |
| Claim/Evidence | internal id | every edge points back to ≥1 of these |

**Edges** (each carries `source_doc[]`, `confidence 0..1`, `license_tag`, `extractor`, `created_at`):
`has_lifespan · expresses · member_of_ortholog_group · ortholog_of · associated_with_hallmark · in_pathway · extends_lifespan_in · regulates · has_phenotype · evidence_supports · contradicts`
**[RT] + `longevity_signal_in (Gene → Species, kind)` · `tested_negative` · `not_tested`**

> **[RT] Schema bug — fixed by a reified `Finding` node.** As first written, the flagship query joins `expresses` + `associated_with_hallmark` + `has_phenotype`. But *every* panel species expresses ERCC1/PCNA/FOXO3 orthologs, and Reactome/GO map gene→hallmark identically for every species — so "genomic instability" reaches "≥3 independent lineages" for all eight species on day one, the hot-find ping fires the first night, and "orphan human lever" candidates cannot exist. The graph could not say *"gene G shows a longevity signal **in species S**."*
>
> **Fix:** convergence is computed **over `Finding` nodes only**:
> `Finding { species, entity@level (gene | ortholog_group | pathway), phenotype, direction (gain | loss | up | down | expansion | retained), evidence_tier (dN/dS or CNV < expression < in-vitro function < in-vivo genetic < intervention < human genetics), effect_size, n, method, primary_source, lab, abstract_only }`.
> A lineage counts toward convergence only if its Finding meets a minimum evidence tier; add a **direction-consistency** term (same direction in ≥3 lineages, else "contested convergence"). CNV/duplication is its own Finding type (lever = *dosage*, which `ortholog_of` collapses to a boolean). Orthology stores **type** (1:1 / 1:many / many:many) and taxonomic level; "orthology unresolved" is a first-class value that zeroes `human_lever_factor`. Gene-level edges are permitted only for the Ensembl-covered species (NMR, killifish, human, hydra) plus HAGR's Bowhead and NMR genome resources — the other five are **"pathway-level only."**

---

## 2. Two kinds of convergence (the key conceptual distinction)

The card's "convergence" field is really **two different, separately-scored claims**:

1. **Ortholog convergence** — *the same gene* (one ortholog group) is independently associated with longevity in multiple lineages.
   *e.g. ERCC1 orthologs flagged in whale and in rockfish.*
2. **Pathway / hallmark convergence** — *different genes*, but they hit the *same pathway or hallmark*, across lineages.
   *e.g. whale via ERCC1, quahog via a different repair gene — both → "genomic instability" hallmark.*

**[RT] Corrected — pathway-level convergence is the *easiest* signal to get by chance,** not the strongest by default. Reactome "DNA repair" has ~300 genes and "Immune System" ~2,000, so essentially every comparative-genomics gene list (positively selected or expanded genes) hits one — which is why every long-lived-species genome paper "finds DNA repair." It becomes strong evidence that the *mechanism* matters **only after it beats a gene-set-size-matched null and a cross-lineage permutation** (§4; red-team F11). Cora must distinguish and score both levels, each against its own null; conflating them — or privileging pathway-level without a null — is a credibility bug.

---

## 3. The phylogenetic-independence problem (the rigor that makes it credible)

**Raw species count is the wrong metric.** Convergent evolution means a trait arose *independently*. Two closely-related rockfish sharing a longevity gene is **shared ancestry, not convergence** — counting them as "2 lineages" inflates the signal. This is the classic phylogenetic non-independence problem (Felsenstein 1985).

**So the graph needs a species tree.** Import divergence times (e.g. **TimeTree**) as a backbone. Then convergence is weighted by *how independent* the supporting lineages are:

- whale + rockfish + quahog (mammal / fish / mollusc — ~hundreds of My apart) → **strong, genuinely independent**
- bowhead + another whale → **weak, likely inherited**

Without this, every convergence score is systematically overstated. With it, Cora says something a naive literature-miner can't.

> **[RT] Three scientific corrections to §3–§4.**
> **(a) The phenotype is confounded.** Greenland shark (1–4 °C, ~1 cm/yr growth), quahog (metabolic depression, anoxia-tolerant), bowhead (Arctic, 60–100 t), deep *Sebastes* — the panel is a **"cold, big, slow"** sample, so raw AnAge maximum lifespan converges on *correlates of a slow life history* (cold-shock proteins — the real bowhead DSB-repair finding is CIRBP, literally *cold-inducible*; protein thermal stability; low mitochondrial ROS), not causes of longevity. Humans (LQ ~4–5) are already extreme for their mass, so "human as anchor" inverts the comparison. Kolora 2021 worked *because* it used 88 closely related species with phylogenetic control — the opposite of eight maximally distant species with no short-lived relatives. **Fix:** phenotype = **residual longevity** (longevity quotient / PGLS residual on log body mass + habitat temperature + class); separate nodes for senescence rate (Gompertz slope), regenerative capacity, and cancer resistance (hydra's constant mortality, *Turritopsis* transdifferentiation and NMR non-aging mortality are categorically different things); Environment attributes on Species; a **confound critic** ("does this mechanism track temperature/mass better than residual longevity?"); within-lineage **short-lived contrasts** (short-lived *Sebastes*, *T. rubra*, *H. oligactis*, *Mercenaria*, mouse) and a **non-cold high-LQ control** (Brandt's bat LQ ~10, birds).
> **(b) Divergence time does not establish trait independence.** A TimeTree weight *saturates* — every cross-phylum pair (>500 My) gets full weight, so for this panel it only separates *Sebastes* from *Sebastes*. Independence is a property of the *trait*: what matters is whether extreme longevity **arose on that branch** — **ancestral-state reconstruction** of residual longevity on the species tree, counting independent *gains*. And **direction is never modeled**: somatic telomerase is active in fish and cnidarians and was *repressed* in mammals, so much panel "convergence" may be **retention of an ancestral state** with humans as the derived, short-lived oddity — a different hypothesis with a different human lever. Score "ancestral retention" as its own class.
> **(c) Publication and assay bias game the score.** NMR has thousands of aging papers, the Greenland shark a few dozen; quahog edges are almost entirely oxidative stress (what a physiology lab can measure on a clam); "pathway convergence" therefore recovers the map of which assays each lab could run. **Fix:** count **primary studies and independent labs**, never review mentions (reviews become pointers with a provenance chain — "N documents / **M independent observations**"); add `tested_negative` / `not_tested` and compute convergence as positives-over-tests with Bayesian shrinkage; normalize by per-species corpus size; a **permutation null** (shuffle mechanism labels preserving per-species edge counts) reported as **empirical FDR on every card**; a **subtractive contradiction term** in `convergence_score` (5-for/4-against must not score like 5-for/0); ordinal evidence bands, not 0–1 decimals; ingest retractions/errata with forced re-verification.

---

## 4. Convergence scoring (formal sketch)

A weighted score, not a count:

```
convergence_score(mechanism) =
     Σ_lineages  ( evidence_strength_in_lineage  ×  phylo_independence_weight )
   × level_factor            // pathway-level > single-gene (configurable)
   × human_lever_factor      // human ortholog exists? druggable?

where phylo_independence_weight downweights lineages that are
close on the species tree (so N related species ≠ N independent data points).
```

Surfaced on the card as `convergence_depth` (now = **effective independent lineages**, not raw species count) plus the breakdown. Every term traces to sourced edges.

---

## 5. Query patterns (conceptual / illustrative — not committed code)

The flagship query Phase 1 *cannot* answer:

```cypher
// "Mechanisms independently linked to extreme lifespan across distant lineages,
//  with a druggable human ortholog" — illustrative
MATCH (h:Hallmark)<-[:ASSOCIATED_WITH_HALLMARK]-(g:Gene)<-[:EXPRESSES]-(s:Species)
MATCH (s)-[:HAS_PHENOTYPE]->(:Phenotype {kind:'extreme_lifespan'})
WITH h, collect(DISTINCT s) AS species, collect(DISTINCT g) AS genes
WHERE effective_independent_lineages(species) >= 3        // §3 phylo weighting
MATCH (g2)-[:ORTHOLOG_OF]->(:Gene {species:'human', druggable:true})
RETURN h, genes, species, convergence_score(h, species) AS score
ORDER BY score DESC
```

Variations the same graph unlocks:
- **Hallmark-driven:** "everything pointing at *loss of proteostasis* across the panel."
- **Target-driven:** "given human gene X, which long-lived species provide convergent support?"
- **Outlier-driven (feeds Phase 3 curiosity):** "species with extreme lifespan but **no** mechanism edges yet" = a gap worth investigating.

---

## 6. Building the graph (the extraction pipeline — the hard part)

```
corpus (P1) ─► (a) NER: tag gene / species / phenotype / pathway mentions
            ─► (b) RELATION EXTRACTION: LLM proposes typed edges, each with a cited quote
            ─► (c) ENTITY RESOLUTION: map mentions → canonical IDs (NCBI Gene/Taxonomy, UniProt)
            ─► (d) ORTHOLOGY: attach genes to ortholog groups (Ensembl Compara / OrthoDB)
            ─► (e) EDGE WRITE: provenance + confidence; SAME citation-verify gate as P1
            ─► (f) MERGE/DEDUP: collapse duplicate edges, accumulate independent support
            ─► (g) CONTRADICTIONS: conflicting claims become `contradicts` edges, not overwrites
```

**Non-negotiable:** no unsourced edges. The graph is **evidence-weighted hypotheses, not asserted truth** — every edge is the same cited-or-killed discipline as the Phase 1 card, applied at write time. A graph full of unverified LLM-extracted edges would be Galactica with extra steps.

---

## 7. Graph + vector, working together

- **Graph** finds *candidate convergences* (structure, scoring, phylo-weighting).
- **Vector store** (from P1) pulls the *supporting passages* for each edge → becomes the card's `evidence` + citations.
- The card's `convergence` / `convergence_depth` fields are now **graph-derived**, not model-guessed. Same card schema as P1 — we just upgraded where two fields come from.

---

## 8. Trust & calibration in the graph

- Every edge: `source_doc[] + confidence + license_tag`.
- **Phenotype evidence quality is first-class** — a node like *"negligible senescence"* carries how well-supported it is, so the critic can flag the popular-science overstatements (lobster/jellyfish "immortality") instead of treating them as fact. This is the §"hype-calibration" problem rendered in the data model.
- `contradicts` edges preserve disagreement rather than silently picking a winner.

---

## 9. Honest limitations & risks

- **Relation extraction is error-prone** → mitigated by cited-edge verification + confidence, never asserted as truth.
- **Entity resolution is genuinely hard** (gene synonyms, cross-species name clashes) → grounded to canonical IDs + ortholog DBs, but expect a long tail of misses.
- **Annotation sparsity for exotic species** (quahog, *Turritopsis*) → orthology mapping will be incomplete; the graph should *show* "unknown," not fabricate. (Also a feature: gaps feed Phase 3 curiosity.)
- **Over-building risk** → scope the graph to the species panel first; resist a general bio-KG.

---

## 10. Stack (Phase 2 additions)

| Concern | Choice |
|---|---|
| Graph DB | ~~Neo4j~~ **[RT] SQLite `edges(src, rel, dst, source_doc, quote, confidence, license, extractor, created_at)` + NetworkX/pandas**, Neo4j-shaped schema; pulled *into P1*. Migrate past ~10⁶ edges. |
| Entity resolution | **[RT] PubTator3 bulk** (NCBI Gene/Taxonomy-normalized); tree-based orthology (OrthoFinder/eggNOG) for invertebrates; BUSCO + contamination screen gates every presence/absence edge |
| Phenotype covariates | **[RT] TimeTree + body mass + habitat temperature + class** (for residual longevity, §3) |
| Orthology | Ensembl Compara / OrthoDB (+ OrthoFinder if needed) |
| Species tree / divergence times | TimeTree import |
| Extraction | frontier model for relation extraction; cheap model for NER/triage |
| Carried from P1 | corpus, vector store, citation-verify gate, card schema |

---

## 11. Definition of done (Phase 2)

1. Graph populated for the species panel — every edge sourced + confidence + license.
2. Species tree imported; **phylogenetically-weighted** convergence score computable.
3. Flagship convergence query returns cited, ranked results.
4. Card `convergence` / `convergence_depth` now **graph-derived** (ortholog *and* pathway level, distinguished).
5. **Eval delta:** re-run retrospective rediscovery — does graph convergence improve recall/precision vs the P1 pre-graph baseline? (If not, the graph isn't earning its complexity — find out *here*.)

Only when convergence is real and measured do we earn Phase 3 (the self-propelling loop), whose curiosity engine literally reads this graph's gaps to decide where to look next.

---

*Living document. No code committed — the target we build against.*
