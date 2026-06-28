# Cora — Phase 2 Build Spec: The Knowledge Graph & Convergence Engine (v0.1, design only)

**Date:** 2026-06-27
**Status:** Design target — *no code yet.*
**Parent:** `docs/Cora-Architecture.md` (Phase 2 of §12). Assumes Phase 1's trustworthy card is in place.

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

---

## 2. Two kinds of convergence (the key conceptual distinction)

The card's "convergence" field is really **two different, separately-scored claims**:

1. **Ortholog convergence** — *the same gene* (one ortholog group) is independently associated with longevity in multiple lineages.
   *e.g. ERCC1 orthologs flagged in whale and in rockfish.*
2. **Pathway / hallmark convergence** — *different genes*, but they hit the *same pathway or hallmark*, across lineages.
   *e.g. whale via ERCC1, quahog via a different repair gene — both → "genomic instability" hallmark.*

**Pathway-level convergence is often the *stronger* evidence** that the *mechanism* (not a specific gene) is what matters for longevity — because evolution found the same solution by different molecular routes. Cora must distinguish and score both; conflating them is a credibility bug.

---

## 3. The phylogenetic-independence problem (the rigor that makes it credible)

**Raw species count is the wrong metric.** Convergent evolution means a trait arose *independently*. Two closely-related rockfish sharing a longevity gene is **shared ancestry, not convergence** — counting them as "2 lineages" inflates the signal. This is the classic phylogenetic non-independence problem (Felsenstein 1985).

**So the graph needs a species tree.** Import divergence times (e.g. **TimeTree**) as a backbone. Then convergence is weighted by *how independent* the supporting lineages are:

- whale + rockfish + quahog (mammal / fish / mollusc — ~hundreds of My apart) → **strong, genuinely independent**
- bowhead + another whale → **weak, likely inherited**

Without this, every convergence score is systematically overstated. With it, Cora says something a naive literature-miner can't.

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
| Graph DB | **Neo4j** (locked default) |
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
