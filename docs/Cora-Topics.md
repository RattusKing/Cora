# Cora — topic packs (v1)

*Written when topics were built, 2026-11-15. The red-team report (`docs/Cora-Red-Team.md`)
still wins on any conflict.*

## Why

Cora started as one question: why do some animals live far longer than their size predicts.
Four more questions have the same shape — an animal already solved a problem human medicine
has not — and reuse almost the whole machine: regeneration, cancer resistance, low-oxygen
tolerance, hibernation and metabolism.

What they share: a species panel, PubMed abstracts as the corpus, the mechanical citation
gate, the pattern-strength check, cards, the ledger, the re-check, and mechanism-level
convergence across lineages. What differs per question: the search terms, the species, the
mechanism vocabulary, and whether any trait table backs the panel.

So a **topic** is exactly that difference, and nothing else.

## What a topic is

```python
Topic(
    key, name, question,      # identity, and the one-line medical question
    terms,                    # PubMed fragment, ANDed with each species query
    species,                  # which panel species belong to this question
    mechanisms,               # keys into graph.MECHANISMS - the convergence vocabulary
    lineage_rank,             # "class" (default) or "phylum" for the convergence proxy
    trait_source,             # "anage" for aging; None where no trait table exists yet
    status,                   # "ready" once the lexicon has been checked on real abstracts
    narrow,                   # optional per-species extra term (zebrafish regeneration)
)
```

Built-in topics are data in `cora/topics.py`. A custom topic is the same dataclass, stored
as JSON in the database (`custom_topics`) and loaded alongside the built-ins, so a topic
somebody adds is not a second-class citizen — it goes through the same ingest, the same
gate, and the same convergence.

## Decisions, and why

**One query per (topic, species) pair, not per species.** The naked mole-rat appears in
three topics; each needs a different corpus (`longevity` vs `cancer` vs `hypoxia`). Papers
are stored once in `docs` and linked to each topic in `doc_topic`, so a paper that two
topics both find is one row and is counted once per topic. `ingest_state` records which
pairs have been fetched, so adding a topic later fetches only the new pairs.

**Everything that reads the corpus is scoped.** Retrieval, cards, findings, convergence,
counts, the AnAge panel table and the re-check all take `topic_keys` and default to the
enabled set. Without this, a hibernation abstract could be retrieved for an aging question
and the corpus counts would quietly stop meaning anything.

**A topic only sees its own mechanisms.** The lexicon scans each abstract with the
vocabulary of the topics that abstract belongs to, so a bear paper is never scanned for
blastemas. Overlap between vocabularies is deliberate and harmless: a sentence can be a
finding for several mechanisms, as it always could.

**The aging numbers had to stay reproducible.** The existing 18 mechanisms were not touched
— the four new topics add 19 new keys — and `aging` keeps `lineage_rank="class"`. A fresh
aging-only ingest after this change produced 662 abstracts and 1,717 findings with 0 dropped
by the gate, against 654 / 1,696 a month earlier: the difference is new papers, not a
changed pipeline.

**A single-lineage panel says so.** Every cancer-resistance species is a mammal, so no
mechanism can ever reach two class-level lineages there. Reporting "no mechanism reaches 2
lineages" would read as a null result about biology when it is a fact about the panel, so
`render_convergence` states the panel's ceiling instead. Mixing topics takes the *coarsest*
lineage rank, so a combined run can never claim more independence than either topic alone.

**Beta until checked on real data.** Only `aging` is `status="ready"`. The other four
lexicons are new regular expressions over abstracts nobody has read yet, and the aging
lexicon needed two real-data fixes before it was trustworthy (the "rather than" negation
bug, and the hedge-stripping canary). Marking them `beta` in `cora topics`, in the picker
and in `cora setup` is cheaper than pretending.

**`cora topics --add` checks before it stores.** A custom topic's species queries are run
against PubMed and the hit counts printed before anything is saved, because the failure mode
of a hand-written binomial is silent: a misspelling returns zero papers and the topic looks
merely empty. Scope is animal traits → human health.

## What this does not do

- No trait table for the four new topics. AnAge is lifespan only. Until a source exists per
  question (regeneration rate, tumour incidence, dive duration, torpor depth), those panels
  have no equivalent of the longevity quotient, and `cora species` shows no AnAge row.
- No phylogeny. The lineage proxy is still a taxonomic rank; TimeTree weighting remains the
  next real statistical improvement, now per topic.
- No cross-topic convergence. A mechanism that recurs in both hibernation and low-oxygen
  animals is a genuinely interesting signal and is *not* computed yet: enabling two topics
  pools their abstracts, it does not compare them. That is the obvious next step once a
  second topic has actually been used.

## Standing rule

Unchanged: **no new spec until the previous phase has been used.** Topics were built because
the first user asked for four specific questions, not to make Cora general. A sixth built-in
topic needs a user who will run it weekly.
