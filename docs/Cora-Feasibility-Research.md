# Cora — Feasibility Research

**Question:** Is "Cora" — an AI research assistant that surfaces ideas connecting marine biology to human medicine (RAG over marine-genomics literature + LLM) — a real, feasible idea to build as a **startup/product**?

**Date:** 2026-06-22
**Method:** Multi-agent deep research across 5 angles (scientific premise, AI feasibility, market/competitors, data/licensing, regulatory/biosecurity), with adversarial verification of load-bearing claims. Confidence levels and caveats are flagged inline.

---

## TL;DR Verdict

**The idea is real and the premise is scientifically sound — but the version described in the repo is a demo, not a defensible startup, and the market it points at is the hardest kind to win.**

- **Scientific premise: VALIDATED.** Marine → medicine is a genuine, decades-old, productive field — **13–17 FDA/EMA-approved marine-derived drugs** and ~30 in clinical trials. ✅
- **The specific "bio-inspiration" angle (octopus/dolphin/seahorse): REAL SCIENCE, UNPROVEN THERAPIES.** These are legitimate, peer-reviewed research programs, but almost none have produced human treatments. Cora is on safe ground calling them *frontiers*; it would overreach implying *cures*. ⚠️
- **AI feasibility: FEASIBLE to *surface* candidate connections; IMPOSSIBLE to guarantee "no hallucinations ever."** That promise in the system prompt is literally unachievable and should be dropped. The honest, achievable target is *bounded, citation-verified* output. ⚠️
- **Market: a genuine whitespace, but a thin and dangerous one.** No Cora-like marine-focused AI assistant exists — but the buyers are few, academic, and low-paying, while the adjacent space is full of $100M+ incumbents and one spectacular flame-out (BenevolentAI, −90%). ❌ for "easy market," ✅ for "uncontested niche."
- **Data + regulatory: buildable but bounded.** Abstracts and genomic data are free; most full-text journals are not. A "research only" disclaimer does **not** reliably shield you from FDA device rules, and "refuse lab protocols + disclaimer" is a necessary-but-insufficient biosecurity baseline.

**Bottom line:** Don't build "an AI chatbot that knows marine biology." That's a thin GPT-wrapper with no moat. **Do** build a *narrow, evidence-grounded discovery tool over the fragmented marine-natural-product data assets* (MarinLit, CMNPD, GenBank) that no one has packaged yet — and be honest that it's a hypothesis-generation aid, not an oracle. That version is feasible. The current one is a portfolio piece.

---

## 1. Scientific Premise — Is "marine biology → human medicine" real?

**Yes, strongly — for molecule discovery. Weaker for charismatic-species "bio-inspiration."**

### The field is real and productive
- **~28,500–31,000+ marine natural products** catalogued since the first one (1950); ~1,200–1,300 new compounds added per year (RSC annual census). [high]
  Sources: [Frontiers in Marine Science 2025](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2025.1698327/full), [CMNPD / Nucleic Acids Research](https://pmc.ncbi.nlm.nih.gov/articles/PMC7779072/), [Natural Product Reports 2026](https://pubs.rsc.org/en/content/articlehtml/2026/np/d5np00080g)
- **"Blue biotechnology" is a formal OECD/EU economic sector** — EU turnover €973M (2023); global market ~$6–7B (2024–25), projected ~$12–16B by 2032–35. [high for EU; medium for global]
  Source: [EU Blue Economy Report 2025 (DG-MARE)](https://op.europa.eu/webpub/mare/eu-blue-economy-report-2025/blue-economic-sectors/blue-biotechnology.html)

### Approved drugs traceable to the sea (the proof points)
**13 approved as of the canonical Dec-2021 review; ~17 by end-2022** (counting ADC payloads), ~71% oncology. [high]
Source: ["From Life in the Sea to the Clinic," Biomedicines/MDPI 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8704254/), [marinepharmacology.org](https://www.marinepharmacology.org/approved)

| Drug | Marine origin | Use | Approved |
|---|---|---|---|
| Cytarabine (Ara-C) | Sponge *Tectitethya crypta* (inspired) | Leukemia | 1969 |
| Vidarabine (Ara-A) | Same sponge nucleosides | Antiviral | 1976 |
| Ziconotide (Prialt) | Cone snail *Conus magus* venom | Chronic pain | 2004 |
| Trabectedin (Yondelis) | Tunicate *Ecteinascidia turbinata* | Sarcoma | 2007/2015 |
| Eribulin (Halaven) | Sponge *Halichondria okadai* (synthetic analog) | Breast cancer | 2010 |
| Brentuximab vedotin (Adcetris) | Auristatin payload ← sea hare *Dolabella* | Lymphoma | 2011 |
| Lurbinectedin (Zepzelca) | Same tunicate chemotype | Small-cell lung cancer | 2020 |

**Two crucial honesty caveats** (these matter for product credibility):
1. **Most "marine drugs" are *synthetic molecules inspired by* a marine compound, not harvested from the sea.** [high]
2. **The true biosynthetic producer is frequently a symbiotic microbe/cyanobacterium, not the headline animal** (e.g., dolastatin → cyanobacteria, not the sea hare). [high]
   Source: [Dolastatin biosynthesis, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10915760/)

### The species Cora actually cites — verified, but mostly pre-therapeutic
- **Octopus A-to-I RNA editing** — real and dramatic (Rosenthal/Eisenberg, *Cell* 2023: >13,000 recoding sites on cold acclimation). The clinical analog — ADAR RNA-editing drugs (Wave's WVE-006, first in clinic 2023) — uses the **human** enzyme; calling that pipeline "octopus-inspired" is rhetorically loose. [high]
  Sources: [Cell 2023](https://www.sciencedirect.com/science/article/pii/S0092867423005238), [Wave WVE-006](https://www.biopharmadive.com/news/wave-rna-editing-aatd-first-trial-data/729981/)
- **Dolphin/diving-mammal hypoxia & ischemia tolerance** — explicitly framed by researchers as a model for human stroke/ischemia, but hypothesis-level. [high research / low therapy]
  Source: [Frontiers in Physiology 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6763568/)
- **Seahorse male-pregnancy immune tolerance** — real immunogenetics (*PNAS* 2020). **Caveat:** the literature links it to *pregnancy* tolerance, **not** organ-transplant tolerance — that extrapolation is popular-science, not a documented research thread. [high for pregnancy; flag transplant claims]
  Source: [PNAS 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7196912/)
- **Octopus arm/nerve regeneration** (scar-free) — real, peer-reviewed, no human therapy yet. [high research / low therapy]

### The translation gap (the skeptical core)
Baseline drug R&D attrition: **~1 in 20,000–30,000 screened compounds reach approval; ~90% of clinical candidates fail (~5–10% Phase-I-to-approval).** Bio-inspired leads are judged against this. The strongest cross-species near-translation is the **naked mole-rat** (high-molecular-mass hyaluronan → cancer resistance; transgenic-mouse validation, PNAS 2023) — and even that has **no human therapy** yet. [high]
Source: ["Why 90% of clinical drug development fails," PMC 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9293739/)

**Verdict on premise:** Sound. Marine pharmacology is a real engine of approved drugs. The charismatic-species bio-inspiration is genuine science but should be positioned as *inspiration / hypothesis generation*, never as proven therapeutic pipelines.

---

## 2. Technical / AI Feasibility — Can RAG + LLM surface valid connections without hallucinating?

**Yes to surfacing candidates. No to "zero hallucination." The repo's "no hallucinations ever allowed" is impossible and should be removed.**

### Cross-domain discovery from literature is a real technique
- **Literature-Based Discovery (Swanson's ABC model)** has a 30+ year track record: the Raynaud's↔fish-oil (1986) and migraine↔magnesium (1988) links, later **algorithmically replicated** (Weeber et al. 2001). [high]
  Sources: [Springer (Bruza et al.)](https://link.springer.com/chapter/10.1007/11563983_9), [JASIST 2001](https://onlinelibrary.wiley.com/doi/abs/10.1002/asi.1104)
- **BUT the field has a documented validity problem:** a 2023 *Bioinformatics* paper argues LBD is "built on sand due to a lack of appropriate evaluation method." Generating a *plausible* link is easy; proving it's *true* is the hard, under-tested part. [high]
  Source: [Bioinformatics 2023](https://academic.oup.com/bioinformatics/article/39/2/btad090/7036333)

### State of the art (2025–26) — real, but vendor-hyped
- **Google "AI co-scientist"** (Gemini 2.0 multi-agent, Feb 2025): produced wet-lab-tested drug-repurposing and anti-fibrotic hypotheses; its headline AMR result was a **re-derivation of an already-validated finding**, not a novel discovery. The viral "10 years in 2 days" framing is **not** in Google's primary blog. [medium]
  Source: [Google Research blog](https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/)
- **FutureHouse Robin / PaperQA2**: proposed + experimentally validated ripasudil for dry AMD (cell-culture level); PaperQA2 hit **85.2% precision** on LitQA2, beating human experts on precision (but only matching on accuracy). [medium-high]
  Sources: [FutureHouse Robin](https://www.futurehouse.org/research-announcements/demonstrating-end-to-end-scientific-discovery-with-robin-a-multi-agent-system), [PaperQA2 arXiv](https://arxiv.org/abs/2409.13740)
- **Biomedical LLMs**: BioGPT 78.2% on PubMedQA; Med-PaLM 2 86.5% on MedQA — strong benchmark scores, **not** generation guarantees. [high]

### "Zero hallucination" is not achievable — drop the claim
- **Citation fabrication is empirically large:** across generated literature reviews, **55% of GPT-3.5 and 18% of GPT-4 citations were entirely fabricated** (Scientific Reports 2023); a 2025 study found GPT-4o fabricated ~20% of citations. [high]
  Source: [Scientific Reports 2023](https://www.nature.com/articles/s41598-023-41032-5)
- **Galactica** (Meta's 120B science model) was **pulled after 3 days** in 2022 for confidently fabricating citations and authoritative nonsense — a direct cautionary tale for "scientific LLM." [high]
  Source: [MIT Technology Review](https://www.technologyreview.com/2022/11/18/1063487/meta-large-language-model-ai-only-survived-three-days-gpt-3-science/)
- **Formal arguments that hallucination is inevitable** for general LLMs (Xu et al. 2024; OpenAI's Kalai et al. 2025). [medium-high]
  Sources: [arXiv 2401.11817](https://arxiv.org/abs/2401.11817), [arXiv 2509.04664](https://arxiv.org/pdf/2509.04664)

### What *does* work: bounded, verifiable grounding
Grounded/citation-verified RAG + knowledge graphs measurably reduce (don't zero) errors: **Self-RAG cut clinical hallucinations to ~5.8%; multi-evidence RAG >40% reduction.** Every claim made human-checkable against a real source is the honest engineering goal. [medium]
Source: [MDPI Electronics (Self-RAG)](https://www.mdpi.com/2079-9292/14/21/4227), [MEGA-RAG, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12540348/)

**Verdict on AI feasibility:** The architecture class is sound and proven by 2025-era systems. The current repo stack (ChromaDB + all-MiniLM-L6-v2 + Llama-3-8B over a tiny hand-written seed KB) is a working demo but nowhere near this bar. The "no hallucinations" promise is a liability — replace with "every statement is grounded in a retrievable, cited source, and the system says 'I don't know' when it can't ground a claim."

---

## 3. Market & Competitors — Is there a defensible gap, and who pays?

**There's a real, uncontested *marine-specific* niche — but it's thin and academic, surrounded by deep-pocketed organism-agnostic incumbents.**

### The money is in AI drug discovery — and it's organism-agnostic
- **AI-in-drug-discovery market: ~$1.7–1.9B (2024) → ~$8.5–9.2B (2030), ~29–31% CAGR** (consistent across Grand View, Precedence, GM Insights). [medium — figures cluster but vary by definition]
  Source: [Grand View Research](https://www.grandviewresearch.com/industry-analysis/artificial-intelligence-drug-discovery-market)
- **Well-funded incumbents:** Insilico Medicine ($500M+, HK IPO Dec 2025, 10 clinical-stage programs); Isomorphic Labs ($600M raise, Novartis/Lilly deals ~$3B); Recursion (acquired Exscientia ~$700M, NVIDIA-backed); **BenchSci** ($218M; used by 16 of top-20 pharma); **Causaly** ($93M; 12 of top-20 pharma). [high]
- **Cautionary tale:** **BenevolentAI collapsed >90%** (>£1B → ~£85M), repeated layoffs, moved to delist. Narrow, unvalidated AI-discovery bets get punished. [high]
  Source: [Fierce Biotech](https://www.fiercebiotech.com/biotech/benevolentai-lays-30-staff-exits-us-site-funding-gap-looms)

### Literature/research-assistant tier (closer to Cora's form factor)
Elicit (125M+ papers, ~$10–49/mo), Consensus (~$10/mo), Scite (~$20/mo), SciSpace — all **organism-agnostic**. Enterprise money concentrates in pharma-facing tools (BenchSci, Causaly), not individual-researcher subscriptions. [high on pricing; self-reported user counts treated cautiously]

### The marine-specific gap is real — but not empty next door
- **No commercial marine-biology-focused AI research assistant / discovery SaaS exists.** Closest: **Marine Biologics + MQS "MacroLink"** (AI for marine *ingredients* — food/cosmetics/biomaterials, **not pharma**); academic prototypes (MARINA/OceanAI at NC State, OceanGPT — oceanography/conservation, not biotech). [high that no direct competitor; medium on completeness — a stealth startup could exist]
  Source: [The Quantum Insider](https://thequantuminsider.com/2025/09/19/marine-biologics-partners-with-molecular-quantum-solutions-to-accelerate-next-generation-biomaterials-development/)
- **But marine *drug discovery* itself is NOT empty:** PharmaMar (made trabectedin/lurbinectedin), Biosortia, **Sirenas Marine Discovery**, GlycoMar, SeaLife Pharma are established marine-pharma players, and an "AI + marine natural products" research wave is forming. [high]
- **The data infrastructure already exists but is fragmented:** MarinLit (RSC, subscription), Dictionary of Marine Natural Products, **CMNPD** (open, 31,000+ entities). No one has packaged these into a commercial AI assistant. [high]
  Source: [CMNPD, Nucleic Acids Research](https://academic.oup.com/nar/article/49/D1/D509/5912565)

**Verdict on market:** A defensible whitespace exists *only if narrow* — packaging fragmented marine-NP data into a grounded discovery tool. But the buyer pool (marine-NP chemists, blue-biotech startups, pharmacognosy/oceanography labs) is small and low-WTP versus the pharma R&D budgets ($4B→$25B by 2030) that fund the giants. This is a "wedge niche," not a land grab.

---

## 4. Data, Safety & Regulatory — Can you source the data and stay legal?

**Buildable, but with hard boundaries on data and real regulatory exposure.**

### Data licensing — abstracts free, full-text mostly not
- **PubMed: 38M+ abstracts**, bulk-downloadable; **E-utilities API capped at 3 req/s (10 with free key)**, abuse → IP block. [high]
  Sources: [PubMed about](https://pubmed.ncbi.nlm.nih.gov/about/), [E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25497/)
- **PMC Open Access Subset (~3.4M+ full-text)** is the only sanctioned-for-TDM tier — **but license is per-article**; only the CC0/CC-BY/CC-BY-SA tier permits commercial use (CC-BY-NC and "Other" do not). You must filter, not bulk-ingest. [high]
  Source: [PMC OA list](https://pmc.ncbi.nlm.nih.gov/tools/openftlist/)
- **Most journal full-text is paywalled/copyrighted; no general US TDM safe harbor** (fair-use for AI training is unsettled and in active litigation as of mid-2026). EU has a *commercial* TDM exception (DSM Art. 4) but rightsholders can opt out; UK's §29A is **non-commercial only**. → You cannot lawfully ingest most full-text at scale without **paid licensing deals**. [high EU/UK; medium US]
  Source: [EU DSM Directive](https://eur-lex.europa.eu/eli/dir/2019/790/oj)
- **GenBank & Ensembl: openly reusable** (no DB-level copyright), with the caveat that third-party submitters may hold IP in portions. [high]
  Sources: [NCBI policies](https://www.ncbi.nlm.nih.gov/home/about/policies/), [Ensembl legal](https://www.ensembl.org/info/about/legal/index.html)

### Regulatory — the disclaimer does NOT save you
- **FDA device status is set by *intended use* (21 CFR §801.4) — claims, design, marketing — NOT by a disclaimer.** "Research only / not medical advice" is **not a reliable shield** if function or claims indicate a medical purpose. [high]
  Sources: [21 CFR 801.4](https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-801/subpart-A/section-801.4), [FDA intended-use rule](https://www.federalregister.gov/documents/2021/08/02/2021-15980/regulations-regarding-intended-uses)
- **Non-device Clinical Decision Support requires meeting ALL FOUR criteria** (FD&C §520(o)(1)(E)); software aimed at **patients/caregivers** (not HCPs) falls outside the carve-out and is more likely a regulated device. Keep Cora research/HCP-facing and non-diagnostic. [high]
  Source: [FDA CDS guidance](https://www.fda.gov/medical-devices/software-medical-device-samd/clinical-decision-support-software-frequently-asked-questions-faqs)

### Biosecurity — "refuse protocols + disclaimer" is a floor, not a ceiling
- US binding framework weakened (Biden's **EO 14110 revoked Jan 2025**), but **expectations rose** via voluntary frontier-lab frameworks (Anthropic RSP / ASL-3, OpenAI Preparedness, DeepMind Frontier Safety) treating bio-uplift as top-tier risk. [high]
- Expert consensus (NTI|bio 2025) and red-team evidence — including a Microsoft-led *Science* (Oct 2025) study where AI-redesigned toxin proteins **evaded gene-synthesis screening at up to ~100%** — show content-level refusals are **necessary but insufficient**. [medium-high]
  Sources: [NTI|bio statement](https://www.nti.org/analysis/articles/statement-on-biosecurity-risks-at-the-convergence-of-ai-and-the-life-sciences/), [MIT Tech Review on Science paper](https://www.technologyreview.com/2025/10/02/1124767/microsoft-says-ai-can-create-zero-day-threats-in-biology/)

**Verdict on data/regulatory:** Legal sourcing is feasible but bounded — free abstracts + open genomic data + filtered OA full-text get you a real corpus; everything else needs licensing. Stay strictly research/HCP-facing and non-diagnostic to avoid FDA device territory, and treat the disclaimer as a baseline, layering capability-aware refusal and avoidance of synthesis-actionable outputs (relevant because Cora's domain includes marine toxins/venoms like conotoxins).

---

## 5. What Cora would need to differentiate and survive

### Realistic positioning
**"A grounded discovery & evidence engine for marine natural products and comparative-biology leads"** — not "a marine biology chatbot." Every answer is a set of cited, retrievable passages + a structured hypothesis, with explicit "I can't ground this" honesty. The moat is **curated, structured marine data + verification quality**, not the LLM.

### Minimum viable product (the honest version)
1. **Pick one wedge:** marine-natural-product → target/indication evidence linking (the area with actual approved-drug precedent), not all of marine biology.
2. **Real corpus:** PubMed abstracts (E-utilities) + filtered PMC-OA commercial-tier full text + CMNPD/MarinLit + GenBank — replacing the hand-written seed KB.
3. **Grounded RAG with citation verification:** retrieve → generate → verify each claim maps to a real source → refuse/flag when it doesn't. Add a knowledge-graph layer for ABC-style cross-domain bridges.
4. **Honest guardrails:** drop "no hallucinations ever"; add capability-aware refusal + non-diagnostic, HCP/research-only framing.
5. **A real evaluation harness** (since LBD's validity problem is the field's Achilles heel) — measure precision of surfaced links against held-out known associations.

### Biggest risks that could kill it
1. **No moat / GPT-wrapper risk** — the #1 killer. A generic chatbot over a small KB is trivially replicable. Defensibility must come from proprietary curated/structured marine data + verification, or it dies.
2. **Thin market** — marine-specific buyers are few and low-paying; you may have to broaden to "natural-product / comparative-biology discovery" to reach a fundable TAM, which puts you against Causaly/BenchSci.
3. **Credibility collapse** — one Galactica-style fabricated-citation incident in a scientific tool destroys trust permanently. The "no hallucinations" claim invites exactly this.
4. **Data licensing wall** — the most valuable content (full-text journals) is paywalled; scaling requires costly deals.
5. **The translation-gap critique** — sophisticated buyers know bio-inspiration rarely yields drugs; overclaiming ("ocean cures") burns credibility with the exact experts you need.
6. **Regulatory drift** — any slide toward patient-facing medical advice triggers FDA exposure a disclaimer won't cover.

---

## How the current repo compares

The repo is a **legitimate working prototype of the *concept*, but a long way from the *product*:**
- Stack (FastAPI + ChromaDB + all-MiniLM-L6-v2 + Groq/Llama-3-8B) is a reasonable demo skeleton.
- **Knowledge base is ~6 hand-written seed facts** with `example.org` placeholder links — not a real corpus; retrieval is currently theater.
- **`SYSTEM_PROMPT` promises "no hallucinations or false data is ever allowed"** — impossible; this is the single most important thing to change.
- Implementation bugs to fix before it even runs as a demo: `safety.py` references an undefined `DISALLOWED_KEYWORDS`; `rag_engine.py` imports `dolphin` but the module is `dolphins.py`; `app.py` mounts static files from `public/` while `index.html` lives at the repo root and mounting at `/` can shadow `/api/chat`.

None of these are fatal — they're the normal gap between "weekend prototype" and "product." The point is that the *hard* parts of feasibility are conceptual (corpus, grounding, moat, market), not the plumbing.

---

## Sources (selected, by confidence)

**High-confidence anchors:**
- Marine drugs review — [Biomedicines/MDPI 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8704254/); [marinepharmacology.org](https://www.marinepharmacology.org/approved)
- Octopus RNA editing — [Cell 2023](https://www.sciencedirect.com/science/article/pii/S0092867423005238)
- Diving-mammal ischemia — [Frontiers in Physiology 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6763568/)
- Seahorse pregnancy immunity — [PNAS 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7196912/)
- Drug attrition — [PMC 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9293739/)
- LBD foundations — [JASIST 2001](https://onlinelibrary.wiley.com/doi/abs/10.1002/asi.1104); critique [Bioinformatics 2023](https://academic.oup.com/bioinformatics/article/39/2/btad090/7036333)
- Citation fabrication — [Scientific Reports 2023](https://www.nature.com/articles/s41598-023-41032-5)
- Galactica — [MIT Tech Review 2022](https://www.technologyreview.com/2022/11/18/1063487/meta-large-language-model-ai-only-survived-three-days-gpt-3-science/)
- AI drug discovery market — [Grand View Research](https://www.grandviewresearch.com/industry-analysis/artificial-intelligence-drug-discovery-market)
- BenevolentAI collapse — [Fierce Biotech](https://www.fiercebiotech.com/biotech/benevolentai-lays-30-staff-exits-us-site-funding-gap-looms)
- Data licensing — [PMC OA list](https://pmc.ncbi.nlm.nih.gov/tools/openftlist/); [E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25497/)
- FDA intended use — [21 CFR 801.4](https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-801/subpart-A/section-801.4)
- Biosecurity — [NTI|bio 2025](https://www.nti.org/analysis/articles/statement-on-biosecurity-risks-at-the-convergence-of-ai-and-the-life-sciences/); [Science/Microsoft 2025 via MIT TR](https://www.technologyreview.com/2025/10/02/1124767/microsoft-says-ai-can-create-zero-day-threats-in-biology/)

**Medium/flagged:** exact current marine-drug count (13 confirmed / ~17 plly), all market-size CAGRs (firm-dependent), Google co-scientist "10-years-in-2-days" (press embellishment, not in primary source), Isomorphic "$2.1B Series B" (unverified; $600M confirmed).

*Caveat: market-size figures and 2025–26 funding/clinical milestones drift; treat absolute numbers as directional.*
