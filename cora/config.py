"""Configuration: species panel, PubMed query terms, paths, steering rules.

Species are queried by binomial (or genus) name only. Common names are deliberately
avoided in queries: "quahog" pulls *Mercenaria mercenaria* (a different, short-lived
clam), "killifish" pulls *Fundulus*, and "rockfish" pulls Chesapeake striped bass.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# The reasoning model used to draft cards. Only the *draft* comes from a model; the
# citation gate that decides what survives is mechanical (see cora/gate.py).
DEFAULT_MODEL = os.environ.get("CORA_MODEL", "claude-opus-5")

# The judge that checks the pattern line against the verified quotes (cora/verify.py).
# It must be a different model than the drafter so their blind spots are not shared.
JUDGE_MODEL = os.environ.get("CORA_JUDGE_MODEL", "claude-sonnet-5")
JUDGE_MODEL_EXPLICIT = "CORA_JUDGE_MODEL" in os.environ
# How many times the drafter may rewrite a pattern the judge called "stronger" (0 = never).
PATTERN_RETRIES = int(os.environ.get("CORA_PATTERN_RETRIES", "1"))

# Strong/causal words: flagged mechanically when they appear in a pattern but in none of
# its verified quotes. The judge decides; the flag is recorded and compared with its verdict.
STRONG_WORDS = [
    "causes", "cause", "caused", "causal", "causally",
    "drives", "drive", "driven",
    "proves", "prove", "proven", "proof",
    "demonstrates", "demonstrate", "demonstrated",
    "establishes", "establish", "established",
    "confirms", "confirm", "confirmed",
    "determines", "determine", "determined",
    "required for", "necessary for", "sufficient for", "responsible for",
    "leads to", "results in", "directly",
    "definitively", "conclusively", "unequivocally",
]


@dataclass(frozen=True)
class Species:
    key: str
    name: str  # display only - never used in a query
    binomial: str
    query: str  # PubMed query fragment - binomial / genus / unambiguous names only
    taxid: int | None = None


SPECIES: dict[str, Species] = {
    "naked_mole_rat": Species(
        key="naked_mole_rat",
        name="naked mole-rat",
        binomial="Heterocephalus glaber",
        taxid=10181,
        query=(
            '("Heterocephalus glaber"[Title/Abstract] OR "naked mole-rat"[Title/Abstract]'
            ' OR "naked mole rat"[Title/Abstract] OR "naked mole-rats"[Title/Abstract])'
        ),
    ),
    "ocean_quahog": Species(
        key="ocean_quahog",
        name="ocean quahog",
        binomial="Arctica islandica",
        query='"Arctica islandica"[Title/Abstract]',
    ),
    "rockfish": Species(
        key="rockfish",
        name="rockfish (Sebastes)",
        binomial="Sebastes",
        query="Sebastes[Title/Abstract]",
    ),
    # Available, off by default for P0.5 (pass --species to include):
    "bowhead_whale": Species(
        key="bowhead_whale",
        name="bowhead whale",
        binomial="Balaena mysticetus",
        query='("Balaena mysticetus"[Title/Abstract] OR "bowhead whale"[Title/Abstract])',
    ),
    "greenland_shark": Species(
        key="greenland_shark",
        name="Greenland shark",
        binomial="Somniosus microcephalus",
        query='("Somniosus microcephalus"[Title/Abstract] OR "Greenland shark"[Title/Abstract])',
    ),
    "hydra": Species(
        key="hydra",
        name="hydra",
        binomial="Hydra",
        query='("Hydra vulgaris"[Title/Abstract] OR "Hydra oligactis"[Title/Abstract] OR "Hydra magnipapillata"[Title/Abstract])',
    ),
    "turritopsis": Species(
        key="turritopsis",
        name="immortal jellyfish",
        binomial="Turritopsis dohrnii",
        query='("Turritopsis dohrnii"[Title/Abstract] OR "Turritopsis nutricula"[Title/Abstract])',
    ),
    "killifish": Species(
        key="killifish",
        name="African turquoise killifish",
        binomial="Nothobranchius furzeri",
        query='"Nothobranchius furzeri"[Title/Abstract]',
    ),
}

DEFAULT_SPECIES: list[str] = ["naked_mole_rat", "ocean_quahog", "rockfish"]

AGING_TERMS = (
    "(aging[Title/Abstract] OR ageing[Title/Abstract] OR aging[MeSH Terms]"
    " OR longevity[Title/Abstract] OR longevity[MeSH Terms]"
    " OR senescence[Title/Abstract] OR lifespan[Title/Abstract]"
    ' OR "life span"[Title/Abstract])'
)

# --- paths ---------------------------------------------------------------
DATA_DIR = Path(os.environ.get("CORA_DATA_DIR", "data"))
DB_PATH = DATA_DIR / "cora.db"
SNAPSHOT_DIR = DATA_DIR / "snapshots"  # raw PubMed XML, never re-fetched for an eval
MANIFEST_PATH = DATA_DIR / "manifest.json"  # PMID list + download date (committed)

# --- NCBI E-utilities ------------------------------------------------------
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_API_KEY = os.environ.get("NCBI_API_KEY")
NCBI_TOOL = "cora"
NCBI_EMAIL = os.environ.get("NCBI_EMAIL")  # optional; never defaulted to a real address
# 3 requests/s without a key, 10/s with one.
EUTILS_SLEEP = 0.11 if NCBI_API_KEY else 0.34

# --- citation gate ----------------------------------------------------------
MIN_QUOTE_CHARS = 25  # shorter "quotes" match trivially and prove nothing

# --- steering rules (explicit, editable; they order and annotate, never verify) ---
STEERING_RULES: list[dict] = [
    {"id": "min_verified_quotes", "min": 2, "text": "has at least 2 gate-verified quotes"},
    {"id": "min_species", "min": 1, "text": "has verified support in at least 1 panel species"},
    {"id": "sparse_species_flag", "min_docs": 10, "text": "flags species with fewer than 10 abstracts (does not hide them)"},
]

# Retrieval
RETRIEVE_K = 8
