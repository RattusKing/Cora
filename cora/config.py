"""Configuration: species registry, paths, steering rules.

Species are queried by binomial (or genus) name only, with a common name added only where
it is unambiguous. Common names are otherwise deliberately avoided: "quahog" pulls
*Mercenaria mercenaria* (a different, short-lived clam), "killifish" pulls *Fundulus*, and
"rockfish" pulls Chesapeake striped bass.

Which species are in play is a property of the *topic*, not of this file - see
cora/topics.py. This module is the registry of species Cora knows how to query.
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
    class_: str = ""  # taxonomic class - the default lineage proxy
    phylum: str = ""  # taxonomic phylum - the coarse proxy for panels spanning phyla


TA = "[Title/Abstract]"


def _or(*names: str) -> str:
    """A Title/Abstract OR-group. Multi-word and hyphenated names become quoted phrases.

    Single bare words are risky: PubMed maps `bats` to BAT (brown adipose tissue) among
    other things, so groups of ambiguous common names are built from genera instead.
    """
    return "(" + " OR ".join(f'"{n}"{TA}' if (" " in n or "-" in n) else f"{n}{TA}" for n in names) + ")"


_BAT_GENERA = ("Myotis", "Rhinolophus", "Pteropus", "Desmodus", "Eptesicus", "Nyctalus",
               "Miniopterus", "Carollia", "Artibeus", "Tadarida", "Hipposideros", "Rousettus")

SPECIES: dict[str, Species] = {
    # --- aging panel (the original, verified against real abstracts) ---------------
    "naked_mole_rat": Species(
        key="naked_mole_rat",
        name="naked mole-rat",
        binomial="Heterocephalus glaber",
        taxid=10181,
        query=_or("Heterocephalus glaber", "naked mole-rat", "naked mole rat", "naked mole-rats"),
        class_="Mammalia", phylum="Chordata",
    ),
    "ocean_quahog": Species(
        key="ocean_quahog",
        name="ocean quahog",
        binomial="Arctica islandica",
        query=_or("Arctica islandica"),
        class_="Bivalvia", phylum="Mollusca",
    ),
    "rockfish": Species(
        key="rockfish",
        name="rockfish (Sebastes)",
        binomial="Sebastes",
        query=_or("Sebastes"),
        class_="Actinopterygii", phylum="Chordata",
    ),
    "bowhead_whale": Species(
        key="bowhead_whale",
        name="bowhead whale",
        binomial="Balaena mysticetus",
        query=_or("Balaena mysticetus", "bowhead whale"),
        class_="Mammalia", phylum="Chordata",
    ),
    "greenland_shark": Species(
        key="greenland_shark",
        name="Greenland shark",
        binomial="Somniosus microcephalus",
        query=_or("Somniosus microcephalus", "Greenland shark"),
        class_="Chondrichthyes", phylum="Chordata",
    ),
    "hydra": Species(
        key="hydra",
        name="hydra",
        binomial="Hydra",
        query=_or("Hydra vulgaris", "Hydra oligactis", "Hydra magnipapillata"),
        class_="Hydrozoa", phylum="Cnidaria",
    ),
    "turritopsis": Species(
        key="turritopsis",
        name="immortal jellyfish",
        binomial="Turritopsis dohrnii",
        query=_or("Turritopsis dohrnii", "Turritopsis nutricula"),
        class_="Hydrozoa", phylum="Cnidaria",
    ),
    "killifish": Species(
        key="killifish",
        name="African turquoise killifish",
        binomial="Nothobranchius furzeri",
        query=_or("Nothobranchius furzeri"),
        class_="Actinopterygii", phylum="Chordata",
    ),
    # --- regeneration panel -------------------------------------------------------
    "axolotl": Species(
        key="axolotl",
        name="axolotl",
        binomial="Ambystoma mexicanum",
        query=_or("Ambystoma mexicanum", "axolotl", "axolotls"),
        class_="Amphibia", phylum="Chordata",
    ),
    "newt": Species(
        key="newt",
        name="newt (Notophthalmus / Pleurodeles / Cynops)",
        binomial="Notophthalmus viridescens",
        query=_or("Notophthalmus viridescens", "Pleurodeles waltl", "Cynops pyrrhogaster"),
        class_="Amphibia", phylum="Chordata",
    ),
    "zebrafish": Species(
        key="zebrafish",
        name="zebrafish",
        binomial="Danio rerio",
        query=_or("Danio rerio", "zebrafish"),
        class_="Actinopterygii", phylum="Chordata",
    ),
    "planarian": Species(
        key="planarian",
        name="planarian flatworm",
        binomial="Schmidtea mediterranea",
        query=_or("Schmidtea mediterranea", "Dugesia japonica"),
        class_="Rhabditophora", phylum="Platyhelminthes",
    ),
    "spiny_mouse": Species(
        key="spiny_mouse",
        name="African spiny mouse",
        binomial="Acomys cahirinus",
        query=_or("Acomys", "spiny mouse", "spiny mice"),
        class_="Mammalia", phylum="Chordata",
    ),
    "acoel": Species(
        key="acoel",
        name="three-banded panther worm",
        binomial="Hofstenia miamia",
        query=_or("Hofstenia miamia"),
        class_="Acoela", phylum="Xenacoelomorpha",
    ),
    "deer": Species(
        key="deer",
        name="deer (antler regeneration)",
        binomial="Cervus elaphus",
        query=f"{_or('Cervus elaphus', 'Cervus nippon')} AND antler*{TA}",
        class_="Mammalia", phylum="Chordata",
    ),
    # --- cancer-resistance panel --------------------------------------------------
    "blind_mole_rat": Species(
        key="blind_mole_rat",
        name="blind mole-rat (Spalax)",
        binomial="Nannospalax galili",
        query=_or("Spalax", "Nannospalax", "blind mole rat", "blind mole-rat", "blind mole rats", "blind mole-rats"),
        class_="Mammalia", phylum="Chordata",
    ),
    "elephant": Species(
        key="elephant",
        name="elephant",
        binomial="Loxodonta africana",
        query=_or("Loxodonta africana", "Elephas maximus", "Loxodonta", "African elephant", "Asian elephant", "elephants"),
        class_="Mammalia", phylum="Chordata",
    ),
    "bats": Species(
        key="bats",
        name="bats (Chiroptera)",
        binomial="Myotis lucifugus",
        # MeSH plus genera: a bare "bats" maps to BAT (brown adipose tissue) and pulls noise.
        query="(Chiroptera[MeSH Terms] OR Chiroptera" + TA + " OR " + " OR ".join(g + TA for g in _BAT_GENERA) + ")",
        class_="Mammalia", phylum="Chordata",
    ),
    # --- low-oxygen panel ---------------------------------------------------------
    "painted_turtle": Species(
        key="painted_turtle",
        name="painted turtle",
        binomial="Chrysemys picta",
        query=_or("Chrysemys picta", "painted turtle", "painted turtles"),
        class_="Reptilia", phylum="Chordata",
    ),
    "red_eared_slider": Species(
        key="red_eared_slider",
        name="red-eared slider",
        binomial="Trachemys scripta",
        query=_or("Trachemys scripta", "red-eared slider", "red-eared sliders"),
        class_="Reptilia", phylum="Chordata",
    ),
    "crucian_carp": Species(
        key="crucian_carp",
        name="crucian carp",
        binomial="Carassius carassius",
        query=_or("Carassius carassius"),
        class_="Actinopterygii", phylum="Chordata",
    ),
    "weddell_seal": Species(
        key="weddell_seal",
        name="Weddell seal",
        binomial="Leptonychotes weddellii",
        query=_or("Leptonychotes weddellii", "Weddell seal", "Weddell seals"),
        class_="Mammalia", phylum="Chordata",
    ),
    "bar_headed_goose": Species(
        key="bar_headed_goose",
        name="bar-headed goose",
        binomial="Anser indicus",
        query=_or("Anser indicus", "bar-headed goose", "bar-headed geese"),
        class_="Aves", phylum="Chordata",
    ),
    "epaulette_shark": Species(
        key="epaulette_shark",
        name="epaulette shark",
        binomial="Hemiscyllium ocellatum",
        query=_or("Hemiscyllium ocellatum", "epaulette shark"),
        class_="Chondrichthyes", phylum="Chordata",
    ),
    # --- hibernation panel --------------------------------------------------------
    "ground_squirrel": Species(
        key="ground_squirrel",
        name="thirteen-lined ground squirrel",
        binomial="Ictidomys tridecemlineatus",
        query=_or("Ictidomys tridecemlineatus", "Spermophilus tridecemlineatus",
                  "thirteen-lined ground squirrel", "thirteen-lined ground squirrels",
                  "13-lined ground squirrel", "13-lined ground squirrels"),
        class_="Mammalia", phylum="Chordata",
    ),
    "arctic_ground_squirrel": Species(
        key="arctic_ground_squirrel",
        name="arctic ground squirrel",
        binomial="Urocitellus parryii",
        query=_or("Urocitellus parryii", "Spermophilus parryii", "arctic ground squirrel", "arctic ground squirrels"),
        class_="Mammalia", phylum="Chordata",
    ),
    "bear": Species(
        key="bear",
        name="bear (brown / American black)",
        binomial="Ursus arctos",
        query=_or("Ursus arctos", "Ursus americanus", "brown bear", "brown bears", "black bear", "black bears"),
        class_="Mammalia", phylum="Chordata",
    ),
    "dwarf_lemur": Species(
        key="dwarf_lemur",
        name="fat-tailed dwarf lemur",
        binomial="Cheirogaleus medius",
        query=_or("Cheirogaleus medius", "Cheirogaleus", "dwarf lemur", "dwarf lemurs"),
        class_="Mammalia", phylum="Chordata",
    ),
    "wood_frog": Species(
        key="wood_frog",
        name="wood frog",
        binomial="Lithobates sylvaticus",
        query=_or("Lithobates sylvaticus", "Rana sylvatica", "wood frog", "wood frogs"),
        class_="Amphibia", phylum="Chordata",
    ),
}

# The aging panel, kept as a named list: it is the corpus every published Cora number so
# far was measured on. Which species are actually in play comes from the enabled topics
# (cora/topics.py), not from this file.
AGING_SPECIES: list[str] = [
    "naked_mole_rat", "ocean_quahog", "rockfish", "bowhead_whale",
    "greenland_shark", "hydra", "turritopsis", "killifish",
]
DEFAULT_SPECIES: list[str] = AGING_SPECIES  # back-compat alias for callers without a topic
P05_SPECIES: list[str] = ["naked_mole_rat", "ocean_quahog", "rockfish"]

# AnAge (HAGR), CC BY 3.0.
ANAGE_URL = "https://genomics.senescence.info/species/dataset.zip"

# Coarse lineage proxy for convergence until a species tree is loaded: a taxonomic rank.
# Two mammals are ONE lineage at class rank; that is the point. A topic picks its rank
# (cora/topics.py); these maps are the lookup.
SPECIES_CLASS: dict[str, str] = {k: sp.class_ for k, sp in SPECIES.items()}
SPECIES_PHYLUM: dict[str, str] = {k: sp.phylum for k, sp in SPECIES.items()}
GRAPH_PERMUTATIONS = int(os.environ.get("CORA_GRAPH_PERMUTATIONS", "200"))

# --- paths ---------------------------------------------------------------
DATA_DIR = Path(os.environ.get("CORA_DATA_DIR", "data"))
DB_PATH = DATA_DIR / "cora.db"
SNAPSHOT_DIR = DATA_DIR / "snapshots"  # raw PubMed XML, never re-fetched for an eval
MANIFEST_PATH = DATA_DIR / "manifest.json"  # PMID list + download date (committed)
ANAGE_PATH = DATA_DIR / "anage" / "anage_data.txt"  # raw AnAge table (local; CC BY 3.0)

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
