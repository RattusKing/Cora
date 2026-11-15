"""Topic packs: the research question Cora is pointed at.

A **topic** bundles everything that is question-specific:

- `terms`: the PubMed fragment ANDed with each species query (aging terms, regeneration
  terms, ...). This is what makes a corpus about a question rather than about an animal.
- `species`: which panel species belong to this topic, and the lineage rank used for the
  convergence proxy (class for vertebrate-heavy panels, phylum where the panel spans
  cnidarians and flatworms, so two flatworms are not counted as two independent lineages).
- `mechanisms`: the controlled vocabulary convergence is computed over, as a list of keys
  into cora.graph.MECHANISMS. A topic only ever sees its own mechanisms.
- `trait_source`: which trait table, if any, backs the species panel. Only aging has one
  (AnAge max longevity); the others are honest about having none yet.
- `status`: "ready" once its lexicon has been checked against real abstracts, else "beta".

Built-in topics are data, not code: a custom topic is the same dataclass, stored in the
database and loaded alongside them. `all_topics(conn)` is the registry callers use.

Species counts in the comments are PubMed hits for (species AND topic terms) measured
2026-11-15; they are a sanity check on the queries, not a promise.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace

TA = "[Title/Abstract]"

# Lineage ranks a topic may use for the convergence proxy, coarsest last.
LINEAGE_RANKS = ("class", "phylum")


@dataclass(frozen=True)
class Topic:
    key: str
    name: str
    question: str  # one line: the medical question this topic is pointed at
    terms: str  # PubMed fragment, ANDed with the species query
    species: tuple[str, ...]
    mechanisms: tuple[str, ...]
    lineage_rank: str = "class"
    trait_source: str | None = None  # e.g. "anage" - a species trait table that backs it
    status: str = "beta"  # ready | beta
    builtin: bool = True
    note: str = ""
    # Per-species extra PubMed fragment, for a species whose literature is far larger than
    # the topic (zebrafish regeneration). Topic-specific, so it lives on the topic.
    narrow: tuple[tuple[str, str], ...] = ()

    def narrow_for(self, species_key: str) -> str | None:
        return dict(self.narrow).get(species_key)

    def query(self, species_query: str, species_key: str | None = None) -> str:
        q = f"{species_query} AND {self.terms}"
        extra = self.narrow_for(species_key) if species_key else None
        return f"{q} AND {extra}" if extra else q


# --- topic term fragments ---------------------------------------------------------

AGING_TERMS = (
    "(aging[Title/Abstract] OR ageing[Title/Abstract] OR aging[MeSH Terms]"
    " OR longevity[Title/Abstract] OR longevity[MeSH Terms]"
    " OR senescence[Title/Abstract] OR lifespan[Title/Abstract]"
    ' OR "life span"[Title/Abstract])'
)
REGENERATION_TERMS = (
    f"(regenerat*{TA} OR blastema{TA} OR \"wound healing\"{TA} OR scarless{TA}"
    f" OR \"scar-free\"{TA} OR \"tissue repair\"{TA})"
)
CANCER_TERMS = (
    f"(cancer*{TA} OR tumor*{TA} OR tumour*{TA} OR neoplas*{TA} OR malignan*{TA}"
    f" OR oncogen*{TA} OR carcinogen*{TA})"
)
HYPOXIA_TERMS = (
    f"(hypoxi*{TA} OR anoxi*{TA} OR \"low oxygen\"{TA} OR \"oxygen deprivation\"{TA}"
    f" OR ischemi*{TA} OR ischaemi*{TA} OR \"high-altitude\"{TA} OR \"breath-hold\"{TA}"
    f" OR \"oxygen store\"{TA} OR \"oxygen stores\"{TA})"
)
HIBERNATION_TERMS = (
    f"(hibernat*{TA} OR torpor{TA} OR \"freeze tolerance\"{TA} OR \"freeze-tolerant\"{TA}"
    f" OR estivation{TA} OR aestivation{TA} OR \"metabolic depression\"{TA})"
)


# --- built-in topics -----------------------------------------------------------------

BUILTIN: dict[str, Topic] = {
    "aging": Topic(
        key="aging",
        name="Aging & longevity",
        question="Which mechanisms let some animals live far longer than their body size predicts, and what do they imply for human aging?",
        terms=AGING_TERMS,
        species=(
            "naked_mole_rat", "ocean_quahog", "rockfish", "bowhead_whale",
            "greenland_shark", "hydra", "turritopsis", "killifish",
        ),
        mechanisms=(
            "dna_repair", "telomere", "proteostasis", "oxidative_stress", "mitochondria",
            "senescence", "nutrient_sensing", "stem_cells", "immune_inflammation",
            "epigenetics", "hyaluronan", "extracellular_matrix", "cancer_resistance",
            "hypoxia", "metabolic_rate", "cold_adaptation", "diapause", "apoptosis",
        ),
        lineage_rank="class",
        trait_source="anage",
        status="ready",
        note="The original panel. Lexicon checked against 662 real abstracts (Nov 2026).",
    ),
    "regeneration": Topic(
        key="regeneration",
        name="Regeneration & wound repair",
        question="How do animals regrow lost tissue without scarring, and which of those pathways exist in humans?",
        terms=REGENERATION_TERMS,
        species=("axolotl", "newt", "zebrafish", "planarian", "spiny_mouse", "hydra", "acoel", "deer"),
        mechanisms=(
            "stem_cells", "blastema", "dedifferentiation", "positional_memory", "scarring",
            "extracellular_matrix", "immune_inflammation", "wnt_signaling", "epigenetics",
            "nerve_dependence", "apoptosis", "proteostasis",
        ),
        lineage_rank="class",
        # Zebrafish regeneration alone is ~3,800 abstracts, several times the rest of the
        # panel combined; requiring the topic word in the title keeps the panel balanced.
        narrow=(("zebrafish", "regenerat*[Title]"),),
    ),
    "cancer_resistance": Topic(
        key="cancer_resistance",
        name="Cancer resistance",
        question="Why do some large or long-lived animals almost never get cancer (Peto's paradox), and which defences are human-actionable?",
        terms=CANCER_TERMS,
        species=("naked_mole_rat", "blind_mole_rat", "elephant", "bowhead_whale", "bats"),
        mechanisms=(
            "tumor_suppressor", "cancer_resistance", "contact_inhibition", "hyaluronan",
            "dna_repair", "senescence", "apoptosis", "telomere", "immune_inflammation",
            "necrosis", "extracellular_matrix",
        ),
        lineage_rank="class",  # every species here is a mammal: expect 1 lineage, by design
        note="All five species are mammals, so lineage counts stay at 1 - the convergence "
             "number is deliberately uninformative here and the report says so.",
    ),
    "hypoxia": Topic(
        key="hypoxia",
        name="Low-oxygen tolerance",
        question="How do animals survive hours to months without oxygen, and what does that suggest for stroke, heart attack and organ preservation?",
        terms=HYPOXIA_TERMS,
        species=(
            "naked_mole_rat", "painted_turtle", "red_eared_slider", "crucian_carp",
            "weddell_seal", "bar_headed_goose", "epaulette_shark",
        ),
        mechanisms=(
            "hypoxia", "metabolic_rate", "oxidative_stress", "mitochondria", "glycolysis",
            "ion_channel_arrest", "oxygen_transport", "neuroprotection", "hif_signaling",
            "apoptosis", "cold_adaptation",
        ),
        lineage_rank="class",
    ),
    "hibernation": Topic(
        key="hibernation",
        name="Hibernation & metabolism",
        question="How do hibernators drop their metabolism for months without losing muscle or bone, and what does that imply for bed rest, diabetes and organ preservation?",
        terms=HIBERNATION_TERMS,
        species=(
            "ground_squirrel", "arctic_ground_squirrel", "bear", "dwarf_lemur", "bats", "wood_frog",
        ),
        mechanisms=(
            "torpor", "metabolic_rate", "muscle_atrophy_resistance", "bone_maintenance",
            "insulin_resistance", "cold_adaptation", "freeze_tolerance", "mitochondria",
            "oxidative_stress", "proteostasis", "nutrient_sensing", "neuroprotection",
        ),
        lineage_rank="class",
    ),
}

DEFAULT_TOPIC = "aging"


# --- custom topics (stored in the database) ------------------------------------------

_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{1,31}$")


class TopicError(ValueError):
    pass


def validate_key(key: str) -> str:
    if not _KEY_RE.match(key or ""):
        raise TopicError(
            f"invalid topic key {key!r}: use 2-32 characters, lowercase letters, digits and underscores, starting with a letter"
        )
    if key in BUILTIN:
        raise TopicError(f"{key!r} is a built-in topic; pick another key")
    return key


def custom_topic(
    key: str,
    name: str,
    question: str,
    terms: str,
    species: list[str],
    mechanisms: list[str],
    lineage_rank: str = "class",
    note: str = "",
) -> Topic:
    """Build (but do not store) a custom topic, validating every field."""
    validate_key(key)
    if not name.strip() or not question.strip():
        raise TopicError("a custom topic needs a name and a one-line question")
    if not terms.strip():
        raise TopicError("a custom topic needs PubMed search terms")
    if not species:
        raise TopicError("a custom topic needs at least one species")
    if lineage_rank not in LINEAGE_RANKS:
        raise TopicError(f"lineage_rank must be one of {', '.join(LINEAGE_RANKS)}")
    return Topic(
        key=key, name=name.strip(), question=question.strip(), terms=terms.strip(),
        species=tuple(species), mechanisms=tuple(mechanisms), lineage_rank=lineage_rank,
        trait_source=None, status="beta", builtin=False, note=note.strip(),
    )


def to_json(topic: Topic) -> str:
    return json.dumps(
        {
            "key": topic.key, "name": topic.name, "question": topic.question, "terms": topic.terms,
            "species": list(topic.species), "mechanisms": list(topic.mechanisms),
            "lineage_rank": topic.lineage_rank, "note": topic.note,
        },
        indent=1,
    )


def from_json(blob: str | dict) -> Topic:
    d = json.loads(blob) if isinstance(blob, str) else dict(blob)
    try:
        return custom_topic(
            key=d["key"], name=d["name"], question=d["question"], terms=d["terms"],
            species=list(d["species"]), mechanisms=list(d.get("mechanisms") or []),
            lineage_rank=d.get("lineage_rank", "class"), note=d.get("note", ""),
        )
    except KeyError as e:
        raise TopicError(f"custom topic is missing the field {e.args[0]!r}") from None


# --- registry -------------------------------------------------------------------------

def all_topics(conn=None) -> dict[str, Topic]:
    """Built-in topics plus any custom topics stored in this database."""
    out = dict(BUILTIN)
    if conn is not None:
        from . import db

        for blob in db.custom_topic_rows(conn):
            try:
                t = from_json(blob)
            except TopicError:
                continue  # a malformed stored topic must not break the whole registry
            out[t.key] = t
    return out


def get(key: str, conn=None) -> Topic:
    topics = all_topics(conn)
    if key not in topics:
        raise TopicError(f"unknown topic {key!r}; known: {', '.join(topics)}")
    return topics[key]


def resolve(keys: list[str] | None, conn=None) -> list[str]:
    """Validate a list of topic keys, or fall back to the enabled set (then the default)."""
    topics = all_topics(conn)
    if keys:
        unknown = [k for k in keys if k not in topics]
        if unknown:
            raise TopicError(f"unknown topic{'s' if len(unknown) > 1 else ''} {', '.join(unknown)}; known: {', '.join(topics)}")
        seen: list[str] = []
        for k in keys:  # preserve order, drop duplicates
            if k not in seen:
                seen.append(k)
        return seen
    if conn is not None:
        from . import db

        enabled = [k for k in db.enabled_topics(conn) if k in topics]
        if enabled:
            return enabled
    return [DEFAULT_TOPIC]


def species_for(topic_keys: list[str], conn=None) -> list[str]:
    """Every species belonging to any of these topics, in topic order, de-duplicated."""
    topics = all_topics(conn)
    out: list[str] = []
    for k in topic_keys:
        for s in topics[k].species:
            if s not in out:
                out.append(s)
    return out


def topics_for_species(species_key: str, topic_keys: list[str] | None = None, conn=None) -> list[str]:
    topics = all_topics(conn)
    keys = topic_keys if topic_keys is not None else list(topics)
    return [k for k in keys if species_key in topics[k].species]


def pairs(topic_keys: list[str], species_keys: list[str] | None = None, conn=None) -> list[tuple[str, str]]:
    """The (topic, species) pairs to ingest: one PubMed query each."""
    topics = all_topics(conn)
    out: list[tuple[str, str]] = []
    for tk in topic_keys:
        for sk in topics[tk].species:
            if species_keys is None or sk in species_keys:
                out.append((tk, sk))
    return out


def mechanisms_for(topic_keys: list[str], conn=None) -> list[str]:
    topics = all_topics(conn)
    out: list[str] = []
    for k in topic_keys:
        for m in topics[k].mechanisms:
            if m not in out:
                out.append(m)
    return out


def lineage_rank_for(topic_keys: list[str], conn=None) -> str:
    """The coarsest rank among these topics: mixing panels must not inflate lineage counts."""
    topics = all_topics(conn)
    ranks = [topics[k].lineage_rank for k in topic_keys if k in topics]
    for rank in reversed(LINEAGE_RANKS):  # coarsest first
        if rank in ranks:
            return rank
    return "class"


def with_status(topic: Topic, status: str) -> Topic:
    return replace(topic, status=status)


def render_picker(topics: dict[str, Topic], enabled: list[str], counts: dict[str, int] | None = None) -> str:
    """The numbered list `cora setup` and `cora topics` show."""
    lines = ["What do you want Cora to research?", ""]
    for i, (key, t) in enumerate(topics.items(), start=1):
        mark = "x" if key in enabled else " "
        bits = [f"{len(t.species)} species"]
        if counts and counts.get(key):
            bits.append(f"{counts[key]} abstracts")
        if t.status != "ready":
            bits.append("beta lexicon")
        if not t.builtin:
            bits.append("custom")
        lines.append(f"  [{mark}] {i}. {t.name:<28} ({', '.join(bits)})")
        lines.append(f"         {t.question}")
    lines.append("")
    lines.append("Pick one or more by number (e.g. 1,3,4), or 'all'. Add your own with `cora topics --add`.")
    return "\n".join(lines)


def parse_selection(answer: str, topics: dict[str, Topic]) -> list[str]:
    """Turn '1,3' / 'all' / 'aging regeneration' into topic keys."""
    keys = list(topics)
    text = (answer or "").strip().lower()
    if not text:
        raise TopicError("nothing selected")
    if text in ("all", "*"):
        return keys
    out: list[str] = []
    for part in re.split(r"[,\s]+", text):
        if not part:
            continue
        if part.isdigit():
            i = int(part)
            if not 1 <= i <= len(keys):
                raise TopicError(f"{i} is not on the list (1-{len(keys)})")
            pick = keys[i - 1]
        elif part in topics:
            pick = part
        else:
            raise TopicError(f"{part!r} is not a topic number or key")
        if pick not in out:
            out.append(pick)
    if not out:
        raise TopicError("nothing selected")
    return out
