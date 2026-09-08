import pytest

from cora import db

# Synthetic corpus: 3 species x 2 abstracts. Long enough for canaries (>120 chars), some with hedges.
DOCS = [
    {
        "pmid": "1000001", "species": "naked_mole_rat", "pub_year": 2013,
        "title": "High-molecular-mass hyaluronan and cancer resistance in the naked mole-rat",
        "abstract": "Naked mole-rats (Heterocephalus glaber) show remarkable longevity relative to body size. We found that high-molecular-mass hyaluronan may contribute to their resistance to cancer, suggesting a possible role for the extracellular matrix in protection. Further work is needed to establish causation.",
    },
    {
        "pmid": "1000002", "species": "naked_mole_rat", "pub_year": 2017,
        "title": "Fructose-driven glycolysis supports anoxia resistance in the naked mole-rat",
        "abstract": "Hypoxia tolerance in the naked mole-rat is associated with altered fructose metabolism in the brain. These findings suggest that metabolic rewiring could underlie survival under low oxygen. The mechanism remains to be tested in other tissues.",
    },
    {
        "pmid": "2000001", "species": "ocean_quahog", "pub_year": 2011,
        "title": "Extreme longevity in Arctica islandica: oxidative stress resistance",
        "abstract": "Arctica islandica is among the longest-lived non-colonial animals known. Proteins from long-lived individuals appear to resist oxidative damage, and mitochondrial membranes show a low peroxidation index. These observations point to proteostasis as a candidate mechanism.",
    },
    {
        "pmid": "2000002", "species": "ocean_quahog", "pub_year": 2014,
        "title": "Sclerochronology and metabolic depression in Arctica islandica",
        "abstract": "Growth rings of Arctica islandica allow cross-dated chronologies spanning centuries. Metabolic depression during anoxia is possibly linked to extended lifespan in this bivalve. Sample sizes were small and further validation is required.",
    },
    {
        "pmid": "3000001", "species": "rockfish", "pub_year": 2021,
        "title": "Origins and evolution of extreme life span in Pacific Ocean rockfishes",
        "abstract": "Rockfishes of the genus Sebastes vary in maximum lifespan from about a decade to over two hundred years. Comparative analyses point to DNA repair pathways and immune gene copy number as candidates associated with extreme longevity. Depth and size also explain part of the variation.",
    },
    {
        "pmid": "3000002", "species": "rockfish", "pub_year": 2019,
        "title": "Otolith-based age validation in deep-dwelling Sebastes",
        "abstract": "Otolith-based ageing of Sebastes species confirms exceptional longevity in deep-dwelling taxa. Depth and body size explain a substantial share of lifespan variation among species. Bomb radiocarbon validation supports the age estimates for several taxa.",
    },
]


def seed_corpus(conn):
    for d in DOCS:
        doc = {k: v for k, v in d.items() if k != "species"}
        doc["retrieved_at"] = "2026-09-08T00:00:00+00:00"
        db.upsert_doc(conn, doc, d["species"])
    conn.commit()


@pytest.fixture
def conn():
    c = db.connect(":memory:")
    seed_corpus(c)
    yield c
    c.close()


@pytest.fixture
def docs_map(conn):
    return {d["pmid"]: d for d in db.get_docs(conn)}


@pytest.fixture
def db_file(tmp_path):
    path = tmp_path / "cora.db"
    c = db.connect(path)
    seed_corpus(c)
    c.close()
    return path
