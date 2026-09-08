from cora.retrieve import Index, build_index, tokenize


def test_tokenize_drops_stopwords():
    assert "the" not in tokenize("the DNA repair of the cell")
    assert "dna" in tokenize("the DNA repair")


def test_search_ranks_relevant_doc_first(conn):
    idx = build_index(conn)
    hits = idx.search("hyaluronan cancer resistance", k=3)
    assert hits and hits[0][0]["pmid"] == "1000001"


def test_species_filter(conn):
    idx = build_index(conn)
    hits = idx.search("longevity lifespan", k=10, species_keys=["rockfish"])
    assert hits and all("rockfish" in d["species"] for d, _ in hits)


def test_empty_index():
    assert Index([]).search("anything") == []
