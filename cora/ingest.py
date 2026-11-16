"""PubMed ingestion via NCBI E-utilities (stdlib only).

- One query per (topic, species) pair: the species' binomial query AND the topic's terms
  (see config.SPECIES and cora/topics.py). The same paper can arrive under several topics;
  it is stored once and linked to each.
- Queries by binomial / genus only (see config.SPECIES).
- Saves the raw efetch XML to a snapshot directory *before* parsing, so an eval can be
  run against exactly the records that were downloaded, never a re-fetch.
- Writes a manifest (PMIDs + query + dates) that is safe to commit.
- `update_species` fetches only PMIDs not seen before, so the weekly re-check
  (cora/recheck.py) costs a few requests, not a full re-download.
- Parses retraction / erratum / expression-of-concern links into per-document flags.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from . import config, db, topics

BATCH = 200
FLAG_REF_TYPES = ("RetractionIn", "ErratumIn", "ExpressionOfConcernIn", "RetractionOf")


def _get(url: str, retries: int = 4, timeout: int = 60) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 - fixed NCBI host
                return r.read()
        except Exception as e:  # network hiccups, 429s, 5xx
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"E-utilities request failed after {retries} attempts: {last}")


def _params(extra: dict) -> str:
    p = {"db": "pubmed", "tool": config.NCBI_TOOL}
    if config.NCBI_API_KEY:
        p["api_key"] = config.NCBI_API_KEY
    if config.NCBI_EMAIL:
        p["email"] = config.NCBI_EMAIL
    p.update(extra)
    return urllib.parse.urlencode(p)


def build_query(species: config.Species, topic: topics.Topic | None = None) -> str:
    """The PubMed term for one (topic, species) pair."""
    topic = topic or topics.BUILTIN[topics.DEFAULT_TOPIC]
    return topic.query(species.query, species.key)


def esearch(term: str, retmax: int = 10000) -> list[str]:
    url = f"{config.EUTILS_BASE}/esearch.fcgi?" + _params(
        {"term": term, "retmax": str(retmax), "retmode": "json", "datetype": "edat"}
    )
    data = json.loads(_get(url))
    time.sleep(config.EUTILS_SLEEP)
    return list(data.get("esearchresult", {}).get("idlist", []))


def efetch_xml(pmids: list[str]) -> str:
    url = f"{config.EUTILS_BASE}/efetch.fcgi?" + _params(
        {"id": ",".join(pmids), "retmode": "xml", "rettype": "abstract"}
    )
    raw = _get(url)
    time.sleep(config.EUTILS_SLEEP)
    return raw.decode("utf-8", errors="replace")


def _text(el: ET.Element | None) -> str:
    return "".join(el.itertext()).strip() if el is not None else ""


def parse_pubmed_xml(xml_text: str, keep_empty: bool = False) -> list[dict]:
    """Parse efetch XML into docs. Records with no abstract are dropped unless keep_empty
    (nothing to gate against), but their retraction/erratum flags are still parsed."""
    root = ET.fromstring(xml_text)
    docs: list[dict] = []
    for art in root.iter("PubmedArticle"):
        pmid = (art.findtext("MedlineCitation/PMID") or "").strip()
        if not pmid:
            continue
        title = _text(art.find("MedlineCitation/Article/ArticleTitle"))
        parts = []
        for at in art.findall("MedlineCitation/Article/Abstract/AbstractText"):
            body = _text(at)
            if not body:
                continue
            label = at.get("Label")
            parts.append(f"{label}: {body}" if label else body)
        abstract = " ".join(parts).strip()
        flags: list[str] = []
        pub_types: list[str] = []
        for pt in art.findall("MedlineCitation/Article/PublicationTypeList/PublicationType"):
            name = (pt.text or "").strip()
            if name:
                pub_types.append(name)
            if "retract" in name.lower():
                flags.append(f"pubtype:{name}")
        for cc in art.findall("MedlineCitation/CommentsCorrectionsList/CommentsCorrections"):
            rt = cc.get("RefType")
            if rt in FLAG_REF_TYPES:
                flags.append(f"{rt}:{(cc.findtext('PMID') or '?').strip()}")
        if not abstract and not keep_empty:
            continue
        journal = art.findtext("MedlineCitation/Article/Journal/Title")
        year = art.findtext("MedlineCitation/Article/Journal/JournalIssue/PubDate/Year")
        if not year:
            md = art.findtext("MedlineCitation/Article/Journal/JournalIssue/PubDate/MedlineDate") or ""
            digits = "".join(ch for ch in md[:4] if ch.isdigit())
            year = digits if len(digits) == 4 else None
        entrez = None
        for d in art.findall("PubmedData/History/PubMedPubDate"):
            if d.get("PubStatus") == "entrez":
                y, m, dd = d.findtext("Year"), d.findtext("Month"), d.findtext("Day")
                if y and m and dd:
                    entrez = f"{y}-{int(m):02d}-{int(dd):02d}"
        docs.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "pub_year": int(year) if year and year.isdigit() else None,
                "entrez_date": entrez,
                "flags": flags,
                "pub_types": pub_types,
            }
        )
    return docs


def fetch_docs(pmids: list[str], tag: str, keep_empty: bool = False, log=print) -> list[dict]:
    """efetch in batches, snapshot each batch's raw XML, parse. `tag` names the snapshot files."""
    config.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = db.today()
    out: list[dict] = []
    for i in range(0, len(pmids), BATCH):
        chunk = pmids[i : i + BATCH]
        xml_text = efetch_xml(chunk)
        snap = config.SNAPSHOT_DIR / f"{tag}_{stamp}_{i // BATCH:03d}.xml"
        snap.write_text(xml_text, encoding="utf-8")
        docs = parse_pubmed_xml(xml_text, keep_empty=keep_empty)
        for d in docs:
            d["retrieved_at"] = db.now_iso()
            d["snapshot_file"] = str(snap)
        out.extend(docs)
        log(f"[{tag}] batch {i // BATCH + 1}: parsed {len(docs)} of {len(chunk)}")
    return out


# --- manifest -------------------------------------------------------------

def load_manifest(path: Path | None = None) -> dict | None:
    path = path or config.MANIFEST_PATH
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(manifest: dict, path: Path | None = None) -> Path:
    path = path or config.MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return path


def _empty_manifest() -> dict:
    return {
        "note": "PMID lists per topic/species pair with the exact query and dates. Raw XML lives in the (uncommitted) snapshot dir.",
        "pairs": {},
        "species": {},  # legacy, pre-topics: aging only. Read, never written.
        "runs": [],
    }


def pair_id(topic_key: str, species_key: str) -> str:
    return f"{topic_key}/{species_key}"


def _snapshot_tag(topic_key: str, species_key: str) -> str:
    return f"{topic_key}_{species_key}"


# --- full ingest ------------------------------------------------------------

def ingest_species(
    conn,
    species: config.Species,
    topic: topics.Topic,
    retmax: int = 10000,
    log=print,
    search_fn=None,
    fetch_fn=None,
) -> dict:
    """Fetch, snapshot, parse and store every abstract for one (topic, species) pair."""
    term = build_query(species, topic)
    tag = _snapshot_tag(topic.key, species.key)
    pmids = search_fn(term) if search_fn else esearch(term, retmax=retmax)
    log(f"[{tag}] {len(pmids)} PMIDs for: {term}")
    docs = (fetch_fn or fetch_docs)(pmids, tag, log=log) if pmids else []
    for d in docs:
        db.upsert_doc(conn, d, species.key, topic_key=topic.key)
    conn.commit()
    db.record_ingest(conn, topic.key, species.key, term, len(pmids), len(docs))
    return {
        "topic": topic.key, "species": species.key, "query": term, "pmids": pmids,
        "stored": len(docs), "dropped_no_abstract": len(pmids) - len(docs), "date": db.today(),
    }


def ingest_all(
    conn,
    species_keys: list[str] | None = None,
    log=print,
    search_fn=None,
    fetch_fn=None,
    topic_keys: list[str] | None = None,
    pairs: list[tuple[str, str]] | None = None,
) -> list[dict]:
    """Ingest a set of (topic, species) pairs.

    `pairs` is explicit; otherwise the pairs are every species of every given topic,
    optionally narrowed to `species_keys`.
    """
    registry = topics.all_topics(conn)
    if pairs is None:
        tks = topics.resolve(topic_keys, conn)
        pairs = topics.pairs(tks, species_keys, conn)
    results = []
    manifest = load_manifest() or _empty_manifest()
    manifest.setdefault("pairs", {})
    for tk, sk in pairs:
        r = ingest_species(conn, config.SPECIES[sk], registry[tk], log=log, search_fn=search_fn, fetch_fn=fetch_fn)
        results.append(r)
        manifest["pairs"][pair_id(tk, sk)] = {
            "topic": tk, "species": sk, "query": r["query"], "date": r["date"], "updated": r["date"],
            "n_pmids": len(r["pmids"]), "stored": r["stored"], "pmids": r["pmids"],
        }
    stored = sum(r["stored"] for r in results)
    manifest.setdefault("runs", []).append(
        {"date": db.today(), "kind": "ingest", "pairs": [pair_id(t, s) for t, s in pairs], "stored": stored}
    )
    save_manifest(manifest)
    db.log_event(conn, "ingest", {"pairs": [pair_id(t, s) for t, s in pairs], "stored": stored})
    return results


# --- incremental update (used by the re-check) --------------------------------

def known_pmids(conn, species_key: str, manifest: dict | None, topic_key: str = topics.DEFAULT_TOPIC) -> set[str]:
    """PMIDs we already have for a (topic, species) pair: the manifest's list, else the DB.

    A manifest written before topics existed holds aging PMIDs under "species"; it is read
    for the aging topic so an existing install is not re-downloaded.
    """
    if manifest:
        entry = manifest.get("pairs", {}).get(pair_id(topic_key, species_key))
        if entry is None and topic_key == topics.DEFAULT_TOPIC:
            entry = manifest.get("species", {}).get(species_key)
        if entry is not None:
            return set(entry.get("pmids", []))
    rows = conn.execute(
        """SELECT ds.pmid FROM doc_species ds JOIN doc_topic dt ON dt.pmid = ds.pmid
           WHERE ds.species_key = ? AND dt.topic_key = ?""",
        (species_key, topic_key),
    ).fetchall()
    return {r["pmid"] for r in rows}


def update_species(
    conn,
    species: config.Species,
    manifest: dict | None,
    topic: topics.Topic | None = None,
    log=print,
    search_fn=None,
    fetch_fn=None,
) -> dict:
    """Search again for one (topic, species) pair, fetch only unseen PMIDs. Returns the diff."""
    topic = topic or topics.BUILTIN[topics.DEFAULT_TOPIC]
    term = build_query(species, topic)
    tag = _snapshot_tag(topic.key, species.key)
    current = (search_fn or esearch)(term)
    old = known_pmids(conn, species.key, manifest, topic.key)
    new = [p for p in current if p not in old]
    gone = sorted(old - set(current))
    docs = (fetch_fn or fetch_docs)(new, tag, log=log) if new else []
    for d in docs:
        db.upsert_doc(conn, d, species.key, topic_key=topic.key)
    conn.commit()
    if new:
        db.record_ingest(conn, topic.key, species.key, term, len(current), len(docs), accumulate=True)
    log(f"[{tag}] {len(current)} PMIDs now; {len(new)} new, {len(gone)} no longer returned; stored {len(docs)}")
    return {
        "topic": topic.key, "species": species.key, "query": term, "pmids": current, "new": new,
        "gone": gone, "stored": len(docs), "docs": docs, "date": db.today(),
    }


def fetch_flags(pmids: list[str], fetch_fn=None, log=print) -> dict[str, list[str]]:
    """Re-fetch a small set of PMIDs (the ones live cards cite) and return their flags."""
    if not pmids:
        return {}
    docs = (fetch_fn or fetch_docs)(list(pmids), "flags", keep_empty=True, log=log)
    return {d["pmid"]: d.get("flags", []) for d in docs}
