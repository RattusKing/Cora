"""PubMed ingestion via NCBI E-utilities (stdlib only).

- Queries by binomial / genus only (see config.SPECIES).
- Saves the raw efetch XML to a snapshot directory *before* parsing, so an eval can be
  run against exactly the records that were downloaded, never a re-fetch.
- Writes a manifest (PMIDs + query + download date) that is safe to commit.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from . import config, db

BATCH = 200


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


def build_query(species: config.Species) -> str:
    return f"{species.query} AND {config.AGING_TERMS}"


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


def parse_pubmed_xml(xml_text: str) -> list[dict]:
    """Parse efetch XML into docs. Records with no abstract are dropped (nothing to gate against)."""
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
        if not abstract:
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
            }
        )
    return docs


def ingest_species(conn, species: config.Species, retmax: int = 10000, log=print) -> dict:
    """Fetch, snapshot, parse and store all aging-related abstracts for one species."""
    term = build_query(species)
    pmids = esearch(term, retmax=retmax)
    log(f"[{species.key}] {len(pmids)} PMIDs for: {term}")
    config.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = db.today()
    stored = 0
    dropped = 0
    for i in range(0, len(pmids), BATCH):
        chunk = pmids[i : i + BATCH]
        xml_text = efetch_xml(chunk)
        snap = config.SNAPSHOT_DIR / f"{species.key}_{stamp}_{i // BATCH:03d}.xml"
        snap.write_text(xml_text, encoding="utf-8")
        docs = parse_pubmed_xml(xml_text)
        dropped += len(chunk) - len(docs)
        for d in docs:
            d["retrieved_at"] = db.now_iso()
            d["snapshot_file"] = str(snap)
            db.upsert_doc(conn, d, species.key)
            stored += 1
        conn.commit()
        log(f"[{species.key}] batch {i // BATCH + 1}: stored {len(docs)} (no-abstract dropped: {len(chunk) - len(docs)})")
    return {"species": species.key, "query": term, "pmids": pmids, "stored": stored, "dropped_no_abstract": dropped, "date": stamp}


def write_manifest(results: list[dict], path: Path | None = None) -> Path:
    path = path or config.MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "note": "PMID lists per species with the exact query and download date. Raw XML lives in the (uncommitted) snapshot dir.",
        "species": {
            r["species"]: {"query": r["query"], "date": r["date"], "n_pmids": len(r["pmids"]), "stored": r["stored"], "pmids": r["pmids"]}
            for r in results
        },
    }
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return path


def ingest_all(conn, species_keys: list[str], log=print) -> list[dict]:
    results = []
    for key in species_keys:
        sp = config.SPECIES[key]
        results.append(ingest_species(conn, sp, log=log))
    write_manifest(results)
    db.log_event(conn, "ingest", {"species": species_keys, "stored": sum(r["stored"] for r in results)})
    return results
