"""PubMed ingestion via NCBI E-utilities (stdlib only).

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

from . import config, db

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
        "note": "PMID lists per species with the exact query and dates. Raw XML lives in the (uncommitted) snapshot dir.",
        "species": {},
        "runs": [],
    }


# --- full ingest ------------------------------------------------------------

def ingest_species(conn, species: config.Species, retmax: int = 10000, log=print, search_fn=None, fetch_fn=None) -> dict:
    """Fetch, snapshot, parse and store ALL aging-related abstracts for one species."""
    term = build_query(species)
    pmids = search_fn(term) if search_fn else esearch(term, retmax=retmax)
    log(f"[{species.key}] {len(pmids)} PMIDs for: {term}")
    docs = (fetch_fn or fetch_docs)(pmids, species.key, log=log) if pmids else []
    for d in docs:
        db.upsert_doc(conn, d, species.key)
    conn.commit()
    return {"species": species.key, "query": term, "pmids": pmids, "stored": len(docs), "dropped_no_abstract": len(pmids) - len(docs), "date": db.today()}


def ingest_all(conn, species_keys: list[str], log=print, search_fn=None, fetch_fn=None) -> list[dict]:
    results = []
    manifest = load_manifest() or _empty_manifest()
    for key in species_keys:
        sp = config.SPECIES[key]
        r = ingest_species(conn, sp, log=log, search_fn=search_fn, fetch_fn=fetch_fn)
        results.append(r)
        manifest["species"][key] = {
            "query": r["query"], "date": r["date"], "updated": r["date"],
            "n_pmids": len(r["pmids"]), "stored": r["stored"], "pmids": r["pmids"],
        }
    manifest.setdefault("runs", []).append({"date": db.today(), "kind": "ingest", "species": species_keys, "stored": sum(r["stored"] for r in results)})
    save_manifest(manifest)
    db.log_event(conn, "ingest", {"species": species_keys, "stored": sum(r["stored"] for r in results)})
    return results


# --- incremental update (used by the re-check) --------------------------------

def known_pmids(conn, species_key: str, manifest: dict | None) -> set[str]:
    """PMIDs we already have for a species: the manifest's list, or the DB if no manifest."""
    if manifest and species_key in manifest.get("species", {}):
        return set(manifest["species"][species_key].get("pmids", []))
    rows = conn.execute("SELECT pmid FROM doc_species WHERE species_key = ?", (species_key,)).fetchall()
    return {r["pmid"] for r in rows}


def update_species(conn, species: config.Species, manifest: dict | None, log=print, search_fn=None, fetch_fn=None) -> dict:
    """Search again, fetch only unseen PMIDs, store them. Returns the diff."""
    term = build_query(species)
    current = (search_fn or esearch)(term)
    old = known_pmids(conn, species.key, manifest)
    new = [p for p in current if p not in old]
    gone = sorted(old - set(current))
    docs = (fetch_fn or fetch_docs)(new, species.key, log=log) if new else []
    for d in docs:
        db.upsert_doc(conn, d, species.key)
    conn.commit()
    log(f"[{species.key}] {len(current)} PMIDs now; {len(new)} new, {len(gone)} no longer returned; stored {len(docs)}")
    return {"species": species.key, "query": term, "pmids": current, "new": new, "gone": gone, "stored": len(docs), "docs": docs, "date": db.today()}


def fetch_flags(pmids: list[str], fetch_fn=None, log=print) -> dict[str, list[str]]:
    """Re-fetch a small set of PMIDs (the ones live cards cite) and return their flags."""
    if not pmids:
        return {}
    docs = (fetch_fn or fetch_docs)(list(pmids), "flags", keep_empty=True, log=log)
    return {d["pmid"]: d.get("flags", []) for d in docs}
