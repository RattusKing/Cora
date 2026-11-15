"""`cora setup`: everything a fresh install needs, in one command.

Runs, in order: database init, the topic choice, PubMed ingest for the chosen topics, the
AnAge lifespan table, the graph-lite findings, and a canary check of the citation gate.
Each step reports ok / skipped / failed and the run continues past a failure where it can,
so one flaky download does not hide the state of everything else.

Safe to re-run. (topic, species) pairs already ingested and an AnAge table already loaded
are kept unless `refresh=True` (new abstracts arrive through `cora recheck`, not setup).
Adding a topic later ingests only that topic's pairs. The graph and the canary always run,
because they are cheap and depend on the current corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import anage, config, db, graph, ingest, topics
from .gate import make_canaries, run_canaries


@dataclass
class Step:
    name: str
    status: str  # ok | skipped | failed
    detail: str = ""


@dataclass
class SetupResult:
    steps: list[Step] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(s.status != "failed" for s in self.steps)

    def add(self, name: str, status: str, detail: str = "") -> Step:
        step = Step(name, status, detail)
        self.steps.append(step)
        return step


def run(
    conn,
    species_keys: list[str] | None = None,
    refresh: bool = False,
    skip: set[str] | None = None,
    search_fn=None,
    fetch_fn=None,
    anage_download=None,
    log=print,
    db_label: str | None = None,
    topic_keys: list[str] | None = None,
) -> SetupResult:
    skip = set(skip or ())
    res = SetupResult()
    res.add("init", "ok", f"database at {db_label or config.DB_PATH}")

    # 1. Topics: what Cora is being pointed at. Remembered, so a re-run needs no answer.
    tks = topics.resolve(topic_keys, conn)
    if topic_keys:
        db.set_enabled_topics(conn, tks)
    registry = topics.all_topics(conn)
    pairs = topics.pairs(tks, species_keys, conn)
    names = ", ".join(registry[k].name for k in tks)
    beta = [registry[k].name for k in tks if registry[k].status != "ready"]
    detail = f"{len(tks)}: {names} ({len(pairs)} topic/species pairs)"
    if beta:
        detail += f"; beta lexicon: {', '.join(beta)}"
    res.add("topics", "ok", detail)
    species_keys = [s for _, s in pairs]

    # 2. PubMed ingest - only pairs not yet ingested, unless refreshing.
    if "ingest" in skip:
        res.add("ingest", "skipped", "--skip ingest")
    else:
        have = db.ingested_pairs(conn)
        todo = list(pairs) if refresh else [p for p in pairs if p not in have]
        if not todo:
            res.add("ingest", "skipped", f"all {len(pairs)} topic/species pairs already ingested (use --refresh to re-fetch; new papers arrive via `cora recheck`)")
        else:
            log(f"[setup] ingesting {len(todo)} topic/species pairs from PubMed")
            try:
                before = db.count_docs(conn)
                ingest.ingest_all(conn, log=log, search_fn=search_fn, fetch_fn=fetch_fn, pairs=todo)
                added = db.count_docs(conn) - before
                res.add("ingest", "ok", f"{added} new unique abstracts for {len(todo)} topic/species pairs")
            except Exception as e:  # network, NCBI outage
                res.add("ingest", "failed", f"{type(e).__name__}: {e}")

    # 3. AnAge lifespan table - only for a topic that is backed by it.
    anage_topics = [k for k in tks if registry[k].trait_source == "anage"]
    if "anage" in skip:
        res.add("anage", "skipped", "--skip anage")
    elif not anage_topics:
        res.add("anage", "skipped", "no enabled topic uses the AnAge lifespan table")
    elif anage.loaded_count(conn) and not refresh:
        res.add("anage", "skipped", f"{anage.loaded_count(conn)} rows already loaded (use --refresh to re-download)")
    else:
        try:
            path = (anage_download or anage.download)()
            n = anage.load(conn, path=path)
            panel = topics.species_for(anage_topics, conn)
            matched = sum(1 for k, t in anage.panel_traits(conn, panel).items() if t)
            res.add("anage", "ok", f"{n} rows loaded; {matched} of {len(panel)} panel species matched")
        except Exception as e:
            res.add("anage", "failed", f"{type(e).__name__}: {e}")

    total_docs = db.count_docs(conn, tks)

    # 4. Graph-lite findings (lexicon extractor: no model needed).
    if "graph" in skip:
        res.add("graph", "skipped", "--skip graph")
    elif total_docs == 0:
        res.add("graph", "skipped", "no abstracts in the corpus yet")
    else:
        try:
            stats = graph.build(conn, graph.LexiconExtractor(), log=log, topic_keys=tks)
            detail = f"{stats['findings']} findings from {stats['docs']} abstracts, {stats['dropped_by_gate']} dropped by the gate"
            res.add("graph", "failed" if stats["dropped_by_gate"] else "ok", detail)
        except Exception as e:
            res.add("graph", "failed", f"{type(e).__name__}: {e}")

    # 5. Canary check of the citation gate on the real corpus.
    if "canary" in skip:
        res.add("canary", "skipped", "--skip canary")
    elif total_docs == 0:
        res.add("canary", "skipped", "no abstracts in the corpus yet")
    else:
        docs = db.get_docs(conn, topic_keys=tks)
        result = run_canaries({d["pmid"]: d for d in docs}, make_canaries(docs, n=20))
        db.log_event(conn, "canary", {k: v for k, v in result.items() if k != "misses"})
        detail = f"{result['known_bad_caught']}/{result['known_bad']} known-bad caught, {result['known_good_passed']}/{result['known_good']} known-good passed"
        res.add("canary", "ok" if result["ok"] else "failed", detail)

    db.log_event(conn, "setup", {"ok": res.ok, "topics": tks, "steps": {s.name: s.status for s in res.steps}})
    return res


# An example question per topic, so the "ready, try:" line fits what was actually ingested.
EXAMPLE_QUERY = {
    "aging": "DNA repair and extreme lifespan",
    "regeneration": "blastema formation without scarring",
    "cancer_resistance": "tumour suppressor copy number and cancer resistance",
    "hypoxia": "metabolic depression under anoxia",
    "hibernation": "muscle preserved through months of torpor",
}


def render(res: SetupResult, conn) -> str:
    mark = {"ok": "ok  ", "skipped": "skip", "failed": "FAIL"}
    lines = ["", "setup summary"]
    for s in res.steps:
        lines.append(f"  [{mark[s.status]}] {s.name:<7} {s.detail}")
    tks = topics.resolve(None, conn)
    counts = db.count_docs_by_species(conn, tks)
    by_topic = db.count_docs_by_topic(conn)
    lines.append(f"  corpus: {db.count_docs(conn, tks)} unique abstracts across {len(counts)} species")
    for k in tks:
        lines.append(f"    {k:<18} {by_topic.get(k, 0)} abstracts")
    lines.append("")
    if res.ok:
        example = EXAMPLE_QUERY.get(tks[0], "a mechanism this topic cares about")
        lines.append("ready. try:")
        lines.append(f'  cora ask "{example}"')
        lines.append(f'  cora ask "{example}" --mock   # no API key needed')
        lines.append("  cora topics                                         # add or remove topics")
        lines.append("  cora serve                                          # web UI at http://127.0.0.1:8000")
    else:
        lines.append("some steps failed - fix the cause above and re-run `cora setup` (completed steps are skipped)")
    return "\n".join(lines)
