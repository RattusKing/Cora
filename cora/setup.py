"""`cora setup`: everything a fresh install needs, in one command.

Runs, in order: database init, PubMed ingest for the panel, the AnAge lifespan table,
the graph-lite findings, and a canary check of the citation gate. Each step reports
ok / skipped / failed and the run continues past a failure where it can, so one flaky
download does not hide the state of everything else.

Safe to re-run. Species already in the corpus and an AnAge table already loaded are kept
unless `refresh=True` (new abstracts arrive through `cora recheck`, not setup). The graph
and the canary always run, because they are cheap and depend on the current corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import anage, config, db, graph, ingest
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
) -> SetupResult:
    species_keys = list(species_keys or config.DEFAULT_SPECIES)
    skip = set(skip or ())
    res = SetupResult()
    res.add("init", "ok", f"database at {db_label or config.DB_PATH}")

    # 1. PubMed ingest - only species not yet in the corpus, unless refreshing.
    if "ingest" in skip:
        res.add("ingest", "skipped", "--skip ingest")
    else:
        have = db.count_docs_by_species(conn)
        todo = species_keys if refresh else [k for k in species_keys if not have.get(k)]
        if not todo:
            res.add("ingest", "skipped", f"all {len(species_keys)} species already ingested (use --refresh to re-fetch; new papers arrive via `cora recheck`)")
        else:
            log(f"[setup] ingesting {len(todo)} species from PubMed: {', '.join(todo)}")
            try:
                before = db.count_docs(conn)
                ingest.ingest_all(conn, todo, log=log, search_fn=search_fn, fetch_fn=fetch_fn)
                added = db.count_docs(conn) - before
                res.add("ingest", "ok", f"{added} new unique abstracts for {len(todo)} species")
            except Exception as e:  # network, NCBI outage
                res.add("ingest", "failed", f"{type(e).__name__}: {e}")

    # 2. AnAge lifespan table.
    if "anage" in skip:
        res.add("anage", "skipped", "--skip anage")
    elif anage.loaded_count(conn) and not refresh:
        res.add("anage", "skipped", f"{anage.loaded_count(conn)} rows already loaded (use --refresh to re-download)")
    else:
        try:
            path = (anage_download or anage.download)()
            n = anage.load(conn, path=path)
            matched = sum(1 for t in anage.panel_traits(conn).values() if t)
            res.add("anage", "ok", f"{n} rows loaded; {matched} of {len(config.SPECIES)} panel species matched")
        except Exception as e:
            res.add("anage", "failed", f"{type(e).__name__}: {e}")

    total_docs = db.count_docs(conn)

    # 3. Graph-lite findings (lexicon extractor: no model needed).
    if "graph" in skip:
        res.add("graph", "skipped", "--skip graph")
    elif total_docs == 0:
        res.add("graph", "skipped", "no abstracts in the corpus yet")
    else:
        try:
            stats = graph.build(conn, graph.LexiconExtractor(), log=log)
            detail = f"{stats['findings']} findings from {stats['docs']} abstracts, {stats['dropped_by_gate']} dropped by the gate"
            res.add("graph", "failed" if stats["dropped_by_gate"] else "ok", detail)
        except Exception as e:
            res.add("graph", "failed", f"{type(e).__name__}: {e}")

    # 4. Canary check of the citation gate on the real corpus.
    if "canary" in skip:
        res.add("canary", "skipped", "--skip canary")
    elif total_docs == 0:
        res.add("canary", "skipped", "no abstracts in the corpus yet")
    else:
        docs = db.get_docs(conn)
        result = run_canaries({d["pmid"]: d for d in docs}, make_canaries(docs, n=20))
        db.log_event(conn, "canary", {k: v for k, v in result.items() if k != "misses"})
        detail = f"{result['known_bad_caught']}/{result['known_bad']} known-bad caught, {result['known_good_passed']}/{result['known_good']} known-good passed"
        res.add("canary", "ok" if result["ok"] else "failed", detail)

    db.log_event(conn, "setup", {"ok": res.ok, "steps": {s.name: s.status for s in res.steps}})
    return res


def render(res: SetupResult, conn) -> str:
    mark = {"ok": "ok  ", "skipped": "skip", "failed": "FAIL"}
    lines = ["", "setup summary"]
    for s in res.steps:
        lines.append(f"  [{mark[s.status]}] {s.name:<7} {s.detail}")
    counts = db.count_docs_by_species(conn)
    lines.append(f"  corpus: {db.count_docs(conn)} unique abstracts across {len(counts)} species")
    lines.append("")
    if res.ok:
        lines.append("ready. try:")
        lines.append('  cora ask "DNA repair and extreme lifespan"          # needs ANTHROPIC_API_KEY or `ant auth login`')
        lines.append('  cora ask "DNA repair and extreme lifespan" --mock   # no key needed')
        lines.append("  cora serve                                          # web UI at http://127.0.0.1:8000")
    else:
        lines.append("some steps failed - fix the cause above and re-run `cora setup` (completed steps are skipped)")
    return "\n".join(lines)
