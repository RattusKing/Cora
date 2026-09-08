"""cora - command line interface for P0.5."""

from __future__ import annotations

import argparse
import json
import sys

from . import config, db, generate, ledger, metrics
from .card import export_draft, render_tight
from .gate import make_canaries, run_canaries
from .llm import get_llm


def _conn(args):
    conn = db.connect(getattr(args, "db", None))
    metrics.record_open(conn)
    return conn


def cmd_init(args):
    conn = _conn(args)
    print(f"database ready: {getattr(args, 'db', None) or config.DB_PATH}")
    print("species (default):", ", ".join(config.DEFAULT_SPECIES))
    conn.close()


def cmd_ingest(args):
    from . import ingest

    conn = _conn(args)
    keys = args.species or config.DEFAULT_SPECIES
    for k in keys:
        if k not in config.SPECIES:
            sys.exit(f"unknown species key {k!r}; known: {', '.join(config.SPECIES)}")
    results = ingest.ingest_all(conn, keys)
    for r in results:
        print(f"{r['species']}: {r['stored']} abstracts stored ({r['dropped_no_abstract']} had no abstract)")
    print(f"manifest: {config.MANIFEST_PATH}")
    conn.close()


def cmd_ask(args):
    conn = _conn(args)
    try:
        llm = get_llm(mock=args.mock)
    except Exception as e:  # SDK import / client construction problems
        sys.exit(f"could not create model client: {e}\n(use --mock, or set ANTHROPIC_API_KEY / `ant auth login`)")
    try:
        card = generate.ask(conn, args.query, species_keys=args.species, llm=llm, k=args.k)
    except generate.NoEvidence as e:
        sys.exit(str(e))
    except Exception as e:
        name = type(e).__name__
        if "Authentication" in name or "PermissionDenied" in name:
            sys.exit(f"model call failed: {e}\n(set ANTHROPIC_API_KEY or run `ant auth login`; or use --mock)")
        raise
    docs_map = db.docs_by_pmid(conn, [e.pmid for e in card.draft.evidence])
    print(render_tight(card, docs_map))
    g = card.gate
    print(f"\n  gate    {g.n_passed}/{g.n_items} quotes verified · fabrication rate {g.fabrication_rate:.0%}")
    for f in g.failed:
        print(f"          dropped [{f['reason']}] PMID {f['pmid']}: \"{f['quote'][:70]}\"")
    conn.close()


def cmd_ledger(args):
    conn = _conn(args)
    rows = ledger.list_cards(conn, include_archived=args.all, limit=args.limit)
    if not rows:
        print("ledger is empty - try: cora ask \"...\"")
    for r in rows:
        card = r["card"]
        docs_map = db.docs_by_pmid(conn, [e.pmid for e in card.draft.evidence])
        tag = f" [{r['state']}]" + (f" (dup of {r['duplicate_of']})" if r["duplicate_of"] else "")
        print(render_tight(card, docs_map).replace(f"[{card.id}]", f"[{card.id}]{tag}", 1))
        print()
    conn.close()


def cmd_show(args):
    conn = _conn(args)
    card = ledger.get(conn, args.id)
    if card is None:
        sys.exit(f"no card {args.id}")
    ledger.mark_expanded(conn, args.id)
    docs_map = db.docs_by_pmid(conn, [e.pmid for e in card.draft.evidence])
    print(render_tight(card, docs_map))
    print("\n  evidence (gate-verified):")
    for e in card.draft.evidence:
        d = docs_map.get(e.pmid, {})
        print(f"   - PMID {e.pmid} ({d.get('pub_year') or '?'}; {', '.join(d.get('species', []))}): \"{e.quote}\"")
    if card.gate.failed:
        print("  dropped by the gate:")
        for f in card.gate.failed:
            print(f"   - [{f['reason']}] PMID {f['pmid']}: \"{f['quote'][:90]}\"")
    hl = card.draft.human_lever
    print(f"  human lever: gene={hl.gene} direction={hl.direction} type={hl.lever_type} - {hl.note}")
    conn.close()


def cmd_feedback(args):
    conn = _conn(args)
    try:
        ledger.feedback(conn, args.id, args.verdict, args.reason)
    except KeyError:
        sys.exit(f"no card {args.id}")
    print(f"recorded {args.verdict} for {args.id}")
    conn.close()


def cmd_export(args):
    conn = _conn(args)
    card = ledger.get(conn, args.id)
    if card is None:
        sys.exit(f"no card {args.id}")
    docs_map = db.docs_by_pmid(conn, [e.pmid for e in card.draft.evidence])
    ledger.mark_exported(conn, args.id)
    print(export_draft(card, docs_map))
    conn.close()


def cmd_canary(args):
    conn = _conn(args)
    docs = db.get_docs(conn)
    if not docs:
        sys.exit("no documents in the corpus - run `cora ingest` first")
    canaries = make_canaries(docs, n=args.n)
    result = run_canaries({d["pmid"]: d for d in docs}, canaries)
    db.log_event(conn, "canary", {k: v for k, v in result.items() if k != "misses"})
    print(json.dumps(result, indent=1))
    if not result["ok"]:
        sys.exit(2)
    conn.close()


def cmd_metrics(args):
    conn = _conn(args)
    print(json.dumps(metrics.compute(conn), indent=1))
    conn.close()


def cmd_checkin(args):
    conn = _conn(args)
    metrics.record_checkin(conn, args.answer == "yes", args.note)
    print("recorded weekly check-in:", args.answer)
    conn.close()


def cmd_evalset(args):
    from . import evalset

    conn = _conn(args)
    path = evalset.write_verifier_candidates(conn, n=args.n, out=args.out)
    print(f"wrote verifier candidates to {path} - label them by hand (see eval/README.md)")
    conn.close()


def cmd_serve(args):
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(db_path=getattr(args, "db", None)), host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cora", description="Cora P0.5 - citation-gated hypothesis cards")
    p.add_argument("--db", help=f"SQLite path (default {config.DB_PATH})")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="create the database").set_defaults(fn=cmd_init)

    s = sub.add_parser("ingest", help="fetch PubMed abstracts for the species panel")
    s.add_argument("--species", nargs="*", help=f"species keys (default: {' '.join(config.DEFAULT_SPECIES)})")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("ask", help="draft one gated card for a query")
    s.add_argument("query")
    s.add_argument("--species", nargs="*")
    s.add_argument("--mock", action="store_true", help="use the deterministic mock drafter (no API key needed)")
    s.add_argument("--k", type=int, default=None, help="passages to retrieve")
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("ledger", help="ranked cards")
    s.add_argument("--all", action="store_true", help="include archived")
    s.add_argument("--limit", type=int, default=50)
    s.set_defaults(fn=cmd_ledger)

    s = sub.add_parser("show", help="expand a card (full evidence)")
    s.add_argument("id")
    s.set_defaults(fn=cmd_show)

    s = sub.add_parser("feedback", help="accept | reject:wrong | reject:uninteresting | dig_deeper")
    s.add_argument("id")
    s.add_argument("verdict", choices=ledger.VERDICTS)
    s.add_argument("--reason")
    s.set_defaults(fn=cmd_feedback)

    s = sub.add_parser("export", help="citation-complete draft paragraph (marks the card as having left Cora)")
    s.add_argument("id")
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser("canary", help="run known-good/known-bad citations through the gate")
    s.add_argument("--n", type=int, default=10)
    s.set_defaults(fn=cmd_canary)

    sub.add_parser("metrics", help="system + user metrics").set_defaults(fn=cmd_metrics)

    s = sub.add_parser("checkin", help="weekly: did Cora tell you something you used?")
    s.add_argument("answer", choices=["yes", "no"])
    s.add_argument("--note")
    s.set_defaults(fn=cmd_checkin)

    s = sub.add_parser("evalset", help="write UNLABELED verifier-gold candidate pairs for hand labeling")
    s.add_argument("--n", type=int, default=60)
    s.add_argument("--out", default="eval/verifier_gold_candidates.jsonl")
    s.set_defaults(fn=cmd_evalset)

    s = sub.add_parser("serve", help="run the web UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
