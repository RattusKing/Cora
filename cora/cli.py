"""cora - command line interface."""

from __future__ import annotations

import argparse
import json
import sys

from . import config, db, generate, ledger, metrics, topics
from .card import export_draft, render_tight
from .gate import make_canaries, run_canaries
from .llm import get_llm
from .verify import get_judge


def _conn(args):
    conn = db.connect(getattr(args, "db", None))
    metrics.record_open(conn)
    return conn


def _quiet(*a, **k):
    pass


def cmd_setup(args):
    from . import setup

    conn = _conn(args)
    tks = args.topic
    if not tks and not db.enabled_topics(conn):
        tks = _ask_topics(conn)  # first run: ask once, then remember
    try:
        tks = topics.resolve(tks, conn)
    except topics.TopicError as e:
        conn.close()
        sys.exit(str(e))
    keys = args.species or topics.species_for(tks, conn)
    _validate_species(keys)
    previously = db.enabled_topics(conn)
    res = setup.run(
        conn, args.species and keys or None, refresh=args.refresh, skip=set(args.skip or ()),
        log=print if args.verbose else _quiet,
        db_label=str(getattr(args, "db", None) or config.DB_PATH), topic_keys=tks,
    )
    if previously and set(previously) != set(tks):
        # --topic is a transient scope everywhere else; here it is remembered, so say so
        # rather than let the next `cora ask` quietly stop seeing the old corpus.
        print(f"note: enabled topics changed from [{', '.join(previously)}] to [{', '.join(tks)}]."
              f" `cora topics --set {' '.join(previously)}` puts it back; the abstracts are all still stored.")
    print(setup.render(res, conn))
    conn.close()
    if not res.ok:
        sys.exit(1)


def _ask_topics(conn) -> list[str]:
    """The first-run question. Falls back to the default topic when stdin is not a terminal."""
    registry = topics.all_topics(conn)
    if not sys.stdin.isatty():
        print(f"(not a terminal: defaulting to the {registry[topics.DEFAULT_TOPIC].name} topic; `cora topics` changes it)")
        return [topics.DEFAULT_TOPIC]
    print(topics.render_picker(registry, enabled=[], counts=db.count_docs_by_topic(conn)))
    for _ in range(3):
        try:
            answer = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        try:
            return topics.parse_selection(answer, registry)
        except topics.TopicError as e:
            print(f"  {e}")
    print(f"(no selection: defaulting to {registry[topics.DEFAULT_TOPIC].name})")
    return [topics.DEFAULT_TOPIC]


def cmd_init(args):
    conn = _conn(args)
    tks = topics.resolve(None, conn)
    print(f"database ready: {getattr(args, 'db', None) or config.DB_PATH}")
    print("topics:", ", ".join(tks))
    print("species:", ", ".join(topics.species_for(tks, conn)))
    print(f"drafter: {config.DEFAULT_MODEL}   judge: {config.JUDGE_MODEL}   pattern retries: {config.PATTERN_RETRIES}")
    conn.close()


def _validate_species(keys):
    for k in keys:
        if k not in config.SPECIES:
            sys.exit(f"unknown species key {k!r}; known: {', '.join(config.SPECIES)}")


def _topics_or_exit(conn, keys) -> list[str]:
    try:
        return topics.resolve(keys, conn)
    except topics.TopicError as e:
        sys.exit(str(e))


def cmd_ingest(args):
    from . import ingest

    conn = _conn(args)
    tks = _topics_or_exit(conn, args.topic)
    keys = args.species
    if keys:
        _validate_species(keys)
    results = ingest.ingest_all(conn, keys, topic_keys=tks)
    for r in results:
        print(f"{r['topic']}/{r['species']}: {r['stored']} abstracts stored ({r['dropped_no_abstract']} had no abstract)")
    print(f"manifest: {config.MANIFEST_PATH}")
    conn.close()


def cmd_recheck(args):
    from . import recheck

    conn = _conn(args)
    if args.report:
        report = recheck.last_report(conn)
        if report is None:
            print("no re-check has run yet")
        else:
            cards = {r["card"].id: r["card"] for r in ledger.list_cards(conn, include_archived=False, limit=10000)}
            print(recheck.render_report(report, cards))
        conn.close()
        return
    tks = _topics_or_exit(conn, args.topic)
    keys = args.species
    if keys:
        _validate_species(keys)
    report = recheck.run(conn, keys, k=args.k, log=print if args.verbose else _quiet, topic_keys=tks)
    cards = {r["card"].id: r["card"] for r in ledger.list_cards(conn, include_archived=False, limit=10000)}
    print(recheck.render_report(report, cards))
    conn.close()


def _credential_hint(e: Exception) -> str:
    name = type(e).__name__
    if "Authentication" in name or "PermissionDenied" in name:
        return f"model call failed: {e}\n(set ANTHROPIC_API_KEY or run `ant auth login`; or use --mock)"
    return ""


def cmd_ask(args):
    conn = _conn(args)
    try:
        llm = get_llm(mock=args.mock)
    except Exception as e:  # SDK import / client construction problems
        sys.exit(f"could not create model client: {e}\n(use --mock, or set ANTHROPIC_API_KEY / `ant auth login`)")
    judge = None
    if not args.no_judge:
        try:
            judge = get_judge(mock=args.mock, drafter_model=getattr(llm, "model", None), model=args.judge_model)
        except ValueError as e:
            sys.exit(str(e))
        except Exception as e:
            sys.exit(f"could not create judge client: {e}\n(use --no-judge for the mechanical check only, or --mock)")
    try:
        card = generate.ask(conn, args.query, species_keys=args.species, llm=llm, judge=judge, k=args.k, use_judge=not args.no_judge, topic_keys=_topics_or_exit(conn, args.topic))
    except generate.NoEvidence as e:
        sys.exit(str(e))
    except Exception as e:
        hint = _credential_hint(e)
        if hint:
            sys.exit(hint)
        raise
    docs_map = db.docs_by_pmid(conn, [e.pmid for e in card.draft.evidence])
    print(render_tight(card, docs_map))
    g = card.gate
    print(f"\n  gate    {g.n_passed}/{g.n_items} quotes verified · fabrication rate {g.fabrication_rate:.0%}")
    for f in g.failed:
        print(f"          dropped [{f['reason']}] PMID {f['pmid']}: \"{f['quote'][:70]}\"")
    pc = card.pattern_check
    if pc is not None:
        if pc.retried:
            print(f"  pattern rewritten once; original: \"{pc.original_pattern}\"")
        if not pc.passed:
            print(f"  status  {card.status.upper()} - {pc.rationale}")
        if pc.disagreement:
            print(f"  note    mechanical strength flags {pc.mechanical_strong_words} disagree with the judge's verdict ({pc.strength})")
    conn.close()


def cmd_ledger(args):
    conn = _conn(args)
    rows = ledger.list_cards(conn, include_archived=args.all, limit=args.limit, topic_keys=_topics_or_exit(conn, args.topic))
    if not rows:
        print("ledger is empty - try: cora ask \"...\"")
    for r in rows:
        card = r["card"]
        docs_map = db.docs_by_pmid(conn, [e.pmid for e in card.draft.evidence])
        tag = f" [{r['state']}]" + (f" (dup of {r['duplicate_of']})" if r["duplicate_of"] else "") + (f" new+{r['new_evidence']}" if r["new_evidence"] else "")
        print(render_tight(card, docs_map).replace(f"[{card.id}]", f"[{card.id}]{tag}", 1))
        print()
    conn.close()


def cmd_show(args):
    conn = _conn(args)
    card = ledger.get(conn, args.id)
    if card is None:
        sys.exit(f"no card {args.id}")
    updates = db.card_updates(conn, args.id)
    ledger.mark_expanded(conn, args.id)
    docs_map = db.docs_by_pmid(conn, [e.pmid for e in card.draft.evidence])
    print(render_tight(card, docs_map))
    print("\n  evidence (gate-verified):")
    for e in card.draft.evidence:
        d = docs_map.get(e.pmid, {})
        flag = f"  !! {', '.join(d['flags'])}" if d.get("flags") else ""
        print(f"   - PMID {e.pmid} ({d.get('pub_year') or '?'}; {', '.join(d.get('species', []))}): \"{e.quote}\"{flag}")
    if card.gate.failed:
        print("  dropped by the gate:")
        for f in card.gate.failed:
            print(f"   - [{f['reason']}] PMID {f['pmid']}: \"{f['quote'][:90]}\"")
    pc = card.pattern_check
    if pc is not None:
        print(f"  pattern check: passed={pc.passed} judged={pc.judged} judge={pc.judge_model} supported={pc.supported} strength={pc.strength} reason={pc.reason}")
        print(f"    rationale: {pc.rationale}")
        if pc.retried:
            print(f"    original pattern: \"{pc.original_pattern}\"")
    if updates:
        print("  what changed since this card was made:")
        for u in updates:
            print(f"   - {u['created_at'][:10]} PMID {u['pmid']}: {u['reason']}")
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
    if card.status != "gated":
        print(f"warning: card status is {card.status}; the export is marked accordingly", file=sys.stderr)
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
    conn.close()
    if not result["ok"]:
        sys.exit(2)


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
    path = evalset.write_verifier_candidates(conn, n=args.n, out=args.out, topic_keys=args.topic)
    print(f"wrote verifier candidates to {path} - label them with `cora label` (see eval/README.md)")
    conn.close()


def cmd_label(args):
    from . import label as lab

    if args.stats or args.score_judge:
        rows = lab.load_rows(args.file)
        print(json.dumps(lab.stats(rows), indent=1))
        if any(r.get("label") in lab.LABELS for r in rows):
            print("mechanical gate baseline:", json.dumps(lab.score_mechanical_gate(args.file), indent=1))
        else:
            print("no labels yet - run `cora label` first")
        if args.score_judge:
            try:
                judge = get_judge(mock=args.mock, model=args.judge_model)
            except Exception as e:
                sys.exit(f"could not create judge client: {e}\n(use --mock, or set ANTHROPIC_API_KEY / `ant auth login`)")
            try:
                print("judge:", json.dumps(lab.score_judge(rows, judge), indent=1))
            except Exception as e:
                hint = _credential_hint(e)
                sys.exit(hint or f"judge scoring failed: {e}")
        return
    try:
        lab.label_loop(args.file, reveal=args.reveal, only_unlabeled=not args.relabel)
    except FileNotFoundError as e:
        sys.exit(str(e))


def cmd_anage(args):
    from . import anage

    conn = _conn(args)
    if args.file:
        path = args.file
    else:
        print(f"downloading {config.ANAGE_URL} ...")
        path = anage.download()
    n = anage.load(conn, path=path)
    print(f"loaded {n} AnAge species rows from {path}")
    print(anage.render_summary(anage.panel_summary(conn)))
    conn.close()


def cmd_species(args):
    from . import anage

    conn = _conn(args)
    tks = _topics_or_exit(conn, args.topic)
    if anage.loaded_count(conn) == 0:
        print("AnAge not loaded yet - run `cora anage` to fetch lifespan records (docs counts below are still real)")
    print(f"topics: {', '.join(tks)}")
    print(anage.render_summary(anage.panel_summary(conn, topic_keys=tks)))
    conn.close()


def cmd_graph(args):
    from . import graph

    conn = _conn(args)
    if args.graph_cmd == "build":
        try:
            extractor = graph.get_extractor(mock=not args.model, model=args.extract_model)
        except Exception as e:
            sys.exit(f"could not create extractor: {e}\n(omit --model to use the lexicon extractor, which needs no credentials)")
        try:
            stats = graph.build(conn, extractor, species_keys=args.species, log=print, topic_keys=_topics_or_exit(conn, args.topic))
        except Exception as e:
            hint = _credential_hint(e)
            sys.exit(hint or f"graph build failed: {e}")
        print(json.dumps({k: v for k, v in stats.items() if k != "by_mechanism"}, indent=1))
        print("findings by mechanism:", json.dumps(stats["by_mechanism"]))
        if stats["docs_without_pub_types"]:
            print(f"note: {stats['docs_without_pub_types']} docs have no publication types (ingested before this feature); re-run `cora ingest` to refresh them so reviews can be separated from primary studies")
    elif args.graph_cmd == "converge":
        if graph.findings_count(conn) == 0:
            sys.exit("no findings yet - run `cora graph build` first")
        rows = graph.converge(conn, min_tier=args.min_tier, perms=args.perms, extractor=args.extractor, topic_keys=_topics_or_exit(conn, args.topic))
        print(graph.render_convergence(rows, min_lineages=args.min_lineages))
    elif args.graph_cmd == "show":
        rows = graph.mechanism_detail(conn, args.mechanism, extractor=args.extractor, topic_keys=_topics_or_exit(conn, args.topic))
        if not rows:
            sys.exit(f"no findings for mechanism {args.mechanism!r}; known: {', '.join(graph.MECHANISMS)}")
        spec = graph.MECHANISMS.get(args.mechanism, {})
        print(f"{args.mechanism} - {spec.get('label', '')}  hallmarks: {', '.join(spec.get('hallmarks', [])) or 'unassigned'}")
        for r in rows:
            d = f" {r['direction']}" if r["direction"] else ""
            print(f"  [{r['species_key']}] {r['tier']}{d} · {r['evidence_type']} · PMID {r['pmid']} ({r['pub_year'] or '?'})")
            print(f"      \"{r['quote'][:200]}\"")
    conn.close()


def cmd_topics(args):
    conn = _conn(args)
    registry = topics.all_topics(conn)
    counts = db.count_docs_by_topic(conn)

    if args.export:
        if args.export not in registry:
            sys.exit(f"unknown topic {args.export!r}; known: {', '.join(registry)}")
        print(topics.to_json(registry[args.export]))
        conn.close()
        return
    if args.add:
        _add_topic(conn, args)
        conn.close()
        return
    if args.remove:
        # Check every key first: deleting some and then exiting on a typo would leave the
        # enabled list naming topics that no longer exist.
        for key in args.remove:
            if key in topics.BUILTIN:
                sys.exit(f"{key!r} is built in and cannot be deleted; drop it from your selection instead (`cora topics --set ...`)")
            if key not in registry:
                sys.exit(f"no custom topic {key!r}; custom topics: {', '.join(k for k in registry if not registry[k].builtin) or 'none'}")
        for key in args.remove:
            db.delete_custom_topic(conn, key)
            n = db.forget_topic(conn, key)
            print(f"deleted custom topic {key} ({n} paper links released; the abstracts themselves are kept)")
        remaining = [k for k in db.enabled_topics(conn) if k not in args.remove]
        db.set_enabled_topics(conn, remaining or [topics.DEFAULT_TOPIC])
        print("enabled topics:", ", ".join(db.enabled_topics(conn)))
        conn.close()
        return

    selection = args.set
    if selection is None and args.interactive:
        print(topics.render_picker(registry, enabled=db.enabled_topics(conn), counts=counts))
        try:
            selection = topics.parse_selection(input("> ").strip(), registry)
        except topics.TopicError as e:
            sys.exit(str(e))
        except (EOFError, KeyboardInterrupt):
            sys.exit("cancelled")
    if selection is not None:
        try:
            keys = topics.resolve(selection, conn)
        except topics.TopicError as e:
            sys.exit(str(e))
        db.set_enabled_topics(conn, keys)
        print("enabled topics:", ", ".join(keys))
        missing = [p for p in topics.pairs(keys, None, conn) if p not in db.ingested_pairs(conn)]
        if missing:
            print(f"{len(missing)} topic/species pairs are not ingested yet - run `cora setup` to fetch them")
        conn.close()
        return

    print(topics.render_picker(registry, enabled=db.enabled_topics(conn), counts=counts))
    for key, t in registry.items():
        pairs = topics.pairs([key], None, conn)
        have = db.ingested_pairs(conn)
        n_ready = sum(1 for p in pairs if p in have)
        lineages = len({config.SPECIES_PHYLUM.get(s) if t.lineage_rank == "phylum" else config.SPECIES_CLASS.get(s) for s in t.species})
        print(f"\n{key}  [{t.status}{'' if t.builtin else ', custom'}]  {t.name}")
        print(f"  question    {t.question}")
        print(f"  species     {len(t.species)} ({n_ready}/{len(pairs)} ingested): {', '.join(t.species)}")
        print(f"  lineages    {lineages} distinct {t.lineage_rank}-level lineage{'s' if lineages != 1 else ''} in the panel"
              + ("  (convergence across lineages cannot be informative here)" if lineages < 2 else ""))
        print(f"  mechanisms  {len(t.mechanisms)}: {', '.join(t.mechanisms)}")
        print(f"  traits      {t.trait_source or 'none yet - no trait table backs this panel'}")
        if t.note:
            print(f"  note        {t.note}")
    conn.close()


def _add_topic(conn, args):
    """Add a custom topic from a JSON file, or interactively. Nothing is fetched here."""
    from . import topics as tp

    if args.file:
        import json as _json
        from pathlib import Path

        try:
            spec = tp.from_json(Path(args.file).read_text(encoding="utf-8"))
        except (OSError, _json.JSONDecodeError, tp.TopicError) as e:
            sys.exit(f"could not read a topic from {args.file}: {e}")
    else:
        if not sys.stdin.isatty():
            sys.exit("`cora topics --add` needs a terminal, or pass --file with a topic JSON file")
        print("A custom topic points Cora at a new question. Nothing is downloaded until you run `cora setup`.")
        print("Scope: animal traits -> human health. Cora is not a tool for making anything more dangerous.\n")
        try:
            key = input("key (lowercase, e.g. venom_analgesia): ").strip()
            name = input("name (e.g. Venom-derived painkillers): ").strip()
            question = input("one-line question: ").strip()
            print(f"\nknown species keys: {', '.join(config.SPECIES)}")
            species = [s for s in input("species (space-separated keys): ").replace(",", " ").split() if s]
            print(f"\nknown mechanisms: {', '.join(_graph_mechanisms())}")
            mechs = [m for m in input("mechanisms (space-separated keys): ").replace(",", " ").split() if m]
            print('\nPubMed terms, ANDed with each species query, e.g. (venom[Title/Abstract] OR analgesi*[Title/Abstract])')
            terms = input("terms: ").strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit("cancelled")
        unknown_sp = [s for s in species if s not in config.SPECIES]
        if unknown_sp:
            sys.exit(f"unknown species: {', '.join(unknown_sp)} (add them to cora/config.py first)")
        unknown_m = [m for m in mechs if m not in _graph_mechanisms()]
        if unknown_m:
            sys.exit(f"unknown mechanisms: {', '.join(unknown_m)}")
        try:
            spec = tp.custom_topic(key=key, name=name, question=question, terms=terms, species=species, mechanisms=mechs)
        except tp.TopicError as e:
            sys.exit(str(e))

    # Check every pair against PubMed before storing: a typo shows up here as 0 hits.
    print("\nchecking each species query against PubMed ...")
    from . import ingest

    zero = []
    for sk in spec.species:
        term = ingest.build_query(config.SPECIES[sk], spec)
        try:
            n = len(ingest.esearch(term, retmax=1000))
        except Exception as e:
            print(f"  {sk:<22} could not check ({type(e).__name__}: {e})")
            continue
        print(f"  {sk:<22} {n:>5} abstracts")
        if n == 0:
            zero.append(sk)
    if zero:
        print(f"\n{len(zero)} species returned nothing: {', '.join(zero)} - usually a misspelled binomial or terms that are too narrow.")
        if args.force:
            print("--force given: storing anyway.")
        elif sys.stdin.isatty():
            if input("store this topic anyway? [y/N] ").strip().lower() not in ("y", "yes"):
                sys.exit("not stored")
        else:
            # No terminal to ask at: storing a topic that finds nothing would look like it
            # worked and then return no cards. Refuse, and say how to override.
            sys.exit("not stored - fix the species or the terms, or re-run with --force")
    # Re-adding or editing a topic whose query changed must re-fetch: the abstracts already
    # stored were collected under the OLD terms, and `cora setup` skips any pair it has
    # already ingested, so nothing else would ever notice.
    previous = registry_of(conn).get(spec.key)
    changed = previous is not None and (previous.terms != spec.terms or set(previous.species) != set(spec.species))
    db.save_custom_topic(conn, spec.key, tp.to_json(spec))
    if changed:
        n = db.forget_ingest_state(conn, spec.key)
        print(f"\nits query changed, so {n} topic/species pair(s) are marked un-ingested and will be fetched again.")
    enabled = db.enabled_topics(conn) or [topics.DEFAULT_TOPIC]
    if spec.key not in enabled:
        enabled = enabled + [spec.key]
    db.set_enabled_topics(conn, enabled)
    print(f"\nstored custom topic {spec.key!r} and enabled it. Run `cora setup` to fetch its abstracts.")
    print(f"share it: cora topics --export {spec.key} > {spec.key}.json")


def registry_of(conn):
    return topics.all_topics(conn)


def _graph_mechanisms() -> list[str]:
    from . import graph

    return list(graph.MECHANISMS)


def cmd_serve(args):
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(db_path=getattr(args, "db", None)), host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cora", description="Cora - citation-gated hypothesis cards")
    p.add_argument("--db", help=f"SQLite path (default {config.DB_PATH})")
    sub = p.add_subparsers(dest="cmd", required=True)

    def topic_arg(parser):
        parser.add_argument("--topic", nargs="*", help="topic keys (default: the topics you enabled; see `cora topics`)")
        return parser

    s = topic_arg(sub.add_parser("setup", help="one command for a fresh install: pick topics, PubMed ingest, AnAge, graph build, canary check"))
    s.add_argument("--species", nargs="*", help="species keys (default: every species of the chosen topics)")
    s.add_argument("--refresh", action="store_true", help="re-fetch species and AnAge even if already present")
    s.add_argument("--skip", nargs="*", choices=["ingest", "anage", "graph", "canary"], help="steps to skip")
    s.add_argument("--verbose", action="store_true", help="show per-batch progress")
    s.set_defaults(fn=cmd_setup)

    sub.add_parser("init", help="create the database").set_defaults(fn=cmd_init)

    s = topic_arg(sub.add_parser("ingest", help="fetch ALL PubMed abstracts for each topic/species pair"))
    s.add_argument("--species", nargs="*", help="species keys (default: every species of the chosen topics)")
    s.set_defaults(fn=cmd_ingest)

    s = topic_arg(sub.add_parser("recheck", help="fetch only new abstracts, flag retracted/corrected sources, report what changed per card"))
    s.add_argument("--species", nargs="*")
    s.add_argument("--k", type=int, default=5, help="a new abstract 'touches' a card if it ranks in the top k for the card's query")
    s.add_argument("--report", action="store_true", help="print the last report without fetching")
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(fn=cmd_recheck)

    s = topic_arg(sub.add_parser("ask", help="draft one gated card for a query"))
    s.add_argument("query")
    s.add_argument("--species", nargs="*")
    s.add_argument("--mock", action="store_true", help="use the deterministic mock drafter and judge (no API key needed)")
    s.add_argument("--no-judge", action="store_true", help="skip the model judge; only the mechanical pattern check runs")
    s.add_argument("--judge-model", default=None, help=f"judge model (default {config.JUDGE_MODEL}; must differ from the drafter)")
    s.add_argument("--k", type=int, default=None, help="passages to retrieve")
    s.set_defaults(fn=cmd_ask)

    s = topic_arg(sub.add_parser("ledger", help="ranked cards"))
    s.add_argument("--all", action="store_true", help="include archived")
    s.add_argument("--limit", type=int, default=50)
    s.set_defaults(fn=cmd_ledger)

    s = sub.add_parser("show", help="expand a card (full evidence, what changed)")
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

    s = topic_arg(sub.add_parser("evalset", help="write UNLABELED verifier-gold candidate pairs for hand labeling"))
    s.add_argument("--n", type=int, default=60)
    s.add_argument("--out", default="eval/verifier_gold_candidates.jsonl")
    s.set_defaults(fn=cmd_evalset)

    s = sub.add_parser("label", help="blind-label the verifier gold candidates; --stats / --score-judge to score")
    s.add_argument("--file", default="eval/verifier_gold_candidates.jsonl")
    s.add_argument("--reveal", action="store_true", help="show the generator's kind/expected after each answer")
    s.add_argument("--relabel", action="store_true", help="revisit already-labeled pairs too")
    s.add_argument("--stats", action="store_true", help="print label counts and the mechanical-gate baseline")
    s.add_argument("--score-judge", action="store_true", help="score the model judge against the labels (implies --stats)")
    s.add_argument("--judge-model", default=None)
    s.add_argument("--mock", action="store_true", help="score the mock judge instead (no API key needed)")
    s.set_defaults(fn=cmd_label)

    s = sub.add_parser("anage", help="download AnAge (HAGR, CC BY 3.0) lifespan records and load them; prints the panel table")
    s.add_argument("--file", default=None, help="load a local anage_data.txt instead of downloading")
    s.set_defaults(fn=cmd_anage)

    topic_arg(sub.add_parser("species", help="the panel: docs per species, AnAge max longevity with quality/sample/origin, naive LQ")).set_defaults(fn=cmd_species)

    g = sub.add_parser("graph", help="graph-lite: extract findings and compute mechanism-level convergence across the panel")
    gsub = g.add_subparsers(dest="graph_cmd", required=True)
    gb = topic_arg(gsub.add_parser("build", help="extract findings from every abstract (lexicon by default; --model uses the model extractor)"))
    gb.add_argument("--species", nargs="*")
    gb.add_argument("--model", action="store_true", help="use the model extractor (needs credentials); quotes still go through the gate")
    gb.add_argument("--extract-model", default=None, help=f"model for --model (default {config.JUDGE_MODEL})")
    gc = topic_arg(gsub.add_parser("converge", help="mechanisms ranked by independent lineages, with a permutation p-value"))
    gc.add_argument("--min-tier", default="mention", choices=["mention", "association", "intervention"])
    gc.add_argument("--min-lineages", type=int, default=2)
    gc.add_argument("--perms", type=int, default=None, help=f"permutations (default {config.GRAPH_PERMUTATIONS})")
    gc.add_argument("--extractor", default=None, help="restrict to one extractor's findings")
    gs = topic_arg(gsub.add_parser("show", help="every finding for one mechanism, per species, with its verbatim quote"))
    gs.add_argument("mechanism")
    gs.add_argument("--extractor", default=None)
    g.set_defaults(fn=cmd_graph)

    s = sub.add_parser("topics", help="list, choose, add or remove research topics")
    s.add_argument("--set", nargs="*", help="enable exactly these topics (keys or numbers)")
    s.add_argument("-i", "--interactive", action="store_true", help="show the picker and ask")
    s.add_argument("--add", action="store_true", help="add a custom topic (interactive, or with --file)")
    s.add_argument("--file", help="with --add: a topic JSON file (as written by --export)")
    s.add_argument("--force", action="store_true", help="with --add: store the topic even if a species returns no abstracts")
    s.add_argument("--export", help="print one topic as JSON, to share or edit")
    s.add_argument("--remove", nargs="*", help="delete custom topics by key")
    s.set_defaults(fn=cmd_topics)

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
